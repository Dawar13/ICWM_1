"""G2 training: tiny pixel JEPA world model in the sprite world, with probes logged during training.

Usage:
  python src/g2_pixel/train.py --out runs/g2/<name> --cfg '{"idm": "warmup", "seed": 0}'

Everything that decides a go/no-go is logged to runs/.../log.jsonl every eval_every steps:
  agent_r2       linear probe: agent position from z_t (ground truth from the simulator)
  dist_r2        linear probe: distractor positions from z_t
  action_dec     R^2 of a_t from (z_t, z_{t+1})
  cas            counterfactual action sensitivity of the predictor
  pred_loss, copy_loss   prediction error vs simply copying z_t forward (SLIM's "no better than copying")
  cov_err        how far latents are from whitened (SIGReg working?)
"""
import argparse
import copy
import dataclasses
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.gpu import gpu_report, peak_mem_gb, pick_device  # noqa: E402
from src.common.metrics import action_decodability, probe_r2  # noqa: E402
from src.common.runlog import RunLog, already_done  # noqa: E402
from src.g2_pixel.env import EnvConfig, SpriteWorld  # noqa: E402
from src.g2_pixel.model import TinyJEPA, sigreg_epps_pulley  # noqa: E402


@dataclass
class TrainConfig:
    # model
    d: int = 32
    h: int = 1
    enc_width: int = 32
    hdim: int = 256
    cond: str = "film"            # "film" (AdaLN-like) or "concat"
    act_init_scale: float = 1.0
    enc_bn: bool = True           # batch-normalised projector output (check against LeWM)
    # objective
    reg: str = "sigreg"           # "sigreg" (end-to-end) or "ema"
    sigreg_w: float = 0.1         # CALIBRATE (prereg/G2.md) or take LeWM's value with LeWM's exact form
    target_grad: bool = True      # LeWM-style end-to-end; verify against LeWM code
    ema_tau: float = 0.996
    idm: str = "none"             # "none", "always", "warmup"
    idm_w: float = 1.0
    warmup_frac: float = 0.05
    hist_dropout: float = 0.0     # prob. of replacing older context latents with copies of z_t
    # data
    dataset_episodes: int = 2000  # 0 = fresh samples every step (infinite data)
    ep_len: int = 40
    # optimisation
    steps: int = 15000
    batch: int = 128
    lr: float = 3e-4
    wd: float = 0.0
    eval_every: int = 250
    eval_n: int = 2048
    seed: int = 0
    max_mem_gb: float = 6.0       # cap on the shared GPU
    save_ckpt: bool = True


def split_cfg(d):
    env_keys = {f.name for f in fields(EnvConfig)}
    env = EnvConfig(**{k: v for k, v in d.items() if k in env_keys})
    tr = TrainConfig(**{k: v for k, v in d.items() if k not in env_keys})
    return env, tr


class Data:
    """Fixed offline dataset of episodes (states and actions only), rendered on the fly."""

    def __init__(self, world: SpriteWorld, tc: TrainConfig, dev):
        self.w, self.tc, self.dev = world, tc, dev
        if tc.dataset_episodes > 0:
            chunks = [world.rollout(min(500, tc.dataset_episodes - i), tc.ep_len)
                      for i in range(0, tc.dataset_episodes, 500)]
            self.ep = {k: torch.cat([c[k] for c in chunks], 0) for k in chunks[0]}
        self.g = torch.Generator().manual_seed(tc.seed + 777)

    def windows(self, B, gen=None):
        """Return a window batch: dict with p, q (B, h+1, ...), a_t (B, 2)."""
        h = self.tc.h
        g = gen or self.g
        if self.tc.dataset_episodes > 0:
            n_ep = self.ep["p"].shape[0]
            ei = torch.randint(0, n_ep, (B,), generator=g).to(self.dev)
            t0 = torch.randint(0, self.tc.ep_len - h, (B,), generator=g).to(self.dev)
            idx = t0[:, None] + torch.arange(h + 1, device=self.dev)[None]
            p = self.ep["p"][ei[:, None], idx]
            q = self.ep["q"][ei[:, None], idx]
            a = self.ep["a"][ei, t0 + h - 1]
        else:
            roll = self.w.rollout(B, h)
            p, q, a = roll["p"], roll["q"], roll["a"][:, h - 1]
        return {"p": p, "q": q, "a": a}

    def frames(self, win):
        return self.w.render_seq({"p": win["p"], "q": win["q"]})


def alt_actions(K, B, env_c: EnvConfig, dev, gen):
    return (torch.randn(K, B, 2, generator=gen) * env_c.act_std).clamp(-1, 1).to(dev)


@torch.no_grad()
def evaluate(model, data: Data, ev_wins, env_c, tc, dev):
    model.eval()
    zs, zn, ps, qs, acts, cas_num = [], [], [], [], [], []
    pred_l, copy_l = 0.0, 0.0
    gen = torch.Generator().manual_seed(4242)
    for i in range(0, ev_wins["p"].shape[0], 256):
        win = {k: v[i:i + 256] for k, v in ev_wins.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z = model.encode_seq(data.frames(win)).float()
            z_hist, z_t, z_next = z[:, :tc.h], z[:, tc.h - 1], z[:, tc.h]
            pred = model.pred(z_hist, win["a"]).float()
            alts = alt_actions(8, z.shape[0], env_c, dev, gen)
            preds = torch.stack([model.pred(z_hist, alts[k]).float() for k in range(8)], 0)
        cas_num.append(preds.var(0).sum(-1))
        pred_l += float(((pred - z_next) ** 2).sum(-1).sum())
        copy_l += float(((z_t - z_next) ** 2).sum(-1).sum())
        zs.append(z_t); zn.append(z_next); ps.append(win["p"][:, tc.h - 1])
        qs.append(win["q"][:, tc.h - 1].flatten(1)); acts.append(win["a"])
    z_t, z_next = torch.cat(zs), torch.cat(zn)
    n = z_t.shape[0]
    cov = torch.cov(z_t.T)
    out = {
        "agent_r2": probe_r2(z_t, torch.cat(ps)),
        "dist_r2": probe_r2(z_t, torch.cat(qs)) if env_c.n_dist > 0 else None,
        "action_dec": action_decodability(z_t, z_next, torch.cat(acts)),
        "cas": float(torch.cat(cas_num).mean() / z_next.var(0).sum().clamp_min(1e-12)),
        "pred_loss": pred_l / n,
        "copy_loss": copy_l / n,
        "cov_err": float((cov - torch.eye(cov.shape[0], device=cov.device)).abs().mean()),
        "z_var": float(z_t.var(0).sum()),
    }
    model.train()
    return out


def idm_weight(step, tc: TrainConfig):
    if tc.idm == "none":
        return 0.0
    if tc.idm == "always":
        return tc.idm_w
    if tc.idm == "warmup":
        return tc.idm_w if step < tc.warmup_frac * tc.steps else 0.0
    raise ValueError(tc.idm)


def run(cfg_dict: dict, out_dir: str):
    if already_done(out_dir):
        print("skip (done):", out_dir)
        return
    env_c, tc = split_cfg(cfg_dict)
    print(gpu_report())
    dev = pick_device(max_mem_gb=tc.max_mem_gb)
    torch.manual_seed(tc.seed)
    world = SpriteWorld(env_c, dev, seed=tc.seed)
    data = Data(world, tc, dev)
    ev_world = SpriteWorld(env_c, dev, seed=10_000 + tc.seed)
    ev_data = Data(ev_world, dataclasses.replace(tc, dataset_episodes=512), dev)  # fixed held-out episodes
    ev_wins = ev_data.windows(tc.eval_n, gen=torch.Generator().manual_seed(99))
    model = TinyJEPA(tc.d, tc.h, tc.enc_width, tc.hdim, tc.cond, tc.act_init_scale, env_c.img, tc.enc_bn).to(dev)
    ema_enc = copy.deepcopy(model.enc).requires_grad_(False) if tc.reg == "ema" else None
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.wd)
    log = RunLog(out_dir, {"env": asdict(env_c), "train": asdict(tc)})
    log.log(kind="setup", response_fraction=world.response_fraction(),
            n_params=sum(p.numel() for p in model.parameters()), gpu=gpu_report())
    t_start = time.time()
    warm_end, warm_saved = int(tc.warmup_frac * tc.steps), False
    for step in range(tc.steps + 1):
        if step % tc.eval_every == 0 or step == tc.steps:
            m = evaluate(model, data, ev_wins, env_c, tc, dev)
            if not math.isfinite(m["pred_loss"]):
                log.finish(status="diverged", step=step)
                return
            log.log(kind="eval", step=step, idm_on=idm_weight(step, tc) > 0,
                    steps_per_s=step / max(1e-6, time.time() - t_start), peak_mem_gb=peak_mem_gb(), **m)
            if tc.idm == "warmup" and tc.save_ckpt and not warm_saved and step >= warm_end:
                torch.save(model.state_dict(), Path(out_dir) / "ckpt_end_of_warmup.pt")
                warm_saved = True
        if step == tc.steps:
            break
        win = data.windows(tc.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            frames = data.frames(win)
            z = model.encode_seq(frames)
            z_hist, z_t = z[:, :tc.h], z[:, tc.h - 1]
            if tc.hist_dropout > 0 and tc.h > 1:
                drop = (torch.rand(z.shape[0], 1, 1, device=dev) < tc.hist_dropout)
                z_hist = torch.where(drop, z_t[:, None].expand_as(z_hist), z_hist)
            if tc.reg == "ema":
                with torch.no_grad():
                    target = ema_enc(frames[:, tc.h])
            else:
                target = z[:, tc.h] if tc.target_grad else z[:, tc.h].detach()
            pred = model.pred(z_hist, win["a"])
            loss_pred = ((pred.float() - target.float()) ** 2).sum(-1).mean()
            loss = loss_pred
            if tc.reg == "sigreg":
                loss = loss + tc.sigreg_w * sigreg_epps_pulley(z.flatten(0, 1).float())
            w = idm_weight(step, tc)
            if w > 0:
                a_hat = model.idm(z_t, z[:, tc.h])
                loss = loss + w * ((a_hat.float() - win["a"]) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema_enc is not None:
            with torch.no_grad():
                for pe, po in zip(ema_enc.parameters(), model.enc.parameters()):
                    pe.mul_(tc.ema_tau).add_(po.detach(), alpha=1 - tc.ema_tau)
    final = evaluate(model, data, ev_wins, env_c, tc, dev)
    if tc.save_ckpt:
        torch.save(model.state_dict(), Path(out_dir) / "ckpt_final.pt")
    log.finish(status="ok", wall_train_s=time.time() - t_start, peak_mem_gb=peak_mem_gb(), **final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="{}", help="JSON dict of EnvConfig and TrainConfig overrides")
    a = ap.parse_args()
    run(json.loads(a.cfg), a.out)


if __name__ == "__main__":
    main()
