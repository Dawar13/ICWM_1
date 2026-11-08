"""Planning check for G2 checkpoints: CEM-MPC in latent space towards a goal image.

Protocol (fixed; write it next to every number):
- Goal: agent at a random position; the goal image shows the distractors at their *current* true
  positions (the goal only specifies the agent; distractors are not controllable).
- CEM: horizon H, S samples, E elites, I iterations, terminal latent L2 cost; execute the first
  action, replan every step (MPC). Budget: `budget` env steps.
- Success: final agent distance to goal < tol (fraction of frame width).
Usage: python src/g2_pixel/plan_eval.py --run runs/g2/<name> [--episodes 100]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.gpu import pick_device  # noqa: E402
from src.g2_pixel.env import EnvConfig, SpriteWorld  # noqa: E402
from src.g2_pixel.model import TinyJEPA  # noqa: E402
from src.g2_pixel.train import TrainConfig  # noqa: E402


def load(run_dir, dev, ckpt="ckpt_final.pt"):
    meta = json.loads((Path(run_dir) / "meta.json").read_text())["config"]
    env_c, tc = EnvConfig(**meta["env"]), TrainConfig(**meta["train"])
    model = TinyJEPA(tc.d, tc.h, tc.enc_width, tc.hdim, tc.cond, tc.act_init_scale, env_c.img, tc.enc_bn).to(dev)
    model.load_state_dict(torch.load(Path(run_dir) / ckpt, map_location=dev))
    return model.eval(), env_c, tc


@torch.no_grad()
def plan_eval(run_dir, episodes=100, H=5, S=256, E=32, iters=3, budget=None, tol=0.06, seed=0,
              ckpt="ckpt_final.pt", max_mem_gb=6.0):
    dev = pick_device(max_mem_gb=max_mem_gb)
    model, env_c, tc = load(run_dir, dev, ckpt)
    world = SpriteWorld(env_c, dev, seed=50_000 + seed)
    g = torch.Generator().manual_seed(60_000 + seed)
    B = episodes
    if budget is None:
        budget = 4 if env_c.agent_mode == "set" else int(math.ceil(0.9 / env_c.agent_speed)) + 5
    st = world.reset(B)
    lo, hi = env_c.margin + 0.1, 1 - env_c.margin - 0.1
    p_goal = (torch.rand(B, 2, generator=g) * (hi - lo) + lo).to(dev)
    frames = []
    for _ in range(tc.h):  # fill the history with random actions
        frames.append(world.render(st))
        st = world.step(st, world.sample_action(st))
    for _ in range(budget):
        frames.append(world.render(st))
        frames = frames[-tc.h:]
        hist = torch.stack(frames, 1)
        z_hist = model.encode_seq(hist)
        z_goal = model.enc(world.render({"p": p_goal, "q": st["q"]}))
        mu = torch.zeros(B, H, 2, device=dev)
        sd = torch.full((B, H, 2), 0.6, device=dev)
        for _ in range(iters):
            acts = (mu[:, None] + sd[:, None] * torch.randn(B, S, H, 2, generator=g).to(dev)).clamp(-1, 1)
            zh = z_hist[:, None].expand(B, S, *z_hist.shape[1:]).reshape(B * S, *z_hist.shape[1:])
            for k in range(H):
                z_next = model.pred(zh, acts[:, :, k].reshape(B * S, 2))
                zh = torch.cat([zh[:, 1:], z_next[:, None]], 1)
            cost = ((zh[:, -1] - z_goal.repeat_interleave(S, 0)) ** 2).sum(-1).view(B, S)
            idx = cost.topk(min(E, max(2, S // 8)), dim=1, largest=False).indices
            elite = torch.gather(acts, 1, idx[:, :, None, None].expand(B, idx.shape[1], H, 2))
            mu, sd = elite.mean(1), elite.std(1).clamp_min(0.05)
        st = world.step(st, mu[:, 0])
    dist = (st["p"] - p_goal).norm(dim=-1)
    out = {"success": float((dist < tol).float().mean()), "mean_final_dist": float(dist.mean()),
           "protocol": {"episodes": B, "H": H, "S": S, "E": E, "iters": iters, "budget": budget, "tol": tol,
                        "seed": seed, "ckpt": ckpt}}
    (Path(run_dir) / f"plan_eval_{Path(ckpt).stem}_s{seed}.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="ckpt_final.pt")
    ap.add_argument("--S", type=int, default=256)
    a = ap.parse_args()
    print(plan_eval(a.run, a.episodes, S=a.S, seed=a.seed, ckpt=a.ckpt))


if __name__ == "__main__":
    main()
