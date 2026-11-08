"""G1. Linear action-conditioned JEPA world model, trained by gradient descent on fresh samples.

Plain-language summary
----------------------
A hidden world has an *agent* (moved by actions) and *environment* factors (move on their own).
We observe a random linear mix of them plus noise. A linear encoder W maps observations to a
k-dimensional latent; a linear predictor maps the last h latents plus the action to the next
latent. A covariance penalty (the Gaussian case of SIGReg) stops collapse. We watch, during
training, when the latent starts to carry the agent and when the predictor starts to use the
action. Optional: an inverse-dynamics (IDM) head, always on or only as a warm-up.

Why linear: the predictions of "Listening Is Learned Last" are about training dynamics, which
are only derivable in the linear regime. The pixel experiment (G2) checks them in a nonlinear model.

Nothing here is a result. It is an instrument. Pre-register before sweeping (prereg/G1.md).
"""
import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.metrics import action_decodability, probe_r2  # noqa: E402
from src.common.runlog import RunLog, already_done  # noqa: E402


@dataclass
class G1Config:
    # world
    D: int = 32                 # observation dimension
    d_a: int = 1                # agent factors (one action dimension per agent factor)
    d_e: int = 4                # environment factors
    agent_mode: str = "integrator"   # "integrator" (exogenous actions) or "unstable_cl" (closed loop)
    rho_a: float = 0.0          # agent persistence. 0 = action sets the agent state (looks like noise without the action);
                                # 0.97 = persistent integrator. Smoke tests (notes/smoke_tests.md) found the trap only at low rho_a
    b: float = 1.0              # action gain on the agent
    g_a: float = 0.2            # observation gain of the agent (how much of the "frame" it occupies)
    agent_noise: float = 0.0    # agent process noise (adds variance that actions do not explain)
    rho_u: float = 1.05         # unstable_cl: open-loop agent multiplier (>1 means unstable)
    k_fb: float = 0.4           # unstable_cl: feedback gain, a = -k_fb * s + explore
    explore: float = 0.5        # unstable_cl: exploration noise on the action
    rho_e: list = field(default_factory=lambda: [0.99, 0.97, 0.95, 0.9])
    g_e: float = 1.0            # observation gain of environment factors
    obs_noise: float = 0.01
    phi: float = 0.0            # lag-one autocorrelation of exogenous actions
    # model
    k: int = 4                  # latent dimension. k < d_a + d_e forces competition for capacity (needed for any failure)
    h: int = 1                  # history length used by the predictor
    act_depth: int = 2          # 1: V a.  2: V2 V1 a (factorised action pathway; can create a saddle)
    act_width: int = 8
    act_init: float = 1e-3      # std of action-pathway weights at init
    enc_init: float = 1.0       # encoder init std, in units of 1/sqrt(D)
    reg: str = "cov"            # "cov" (Gaussian SIGReg), "sigreg" (Epps-Pulley), "ema" (stop-grad EMA target)
    lam: float = 1.0
    ema_tau: float = 0.99
    target_grad: bool = True    # end-to-end gradient into the target (LeWM style). Ignored for "ema".
    # inverse dynamics
    idm: str = "none"           # "none", "always", "warmup"
    idm_weight: float = 10.0     # weight 1 was too weak to hold the agent in smoke tests
    warmup_frac: float = 0.1
    # optimisation
    steps: int = 20000
    batch: int = 1024
    lr: float = 0.05
    clip: float = 10.0
    stride: int = 3             # environment steps between updates (decorrelates the stream)
    burn_in: int = 300
    eval_every: int = 100
    eval_n: int = 8192
    seed: int = 0
    device: str = "cpu"


class LinearWorld:
    """Batched simulator of B independent chains."""

    def __init__(self, cfg: G1Config, B: int, gen: torch.Generator):
        self.c, self.B, self.g = cfg, B, gen
        dev = cfg.device
        d_tot = cfg.d_a + cfg.d_e
        q, _ = torch.linalg.qr(torch.randn(cfg.D, d_tot, generator=gen))  # orthonormal mixing columns
        self.G_a = (q[:, :cfg.d_a] * cfg.g_a).to(dev)
        self.G_e = (q[:, cfg.d_a:] * cfg.g_e).to(dev)
        self.rho_e = torch.tensor(cfg.rho_e[:cfg.d_e], dtype=torch.float32, device=dev)
        assert self.rho_e.numel() == cfg.d_e, "rho_e must list one value per environment factor"
        self.s_a = torch.zeros(B, cfg.d_a, device=dev)
        self.s_e = torch.zeros(B, cfg.d_e, device=dev)
        self.a_prev = torch.zeros(B, cfg.d_a, device=dev)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.g).to(self.c.device)

    def step(self):
        c = self.c
        if c.agent_mode == "integrator":
            a = c.phi * self.a_prev + math.sqrt(max(0.0, 1 - c.phi ** 2)) * self._randn(self.B, c.d_a)
            s_a_next = c.rho_a * self.s_a + c.b * a
        elif c.agent_mode == "unstable_cl":
            a = -c.k_fb * self.s_a + c.explore * self._randn(self.B, c.d_a)
            s_a_next = c.rho_u * self.s_a + c.b * a
        else:
            raise ValueError(c.agent_mode)
        if c.agent_noise > 0:
            s_a_next = s_a_next + c.agent_noise * self._randn(self.B, c.d_a)
        s_e_next = self.rho_e * self.s_e + torch.sqrt(1 - self.rho_e ** 2) * self._randn(self.B, c.d_e)
        out = (self.s_a, self.s_e, a)
        self.s_a, self.s_e, self.a_prev = s_a_next, s_e_next, a
        return out  # state at t and the action applied at t

    def observe(self, s_a, s_e):
        x = s_a @ self.G_a.T + s_e @ self.G_e.T
        return x + self.c.obs_noise * self._randn(*x.shape)

    def windows(self, n_steps_between: int):
        """Advance and return one training window: obs x_{t-h+1..t+1}, action a_t, states at t."""
        c = self.c
        for _ in range(n_steps_between):
            self.step()
        xs, s_as, s_es = [], [], []
        a_t = None
        for i in range(c.h + 1):
            s_a, s_e, a = self.step()
            xs.append(self.observe(s_a, s_e))
            s_as.append(s_a)
            s_es.append(s_e)
            if i == c.h - 1:
                a_t = a  # action taken at the last context step t
        x = torch.stack(xs, 1)  # (B, h+1, D)
        return x, a_t, s_as[c.h - 1], s_es[c.h - 1]


class LinearJEPA(torch.nn.Module):
    def __init__(self, cfg: G1Config, gen: torch.Generator):
        super().__init__()
        c = cfg
        self.c = c
        self.W = torch.nn.Parameter(torch.randn(c.k, c.D, generator=gen) * c.enc_init / math.sqrt(c.D))
        self.A = torch.nn.Parameter(torch.zeros(c.h, c.k, c.k))
        if c.act_depth == 1:
            self.V = torch.nn.Parameter(torch.randn(c.k, c.d_a, generator=gen) * c.act_init)
        elif c.act_depth == 2:
            self.V1 = torch.nn.Parameter(torch.randn(c.act_width, c.d_a, generator=gen) * c.act_init)
            self.V2 = torch.nn.Parameter(torch.randn(c.k, c.act_width, generator=gen) * c.act_init)
        else:
            raise ValueError("act_depth must be 1 or 2")
        self.U = torch.nn.Parameter(torch.randn(c.d_a, 2 * c.k, generator=gen) * 0.1)  # IDM head
        self.register_buffer("W_ema", self.W.detach().clone())

    def V_eff(self):
        return self.V if self.c.act_depth == 1 else self.V2 @ self.V1

    def encode(self, x, ema=False):
        W = self.W_ema if ema else self.W
        return x @ W.T

    def predict(self, z_hist, a):
        """z_hist: (B, h, k) ordered oldest..newest. Returns predicted z_{t+1}."""
        out = a @ self.V_eff().T
        h = self.c.h
        for j in range(h):  # A[j] multiplies z_{t-j}
            out = out + z_hist[:, h - 1 - j] @ self.A[j].T
        return out


def epps_pulley_sigreg(z, n_proj=64, t_max=3.0, n_t=17, gen=None):
    """SIGReg-style test of z against N(0, I) along random 1-D projections (characteristic functions)."""
    d = z.shape[1]
    u = torch.randn(d, n_proj, device=z.device, generator=gen)
    u = u / u.norm(dim=0, keepdim=True)
    x = z @ u
    t = torch.linspace(-t_max, t_max, n_t, device=z.device)
    xt = x.unsqueeze(-1) * t
    re, im = torch.cos(xt).mean(0), torch.sin(xt).mean(0)
    w = torch.exp(-0.5 * t ** 2)
    return (((re - w) ** 2 + im ** 2) * w).mean()


def regulariser(z, c: G1Config):
    if c.reg in ("cov", "ema"):
        zc = z - z.mean(0, keepdim=True)
        cov = zc.T @ zc / (z.shape[0] - 1)
        eye = torch.eye(z.shape[1], device=z.device)
        return ((cov - eye) ** 2).sum() + (z.mean(0) ** 2).sum()
    if c.reg == "sigreg":
        return epps_pulley_sigreg(z)
    raise ValueError(c.reg)


def idm_active(step: int, c: G1Config) -> float:
    if c.idm == "none":
        return 0.0
    if c.idm == "always":
        return c.idm_weight
    if c.idm == "warmup":
        return c.idm_weight if step < c.warmup_frac * c.steps else 0.0
    raise ValueError(c.idm)


def analytic_rankings(c: G1Config, world: LinearWorld, n=200000, gen=None):
    """Normalised one-step prediction error of each factor (lower = more attractive to a whitened
    objective), with and without knowing the action. Estimated by simulation, history 1."""
    sim = LinearWorld(c, 4096, gen if gen is not None else torch.Generator().manual_seed(123))
    sim.G_a, sim.G_e = world.G_a, world.G_e
    for _ in range(c.burn_in):
        sim.step()
    S_a, S_e, A, S_a1, S_e1 = [], [], [], [], []
    for _ in range(max(1, n // 4096)):
        s_a, s_e, a = sim.step()
        S_a.append(s_a); S_e.append(s_e); A.append(a); S_a1.append(sim.s_a); S_e1.append(sim.s_e)
    s_a, s_e, a = torch.cat(S_a), torch.cat(S_e), torch.cat(A)
    s_a1, s_e1 = torch.cat(S_a1), torch.cat(S_e1)

    def nerr(y, X):
        X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1).double()
        w = torch.linalg.lstsq(X, y.double()).solution
        return float(((y.double() - X @ w) ** 2).mean() / y.double().var())

    out = {"agent_var": float(s_a.var()), "agent_static_energy": float(c.g_a ** 2 * s_a.var()),
           "agent_response_energy": float(c.g_a ** 2 * (c.b * a).var())}
    out["agent_nerr_passive"] = nerr(s_a1[:, 0:1], s_a[:, 0:1])
    out["agent_nerr_with_action"] = nerr(s_a1[:, 0:1], torch.cat([s_a[:, 0:1], a[:, 0:1]], 1))
    out["env_nerr"] = [nerr(s_e1[:, i:i + 1], s_e[:, i:i + 1]) for i in range(c.d_e)]
    return out


@torch.no_grad()
def evaluate(model: LinearJEPA, ev, c: G1Config, gen):
    x, a, s_a, s_e = ev
    z_all = model.encode(x)  # (N, h+1, k)
    z_hist, z_t, z_next = z_all[:, :c.h], z_all[:, c.h - 1], z_all[:, c.h]
    pred = model.predict(z_hist, a)
    out = {
        "agent_r2": probe_r2(z_t, s_a),
        "env_r2": [probe_r2(z_t, s_e[:, i:i + 1]) for i in range(c.d_e)],
        "action_dec": action_decodability(z_t, z_next, a),
        "pred_loss": float(((pred - z_next) ** 2).sum(-1).mean()),
        "copy_loss": float(((z_t - z_next) ** 2).sum(-1).mean()),
        "z_var": float(z_t.var(0).sum()),
    }
    # counterfactual action sensitivity: variance over alternative actions / variance of true next latent
    K = 8
    a_alt = torch.randn(K, *a.shape, generator=gen).to(a.device)
    preds = torch.stack([model.predict(z_hist, a_alt[i]) for i in range(K)], 0)
    out["cas"] = float(preds.var(0).sum(-1).mean() / z_next.var(0).sum().clamp_min(1e-12))
    # alignment of the action pathway with the encoder's view of the true action effect
    V_eff = model.V_eff()
    ideal = c.b * (model.W @ model.G_a)  # true effect of the action on z_{t+1}, in the current latent
    out["act_gain"] = float(V_eff.norm())
    out["act_rel_err"] = float((V_eff - ideal).norm() / ideal.norm().clamp_min(1e-12))
    return out


def run(c: G1Config, out_dir: str):
    if already_done(out_dir):
        print("skip (done):", out_dir)
        return
    torch.manual_seed(c.seed)
    gen = torch.Generator().manual_seed(c.seed)
    world = LinearWorld(c, c.batch, gen)
    model = LinearJEPA(c, gen).to(c.device)
    model.register_buffer("G_a", world.G_a.clone())  # true agent direction in observation space
    opt = torch.optim.SGD(model.parameters(), lr=c.lr)
    for _ in range(c.burn_in):
        world.step()
    # fixed evaluation set from an independent stream with the same mixing matrix
    eval_gen = torch.Generator().manual_seed(10_000 + c.seed)
    ev_world = LinearWorld(c, c.eval_n, eval_gen)
    ev_world.G_a, ev_world.G_e = world.G_a, world.G_e
    for _ in range(c.burn_in):
        ev_world.step()
    ev = ev_world.windows(1)
    log = RunLog(out_dir, asdict(c))
    log.log(kind="rankings", **analytic_rankings(c, world))
    for step in range(c.steps + 1):
        if step % c.eval_every == 0:
            m = evaluate(model, ev, c, eval_gen)
            if not all(math.isfinite(v) for v in [m["pred_loss"], m["agent_r2"], m["z_var"]]):
                log.finish(status="diverged", step=step)
                return
            log.log(kind="eval", step=step, idm_on=idm_active(step, c) > 0, **m)
        if step == c.steps:
            break
        x, a, _, _ = world.windows(c.stride)
        z_all = model.encode(x)
        z_hist, z_t = z_all[:, :c.h], z_all[:, c.h - 1]
        if c.reg == "ema":
            target = model.encode(x[:, c.h], ema=True).detach()
        elif c.target_grad:
            target = z_all[:, c.h]
        else:
            target = z_all[:, c.h].detach()
        pred = model.predict(z_hist, a)
        loss = ((pred - target) ** 2).sum(-1).mean() + c.lam * regulariser(z_t, c)
        w_idm = idm_active(step, c)
        if w_idm > 0:
            a_hat = torch.cat([z_t, z_all[:, c.h]], 1) @ model.U.T
            loss = loss + w_idm * ((a_hat - a) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
        opt.step()
        if c.reg == "ema":
            with torch.no_grad():
                model.W_ema.mul_(c.ema_tau).add_((1 - c.ema_tau) * model.W)
    final = evaluate(model, ev, c, eval_gen)
    log.finish(status="ok", **final)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cfg", default="{}", help="JSON dict of G1Config overrides")
    args = p.parse_args()
    c = G1Config(**json.loads(args.cfg))
    run(c, args.out)


if __name__ == "__main__":
    main()
