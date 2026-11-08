"""G2 environment: a batched, GPU-rendered sprite world.

- One controlled agent sprite (red). Two control modes:
    "integrator": p' = p + speed * a   (persistent position, like PushT / TwoRoom agents)
    "set":        p' = centre + span * a  (the action sets the position; looks like noise without the action)
- n_echo echo sprites (also red) that copy the agent's motion at fixed offsets. They raise the
  fraction of the frame that responds to actions without changing the agent's size. This separates
  "how big the agent is" from "how much of the frame responds", which SLIM (2610.03137) found to matter.
- n_dist autonomous distractor balls (green) that bounce at constant speed (highly predictable).
- A fixed, static background texture (blue).
- Actions: 2-D, AR(1) with lag-one autocorrelation phi, clipped to [-1, 1].

States are stored, frames are rendered on the fly, so a fixed offline dataset costs almost no memory.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class EnvConfig:
    img: int = 64
    agent_mode: str = "integrator"   # "integrator" or "set"
    agent_r: float = 2.5             # agent radius in pixels
    agent_speed: float = 0.04        # integrator: displacement per unit action, as a fraction of the frame
    set_span: float = 0.35           # set mode: half-range of positions reachable
    n_echo: int = 0                  # echo sprites copying the agent's motion
    n_dist: int = 2                  # autonomous distractor balls
    dist_r: float = 5.0
    dist_speed: float = 0.03
    phi: float = 0.0                 # action autocorrelation
    act_std: float = 0.5             # std of the AR(1) action process before clipping
    bg_amp: float = 0.3              # static background amplitude
    soft: float = 0.6                # edge softness in pixels
    margin: float = 0.06


class SpriteWorld:
    def __init__(self, cfg: EnvConfig, device, seed: int = 0):
        self.c = cfg
        self.dev = device
        g = torch.Generator().manual_seed(1234)  # background and echo offsets are fixed across runs
        ys, xs = torch.meshgrid(torch.linspace(0, 1, cfg.img), torch.linspace(0, 1, cfg.img), indexing="ij")
        self.grid = torch.stack([xs, ys], -1).to(device)  # (H, W, 2)
        bg = torch.nn.functional.avg_pool2d(torch.rand(1, 1, cfg.img, cfg.img, generator=g), 5, 1, 2)
        self.bg = (bg[0, 0] - bg.mean()).to(device) * cfg.bg_amp * 4
        self.echo_off = torch.rand(max(cfg.n_echo, 1), 2, generator=g).to(device) * 0.8 + 0.1
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    # ---------- dynamics ----------
    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen).to(self.dev)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen).to(self.dev)

    def reset(self, B):
        c = self.c
        st = {
            "p": self._rand(B, 2) * (1 - 2 * c.margin) + c.margin,
            "q": self._rand(B, c.n_dist, 2) * (1 - 2 * c.margin) + c.margin,
        }
        ang = self._rand(B, c.n_dist) * 2 * math.pi
        st["v"] = torch.stack([ang.cos(), ang.sin()], -1) * c.dist_speed
        st["a_prev"] = self._randn(B, 2) * c.act_std  # stationary start for the AR(1) process
        return st

    def sample_action(self, st):
        c = self.c
        a = c.phi * st["a_prev"] + math.sqrt(max(0.0, 1 - c.phi ** 2)) * self._randn(*st["a_prev"].shape) * c.act_std
        return a

    def step(self, st, a):
        """Apply action a (B, 2). Returns the new state. a is clipped to [-1, 1]."""
        c = self.c
        a_c = a.clamp(-1, 1)
        if c.agent_mode == "integrator":
            p = (st["p"] + c.agent_speed * a_c).clamp(c.margin, 1 - c.margin)
        elif c.agent_mode == "set":
            p = (0.5 + c.set_span * a_c).clamp(c.margin, 1 - c.margin)
        else:
            raise ValueError(c.agent_mode)
        q, v = st["q"] + st["v"], st["v"].clone()
        lo, hi = c.margin, 1 - c.margin
        hit_lo, hit_hi = q < lo, q > hi
        v = torch.where(hit_lo | hit_hi, -v, v)
        q = torch.where(hit_lo, 2 * lo - q, q)
        q = torch.where(hit_hi, 2 * hi - q, q)
        return {"p": p, "q": q, "v": v, "a_prev": a}

    # ---------- rendering ----------
    def _blobs(self, centres, r_pix):
        """centres (B, n, 2) in [0,1] -> (B, H, W) max-intensity image of discs."""
        c = self.c
        d = (self.grid[None, None] - centres[:, :, None, None, :]).norm(dim=-1)  # (B, n, H, W)
        r, s = r_pix / c.img, c.soft / c.img
        return torch.sigmoid((r - d) / s).amax(1)

    def render(self, st):
        c = self.c
        p = st["p"][:, None]
        agent_like = p
        if c.n_echo > 0:
            echoes = (p + self.echo_off[None, :c.n_echo]) % 1.0
            agent_like = torch.cat([p, echoes], 1)
        red = self._blobs(agent_like, c.agent_r)
        green = self._blobs(st["q"], c.dist_r) if c.n_dist > 0 else torch.zeros_like(red)
        blue = self.bg[None].expand_as(red) + 0.5
        return torch.stack([red, green, blue], 1)  # (B, 3, H, W) in about [0, 1]

    # ---------- windows ----------
    def rollout(self, B, T, burn_in=2, actions=None):
        """Simulate T steps after a short burn-in. Returns states and actions as tensors:
        p (B, T+1, 2), q (B, T+1, n_dist, 2), a (B, T, 2). Frames are rendered separately."""
        st = self.reset(B)
        for _ in range(burn_in):
            st = self.step(st, self.sample_action(st))
        ps, qs, vs, acts = [st["p"]], [st["q"]], [st["v"]], []
        for t in range(T):
            a = self.sample_action(st) if actions is None else actions[:, t]
            acts.append(a.clamp(-1, 1))
            st = self.step(st, a)
            ps.append(st["p"]); qs.append(st["q"]); vs.append(st["v"])
        return {"p": torch.stack(ps, 1), "q": torch.stack(qs, 1), "v": torch.stack(vs, 1),
                "a": torch.stack(acts, 1)}

    def render_seq(self, roll):
        """Render (B, T+1) frames from a rollout dict. Returns (B, T+1, 3, H, W)."""
        B, T1 = roll["p"].shape[:2]
        st = {"p": roll["p"].reshape(B * T1, 2), "q": roll["q"].reshape(B * T1, -1, 2)}
        return self.render(st).reshape(B, T1, 3, self.c.img, self.c.img)

    def response_fraction(self, n=256):
        """Measured fraction of pixels whose value changes by more than 0.1 under one random action.
        Logged so that 'response fraction' is a measured quantity, not a label."""
        st = self.reset(n)
        f0 = self.render(st)
        st1 = self.step(st, self.sample_action(st))
        st1["q"], st1["v"] = st["q"], st["v"]  # isolate the action's effect
        f1 = self.render(st1)
        return float(((f1 - f0).abs().amax(1) > 0.1).float().mean())
