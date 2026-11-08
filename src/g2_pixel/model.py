"""G2 model: a tiny LeWM-like JEPA world model (pixels -> latent, latent history + action -> next latent).

Deliberately small (well under 1M parameters by default) so that many seeds fit on a shared GPU.
Before trusting any pixel conclusion, Claude Code must compare the design choices below with
the real LeWM code (predictor conditioning, init of the action pathway, SIGReg form and weight,
target gradient) and record the differences in notes/lewm_diff.md.
"""
import math

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d=32, width=32, img=64, out_bn=True):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(w, 2 * w, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(2 * w, 4 * w, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(4 * w, 4 * w, 4, 2, 1), nn.SiLU(),
        )
        self.head = nn.Linear(4 * w * (img // 16) ** 2, d)
        # LeWM uses a projector with batch normalisation (verify in its code). Without it this small CNN
        # starts with near-constant latents and the characteristic-function regulariser has almost no gradient.
        self.bn = nn.BatchNorm1d(d) if out_bn else nn.Identity()

    def forward(self, x):
        return self.bn(self.head(self.net(x).flatten(1)))


class ActionEmbed(nn.Module):
    """Two-layer action embedding. act_init_scale multiplies the default init of the last layer
    (0 gives a zero-initialised action pathway, as in AdaLN-Zero style designs)."""

    def __init__(self, a_dim=2, m=64, act_init_scale=1.0):
        super().__init__()
        self.l1, self.l2 = nn.Linear(a_dim, m), nn.Linear(m, m)
        with torch.no_grad():
            self.l2.weight.mul_(act_init_scale)
            self.l2.bias.mul_(act_init_scale)

    def forward(self, a):
        return self.l2(torch.nn.functional.silu(self.l1(a)))


class FiLMBlock(nn.Module):
    def __init__(self, hdim, m, act_init_scale):
        super().__init__()
        self.norm = nn.LayerNorm(hdim, elementwise_affine=False)
        self.mod = nn.Linear(m, 2 * hdim)
        nn.init.normal_(self.mod.weight, std=act_init_scale / math.sqrt(m))
        nn.init.zeros_(self.mod.bias)
        self.mlp = nn.Sequential(nn.Linear(hdim, 2 * hdim), nn.SiLU(), nn.Linear(2 * hdim, hdim))

    def forward(self, x, e):
        g, b = self.mod(e).chunk(2, -1)
        return x + self.mlp(self.norm(x) * (1 + g) + b)


class Predictor(nn.Module):
    def __init__(self, d=32, h=1, hdim=256, m=64, cond="film", act_init_scale=1.0, n_blocks=2, residual=True):
        super().__init__()
        self.h, self.cond, self.residual = h, cond, residual
        self.aemb = ActionEmbed(2, m, act_init_scale)
        in_dim = h * d + (m if cond == "concat" else 0)
        self.inp = nn.Linear(in_dim, hdim)
        if cond == "film":
            self.blocks = nn.ModuleList([FiLMBlock(hdim, m, act_init_scale) for _ in range(n_blocks)])
        else:
            self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, 2 * hdim), nn.SiLU(),
                                                       nn.Linear(2 * hdim, hdim)) for _ in range(n_blocks)])
        self.out = nn.Linear(hdim, d)

    def forward(self, z_hist, a):
        """z_hist (B, h, d), oldest first; a (B, 2). Returns predicted z_{t+1} (B, d)."""
        e = self.aemb(a)
        x = z_hist.flatten(1)
        if self.cond == "concat":
            x = torch.cat([x, e], -1)
        x = self.inp(x)
        for blk in self.blocks:
            x = blk(x, e) if self.cond == "film" else x + blk(x)
        out = self.out(x)
        return z_hist[:, -1] + out if self.residual else out


class IDMHead(nn.Module):
    def __init__(self, d=32, hdim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hdim), nn.SiLU(), nn.Linear(hdim, 2))

    def forward(self, z_t, z_next):
        return self.net(torch.cat([z_t, z_next], -1))


def sigreg_epps_pulley(z, n_proj=256, t_max=5.0, n_t=17):
    """SIGReg (Epps-Pulley form): compare the empirical characteristic function of random 1-D
    projections of z with that of N(0, 1), integrate over t, scale by batch size, average over slices.
    Reconstructed from the LeJEPA paper's minimal code as we remember it. Before any pixel conclusion,
    diff this against LeWM's own implementation and weight (record in notes/lewm_diff.md)."""
    n, d = z.shape
    u = torch.randn(d, n_proj, device=z.device, dtype=torch.float32)
    u = u / u.norm(dim=0, keepdim=True)
    x = z.float() @ u
    t = torch.linspace(-t_max, t_max, n_t, device=z.device)
    xt = x.unsqueeze(-1) * t
    re, im = torch.cos(xt).mean(0), torch.sin(xt).mean(0)
    w = torch.exp(-0.5 * t ** 2)
    err = ((re - w) ** 2 + im ** 2) * w
    return torch.trapz(err, t, dim=-1).mean() * n


class TinyJEPA(nn.Module):
    def __init__(self, d=32, h=1, enc_width=32, hdim=256, cond="film", act_init_scale=1.0, img=64, enc_bn=True):
        super().__init__()
        self.enc = Encoder(d, enc_width, img, enc_bn)
        self.pred = Predictor(d, h, hdim, cond=cond, act_init_scale=act_init_scale)
        self.idm = IDMHead(d)
        self.h = h

    def encode_seq(self, frames):
        """frames (B, T, 3, H, W) -> (B, T, d)"""
        B, T = frames.shape[:2]
        return self.enc(frames.flatten(0, 1)).view(B, T, -1)
