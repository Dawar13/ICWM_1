"""Sweep registry. Each sweep yields (run_name, script, cfg_dict).

Seeds and levels here must match the pre-registrations in prereg/. If you change a sweep after
seeing results, write a new pre-registration and say so (docs/05_standards.md).
"""
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G1 = "src/g1_linear/sim.py"
G2 = "src/g2_pixel/train.py"


def _name(d):
    return "_".join(f"{k}={v}" for k, v in sorted(d.items()))


def grid(script, base, axes, seeds):
    keys = list(axes)
    for vals in itertools.product(*[axes[k] for k in keys]):
        for s in seeds:
            cfg = dict(base)
            cfg.update(dict(zip(keys, vals)))
            cfg["seed"] = s
            yield _name({**dict(zip(keys, vals)), "seed": s}), script, cfg


# ---------------- G1 (linear, CPU is fine) ----------------
G1_BASE = {}  # defaults in G1Config are the trap regime found in smoke tests (notes/smoke_tests.md)


def g1_calib():  # exploratory: map where plain training fails and where IDM-always fixes it
    yield from grid(G1, G1_BASE, {"rho_a": [0.0, 0.5, 0.9, 0.97], "k": [4, 5], "idm": ["none", "always"]}, [0])


def g1_trap():  # PRIMARY (P3 persistence): is the failure a trap that a one-off warm-up escapes?
    yield from grid(G1, G1_BASE, {"idm": ["none", "always", "warmup"]}, range(5))


def g1_dose():  # how short can the warm-up be?
    yield from grid(G1, {**G1_BASE, "idm": "warmup"},
                    {"warmup_frac": [0.005, 0.01, 0.02, 0.05, 0.1], "idm_weight": [3.0, 10.0]}, range(3))


def g1_rho():  # which agents are trapped: passive persistence boundary
    yield from grid(G1, G1_BASE, {"rho_a": [0.0, 0.25, 0.5, 0.75, 0.9, 0.97], "idm": ["none", "always"]}, range(3))


def g1_init():  # does action-pathway init scale or depth matter?
    yield from grid(G1, G1_BASE, {"act_init": [1e-4, 1e-3, 1e-2, 1e-1, 0.3], "act_depth": [1, 2]}, range(3))


def g1_hist():  # history x action autocorrelation: blind vs deaf
    yield from grid(G1, G1_BASE, {"phi": [0.0, 0.5, 0.8, 0.9, 0.95], "h": [1, 3], "idm": ["none", "always"]}, range(3))


def g1_objective():  # boundary: when the agent is NOT the better choice even with the action, warm-up must fail
    yield from grid(G1, G1_BASE, {"agent_noise": [0.0, 0.2, 0.4, 0.6], "idm": ["none", "always", "warmup"]}, range(3))


def g1_reg():  # generality across anti-collapse choices
    yield from grid(G1, G1_BASE, {"reg": ["cov", "sigreg", "ema"], "idm": ["none", "warmup", "always"]}, range(3))


def g1_unstable():  # Toso et al. regime: unstable agent, closed-loop data
    yield from grid(G1, {**G1_BASE, "agent_mode": "unstable_cl"}, {"idm": ["none", "warmup", "always"]}, range(3))


# Added 8 Nov 2026 after the derivation in notes/theory.md and BEFORE any G1 sweep was run (prereg/G1.md, part B).
def g1_phase():  # restated P3: warm-up persists iff the agent-kept solution is a strict local minimum
    yield from grid(G1, G1_BASE, {"rho_a": [0.0, 0.5, 0.8], "agent_noise": [0.3, 0.75, 2.0],
                                  "idm": ["none", "warmup", "always"]}, range(3))


def g1_depth():  # slow vs fast action pathway: depth-1 runs, paired with the depth-2 idm=none runs of g1_rho
    yield from grid(G1, {**G1_BASE, "act_depth": 1, "idm": "none"}, {"rho_a": [0.25, 0.5, 0.75, 0.9]}, range(3))


# ---------------- G2 (pixels, shared GPU) ----------------
def _chosen():
    p = ROOT / "runs" / "g2" / "CHOSEN_CONFIG.json"
    if not p.exists():
        raise SystemExit("Write runs/g2/CHOSEN_CONFIG.json first (rule in prereg/G2.md).")
    return json.loads(p.read_text())


def g2_calib():  # pick the SIGReg weight by whitening quality only, before looking at agent metrics
    yield from grid(G2, {"steps": 5000, "idm": "always"}, {"sigreg_w": [0.03, 0.1, 0.3, 1.0]}, [0])


def g2_reproduce():  # ladder: find a config where plain training loses the agent and IDM-always keeps it
    for mode in ["set", "integrator"]:
        for n_dist in [2, 4]:
            for d in [16, 32]:
                yield from grid(G2, {"agent_mode": mode, "n_dist": n_dist, "d": d}, {"idm": ["none", "always"]}, [0])


def g2_trap():  # PRIMARY (decisive): warm-up only vs always vs none on the chosen config
    yield from grid(G2, _chosen(), {"idm": ["none", "always", "warmup"]}, range(3))


def g2_response():  # response fraction (echoes) vs agent size
    yield from grid(G2, {**_chosen(), "idm": "none"}, {"n_echo": [0, 1, 3, 7], "agent_r": [1.5, 2.5, 4.0]}, range(2))


def g2_hist():  # history x action autocorrelation x history dropout
    yield from grid(G2, {**_chosen(), "idm": "none"}, {"phi": [0.0, 0.9], "h": [1, 3], "hist_dropout": [0.0, 0.5]}, range(3))


def g2_ema():
    yield from grid(G2, {**_chosen(), "reg": "ema"}, {"idm": ["none", "warmup", "always"]}, range(3))


SWEEPS = {k: v for k, v in globals().items() if k.startswith(("g1_", "g2_")) and callable(v)}
