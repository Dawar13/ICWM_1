"""Reduced theory of the linear model (notes/theory.md). Pure maths plus Monte Carlo checks; no training.

Plain language
--------------
If the predictor is always at its best for the current encoder and the latent is exactly whitened,
the loss depends only on WHICH k-dimensional subspace of the (normalised) factors the encoder reads.
Keeping a set K of whole factors costs sum_{i in K} nerr_i, where nerr_i is factor i's one-step error
once the action is known. Whether such a set is a trap (a local minimum that is not the best) depends on
a simple pairwise number, c(X kept, Y dropped) = (d_X - d_Y)^2 + nerr_Y - nerr_X, where d is a factor's
own autoregressive coefficient. c > 0 for every pair means the set is a strict local minimum.

Assumptions (A1-A5; see notes/theory.md): integrator agent, phi = 0 (i.i.d. exogenous actions), h = 1,
hard whitening, predictor (A, V) at its optimum. Under a slow action pathway (V ~ 0, as with the default
factorised V2 V1 at init 1e-3) the agent's action term is absent and nerr_agent is replaced by its passive
error 1 - d_a^2 ("V0" quantities below).

  python src/analysis/theory_linear.py check        # numeric Hessian and Monte Carlo checks of the formulas
  python src/analysis/theory_linear.py predict      # predicted outcome for every registered G1 sweep cell
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------------------------------
# factor parameters in normalised coordinates
# ----------------------------------------------------------------------------------------------
def factors(cfg: dict):
    """Return dict with arrays over factors [agent, env_1..env_de]:
    d     : autoregressive coefficient of the normalised factor
    beta2 : squared coefficient of the (unit-variance) action in the normalised factor
    nerr  : action-conditioned normalised one-step error (= 1 - d^2 - beta2)
    npas  : passive normalised one-step error (= 1 - d^2 when phi = 0)
    gain  : effective observation gain (observation gain x factor std)
    Only valid for agent_mode 'integrator' with phi = 0 and d_a = 1."""
    from src.g1_linear.sim import G1Config
    c = G1Config(**cfg)
    if c.agent_mode != "integrator" or c.phi != 0.0 or c.d_a != 1:
        raise ValueError("reduced theory needs agent_mode integrator, phi 0, d_a 1")
    var_a = (c.b ** 2 + c.agent_noise ** 2) / (1 - c.rho_a ** 2)
    d = [c.rho_a] + list(c.rho_e[:c.d_e])
    beta2 = [c.b ** 2 / var_a] + [0.0] * c.d_e
    gain = [c.g_a * math.sqrt(var_a)] + [c.g_e] * c.d_e
    d, beta2 = np.array(d), np.array(beta2)
    nerr = 1 - d ** 2 - beta2
    npas = 1 - d ** 2
    return {"d": d, "beta2": beta2, "nerr": nerr, "npas": npas, "gain": np.array(gain), "k": c.k, "lam": c.lam}


def pair_c(f, X, Y, slow_v=False):
    """Curvature coefficient for rotating kept factor X towards dropped factor Y:
    L(s) - L(0) = c * s - (d_X - d_Y)^2 * s^2 exactly along the rotation (s = sin^2 theta)."""
    ne = f["npas"] if slow_v else f["nerr"]
    return (f["d"][X] - f["d"][Y]) ** 2 + ne[Y] - ne[X]


def subset_report(f, K, slow_v=False):
    n = len(f["d"])
    ne = f["npas"] if slow_v else f["nerr"]
    Kc = [i for i in range(n) if i not in K]
    cs = {(X, Y): pair_c(f, X, Y, slow_v) for X in K for Y in Kc}
    loss = float(sum(ne[i] for i in K))
    loss_soft = float(sum(ne[i] - ne[i] ** 2 / (2 * f["lam"]) for i in K))  # logged pred_loss under the soft penalty
    return {"K": list(K), "loss": loss, "pred_loss_soft": loss_soft, "c": cs,
            "min_c": float(min(cs.values())) if cs else float("inf"),
            "argmin": min(cs, key=cs.get) if cs else None}


def classify(cfg: dict):
    """Theory predictions for one config (keys used in notes/theory.md and prereg/G1.md)."""
    f = factors(cfg)
    k, n = f["k"], len(f["d"])
    env = list(range(1, n))
    env_sorted = sorted(env, key=lambda i: f["nerr"][i])
    K_ev = sorted(env_sorted[:k])                      # agent evicted: best k environment factors
    K_kept = sorted([0] + env_sorted[:k - 1])          # agent kept: agent + best k-1 environment factors
    best = sorted(np.argsort(f["nerr"], kind="stable")[:k].tolist())
    ev, kept = subset_report(f, K_ev), subset_report(f, K_kept)
    ev0 = subset_report(f, K_ev, slow_v=True)          # with a dead action pathway (V = 0)
    inside = 0 in best                                 # agent in the action-conditioned top k (original P3)
    passive_inside = bool(0 in np.argsort(f["npas"], kind="stable")[:k])
    ev_c = min(ev["c"][(X, 0)] for X in K_ev)          # evicted: stability against bringing the agent in
    ev0_c = min(ev0["c"][(X, 0)] for X in K_ev)
    kept_c = min(kept["c"][(0, Y)] for Y in range(n) if Y not in K_kept)
    g2 = f["gain"][0] ** 2
    return {
        "nerr_agent_action": float(f["nerr"][0]), "nerr_agent_passive": float(f["npas"][0]),
        "env_nerr": [float(x) for x in f["nerr"][1:]],
        "loss_evicted": ev["loss"], "loss_kept": kept["loss"],
        "pred_loss_soft_evicted": ev["pred_loss_soft"], "pred_loss_soft_kept": kept["pred_loss_soft"],
        "inside_action_topk": bool(inside), "inside_passive_topk": passive_inside,
        "c_evicted_fastV": float(ev_c), "c_evicted_slowV": float(ev0_c), "c_kept": float(kept_c),
        "agent_gain2": float(g2),
        # predictions
        "plain_fastV": "evicted" if ev_c > 0 else "kept (saddle escape)",
        "plain_slowV": "evicted" if ev0_c > 0 else ("kept" if ev0_c < 0 else "marginal"),
        "warmup": "persists" if kept_c > 0 else "decays",
        "region": ("A trap" if inside and ev_c > 0 else
                   "B no trap" if inside else
                   "C multistable" if kept_c > 0 else "D objective"),
        "barrier_share": float(ev_c / (2 * (f["d"][ev["argmin"][0]] - f["d"][0]) ** 2))
        if ev_c > 0 and abs(f["d"][ev["argmin"][0]] - f["d"][0]) > 1e-9 else None,
    }


# ----------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------
def reduced_loss(P, f):
    """L(P) = k - ||P D P^T||_F^2 - ||P b||^2 for P with orthonormal rows (k x n)."""
    D = np.diag(f["d"])
    b = np.sqrt(f["beta2"])
    M = P @ D @ P.T
    return P.shape[0] - (M ** 2).sum() - ((P @ b) ** 2).sum()


def check_hessian(cfg, K, eps=1e-4):
    """Finite-difference Hessian of the reduced loss on the Grassmannian at the coordinate subspace K,
    in the coordinates P = [I_K, X] orthonormalised; compare with 2 * c(X, Y) on the diagonal."""
    f = factors(cfg)
    n, k = len(f["d"]), f["k"]
    Kc = [i for i in range(n) if i not in K]
    E = np.eye(n)

    def P_of(Xm):
        # rows: e_j + sum_i X_ji e_i for j in K, then orthonormalise (QR) -> a point on the Grassmannian
        R = E[K] + Xm @ E[Kc]
        q, _ = np.linalg.qr(R.T)
        return q.T

    m = k * (n - k)
    H = np.zeros((m, m))
    base = reduced_loss(P_of(np.zeros((k, n - k))), f)
    idx = list(itertools.product(range(k), range(n - k)))
    for a, (j1, i1) in enumerate(idx):
        for b_, (j2, i2) in enumerate(idx):
            def L(s1, s2):
                Xm = np.zeros((k, n - k))
                Xm[j1, i1] += s1
                Xm[j2, i2] += s2
                return reduced_loss(P_of(Xm), f)
            H[a, b_] = (L(eps, eps) - L(eps, -eps) - L(-eps, eps) + L(-eps, -eps)) / (4 * eps ** 2)
    pred = np.diag([2 * pair_c(f, K[j], Kc[i]) for j, i in idx])
    return float(np.abs(H - pred).max()), base, sum(f["nerr"][i] for i in K)


def check_monte_carlo(cfg, K, n_steps=400, B=4096, seed=0):
    """Simulate the real LinearWorld, read the factors in K exactly (normalised), fit the best linear
    predictor from (z_t, a_t) by least squares, and compare its residual with sum_{i in K} nerr_i."""
    import torch
    from src.g1_linear.sim import G1Config, LinearWorld
    c = G1Config(**cfg)
    g = torch.Generator().manual_seed(seed)
    w = LinearWorld(c, B, g)
    for _ in range(c.burn_in):
        w.step()
    S, A, S1 = [], [], []
    for _ in range(n_steps):
        s_a, s_e, a = w.step()
        S.append(torch.cat([s_a, s_e], 1)); A.append(a); S1.append(torch.cat([w.s_a, w.s_e], 1))
    S, A, S1 = (torch.cat(t).double().numpy() for t in (S, A, S1))
    sd = S.std(0)
    Z, Z1 = S[:, K] / sd[K], S1[:, K] / sd[K]
    X = np.concatenate([Z, A, np.ones((len(Z), 1))], 1)
    coef, *_ = np.linalg.lstsq(X, Z1, rcond=None)
    resid = ((Z1 - X @ coef) ** 2).sum(1).mean()
    f = factors(cfg)
    return float(resid), float(sum(f["nerr"][i] for i in K))


# ----------------------------------------------------------------------------------------------
# predictions for the registered sweeps
# ----------------------------------------------------------------------------------------------
def sweep_predictions():
    from src.sweeps import SWEEPS
    out = {}
    for name in ["g1_trap", "g1_rho", "g1_objective", "g1_phase", "g1_depth", "g1_dose", "g1_init", "g1_reg"]:
        if name not in SWEEPS:
            continue
        cells = {}
        for run, _, cfg in SWEEPS[name]():
            key = run.rsplit("_seed=", 1)[0]
            if key in cells:
                continue
            try:
                cells[key] = classify(cfg)
            except ValueError as e:
                cells[key] = {"outside_theory": str(e)}
        out[name] = cells
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if mode == "check":
        default = {}
        for cfg, K, label in [(default, [1, 2, 3, 4], "default, agent evicted"),
                              (default, [0, 1, 2, 3], "default, agent kept"),
                              ({"agent_noise": 0.6}, [0, 1, 2, 3], "noise 0.6, kept"),
                              ({"rho_a": 0.75}, [1, 2, 3, 4], "rho_a 0.75, evicted"),
                              ({"rho_a": 0.5, "agent_noise": 2.0}, [0, 1, 2, 3], "rho_a 0.5 noise 2, kept")]:
            err, Lnum, Lthe = check_hessian(cfg, K)
            mc, the = check_monte_carlo(cfg, K)
            print(f"{label:28s} Hessian max abs error {err:.2e}; reduced loss {Lnum:.4f} vs formula {Lthe:.4f}; "
                  f"Monte Carlo residual {mc:.4f} vs formula {the:.4f}")
        # exact 1-D profile along a rotation, checked against c*s - (dX-dY)^2 s^2
        f = factors(default)
        X, Y = 4, 0
        worst = 0.0
        for s in np.linspace(0, 1, 11):
            th = math.asin(math.sqrt(s))
            P = np.eye(5)[[1, 2, 3, 4]]
            P[3] = math.cos(th) * np.eye(5)[X] + math.sin(th) * np.eye(5)[Y]
            num = reduced_loss(P, f) - reduced_loss(np.eye(5)[[1, 2, 3, 4]], f)
            the = pair_c(f, X, Y) * s - (f["d"][X] - f["d"][Y]) ** 2 * s ** 2
            worst = max(worst, abs(num - the))
        print(f"1-D rotation profile (env 0.9 -> agent): max abs error {worst:.2e}")
    else:
        print(json.dumps(sweep_predictions(), indent=1, default=str))


if __name__ == "__main__":
    main()
