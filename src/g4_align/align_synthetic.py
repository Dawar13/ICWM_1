"""G4a. Actions identify the map between two world models, on the controllable subspace only.

Plain-language summary
----------------------
Two models see the same world but each stores it in its own rotated coordinates (SIGReg makes
models identifiable only up to a rotation). If the stored data look the same from every angle
(isotropic Gaussian), no method that only looks at the shape of the data can find the rotation.
But both models must agree on what each action does. Matching the models' predicted responses to
the same actions recovers the rotation on everything the actions can reach, and nothing else.

Baselines: random rotation; Gromov-Wasserstein (shape only); plain OT (assumes no rotation);
paired Procrustes with k pairs (oracle, for a "pairs-equivalence" curve).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def rand_orth(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def procrustes(X, Y):
    """Orthogonal R minimising ||R X - Y||_F (columns are corresponding vectors)."""
    u, _, vt = np.linalg.svd(Y @ X.T)
    return u @ vt


def make_world(n=8, c=4, m=2, aniso=0.0, rng=None):
    """Latent s in R^n: first c dims controllable by m-dim actions, rest autonomous.
    Returns M, B, a mixing basis P, and per-dimension scales (aniso=0 gives an isotropic latent)."""
    Mc = rand_orth(c, rng) * 0.9
    Mu = rand_orth(n - c, rng) * 0.9
    M = np.zeros((n, n)); M[:c, :c] = Mc; M[c:, c:] = Mu
    B = np.zeros((n, m)); B[:c] = rng.standard_normal((c, m))
    P = rand_orth(n, rng)
    scales = np.exp(aniso * np.linspace(-1, 1, n))
    return {"M": P @ M @ P.T, "B": P @ B, "scales": scales, "P": P, "c": c}


def krylov(Q, w, k):
    """Controllability-style response matrix of a model with coordinates z = Q s:
    [Q B, Q M B, ..., Q M^{k-1} B]."""
    blocks, MB = [], w["B"]
    for _ in range(k):
        blocks.append(Q @ MB)
        MB = w["M"] @ MB
    return np.concatenate(blocks, 1)


def subspace_cosines(R, T, U):
    """Mean cosine between R u and T u over orthonormal columns u of U."""
    a, b = R @ U, T @ U
    return float(np.mean(np.sum(a * b, 0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))))


def run(n=8, c=4, m=2, k=4, aniso=0.0, pred_noise=0.0, n_samples=400, seed=0, gw=True):
    rng = np.random.default_rng(seed)
    w = make_world(n, c, m, aniso, rng)
    QA, QB = rand_orth(n, rng), rand_orth(n, rng)
    T = QB @ QA.T  # true map from A's coordinates to B's
    # each model's estimated responses (imperfect predictors: relative noise)
    KA = krylov(QA, w, k); KB = krylov(QB, w, k)
    KA = KA + pred_noise * np.linalg.norm(KA) / np.sqrt(KA.size) * rng.standard_normal(KA.shape)
    KB = KB + pred_noise * np.linalg.norm(KB) / np.sqrt(KB.size) * rng.standard_normal(KB.shape)
    R = procrustes(KA, KB)
    # controllable subspace (in A's coordinates) and its complement
    Uc_full, sv, _ = np.linalg.svd(QA @ krylov(np.eye(n), w, n))
    rank = int((sv > 1e-8 * sv[0]).sum())
    Uc, Uu = Uc_full[:, :rank], Uc_full[:, rank:]
    out = {"rank_true": rank, "sigma_min_K": float(np.linalg.svd(KA, compute_uv=False)[rank - 1]),
           "ours_ctrl": subspace_cosines(R, T, Uc), "ours_unctrl": subspace_cosines(R, T, Uu) if Uu.size else None}
    Rr = rand_orth(n, rng)
    out["random_ctrl"] = subspace_cosines(Rr, T, Uc)
    # unpaired samples from each model (independent draws), isotropic or anisotropic
    S_A = rng.standard_normal((n_samples, n)) * w["scales"]
    S_B = rng.standard_normal((n_samples, n)) * w["scales"]
    ZA, ZB = (QA @ w["P"] @ S_A.T).T, (QB @ w["P"] @ S_B.T).T
    if gw:
        import ot
        CA, CB = ot.dist(ZA, ZA), ot.dist(ZB, ZB)
        CA /= CA.max(); CB /= CB.max()
        p = q = np.full(n_samples, 1 / n_samples)
        G = ot.gromov.gromov_wasserstein(CA, CB, p, q, "square_loss")
        ZB_match = (G / G.sum(1, keepdims=True)) @ ZB
        R_gw = procrustes(ZA.T, ZB_match.T)
        out["gw_ctrl"] = subspace_cosines(R_gw, T, Uc)
        G2 = ot.emd(p, q, ot.dist(ZA, ZB))
        R_ot = procrustes(ZA.T, ((G2 / G2.sum(1, keepdims=True)) @ ZB).T)
        out["ot_ctrl"] = subspace_cosines(R_ot, T, Uc)
    # paired oracle with k pairs (pairs-equivalence curve)
    for kp in [2, 4, 8, 16, 64]:
        S = rng.standard_normal((kp, n)) * w["scales"]
        ZpA = QA @ w["P"] @ S.T
        ZpB = QB @ w["P"] @ S.T + pred_noise * rng.standard_normal((n, kp))
        out[f"paired{kp}_ctrl"] = subspace_cosines(procrustes(ZpA, ZpB), T, Uc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/g4/synthetic.json")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    rows = []
    for aniso in [0.0, 0.5, 1.0]:
        for pred_noise in [0.0, 0.05, 0.2]:
            for s in range(a.seeds):
                r = run(aniso=aniso, pred_noise=pred_noise, seed=s)
                rows.append({"aniso": aniso, "pred_noise": pred_noise, "seed": s, **r})
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
