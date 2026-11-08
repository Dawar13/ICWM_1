"""G4b. Align two independently trained tiny pixel world models (from G2) using actions only.

Each model sees only its OWN states (unpaired). Paired states are used ONLY for evaluation.
Steps:
  1. For each model, encode its own sampled states; compute mean multi-step responses to unit
     actions: R_k[:, j] = mean_z [ F^k(z, a=delta*e_j) - F^k(z, a=0) ]  (k = 1..K)
  2. Orthogonal Procrustes between the stacked responses -> map T (A coords -> B coords).
     Also fit a general linear map by least squares (the models need not be exact rotations).
  3. Evaluate on paired states: a probe trained on model B (agent position, distractor positions)
     applied to T z_A, compared with B's own within-model R^2.
Prediction: agent (controllable) transfers; distractors (uncontrollable) do not, unless the
geometry happens to pin them down.
Baseline: Gromov-Wasserstein on unpaired latents.
Requires models with h = 1 (history length one). Usage:
  python src/g4_align/align_tiny_models.py --a runs/g2/.../modelA --b runs/g2/.../modelB
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.gpu import pick_device  # noqa: E402
from src.common.metrics import r2_score, ridge_fit  # noqa: E402
from src.g2_pixel.env import SpriteWorld  # noqa: E402
from src.g2_pixel.plan_eval import load  # noqa: E402


@torch.no_grad()
def responses(model, world, n=1024, K=4, delta=0.5, dev="cpu"):
    roll = world.rollout(n, 1)
    z0 = model.enc(world.render({"p": roll["p"][:, 0], "q": roll["q"][:, 0]}))
    cols = []
    for j in range(2):
        a = torch.zeros(n, 2, device=dev); a[:, j] = delta
        za, zb = z0[:, None], z0[:, None]
        for _ in range(K):
            na = model.pred(za, a); nb = model.pred(zb, torch.zeros_like(a))
            cols.append((na - nb).mean(0))
            za, zb = na[:, None], nb[:, None]
    return torch.stack(cols, 1).cpu().double().numpy(), z0.cpu().double().numpy()


def procrustes(X, Y):
    u, _, vt = np.linalg.svd(Y @ X.T)
    return u @ vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = pick_device(max_mem_gb=4.0)
    mA, env_c, tcA = load(args.a, dev); mB, env_cB, tcB = load(args.b, dev)
    assert tcA.h == 1 and tcB.h == 1, "use h=1 models"
    KA, ZA = responses(mA, SpriteWorld(env_c, dev, seed=111), dev=dev)   # unpaired: different seeds
    KB, ZB = responses(mB, SpriteWorld(env_cB, dev, seed=222), dev=dev)
    T_orth = procrustes(KA, KB)
    T_lin = KB @ np.linalg.pinv(KA)  # minimum-norm linear map consistent with the responses
    # paired evaluation states
    world = SpriteWorld(env_c, dev, seed=333)
    roll = world.rollout(2048, 1)
    frames = world.render({"p": roll["p"][:, 0], "q": roll["q"][:, 0]})
    with torch.no_grad():
        zA = mA.enc(frames).cpu().double(); zB = mB.enc(frames).cpu().double()
    p = roll["p"][:, 0].cpu().double(); q = roll["q"][:, 0].flatten(1).cpu().double()
    half = 1024
    out = {}
    for name, y in [("agent", p), ("dist", q)]:
        wB, bB = ridge_fit(zB[:half], y[:half])
        out[f"{name}_within_B"] = r2_score(y[half:], zB[half:] @ wB + bB)
        for tn, T in [("orth", T_orth), ("lin", T_lin)]:
            zA_mapped = zA[half:] @ torch.from_numpy(T).T
            out[f"{name}_transfer_{tn}"] = r2_score(y[half:], zA_mapped @ wB + bB)
        out[f"{name}_unaligned"] = r2_score(y[half:], zA[half:] @ wB + bB)
    try:
        import ot
        n = 400
        CA, CB = ot.dist(ZA[:n], ZA[:n]), ot.dist(ZB[:n], ZB[:n])
        G = ot.gromov.gromov_wasserstein(CA / CA.max(), CB / CB.max(), ot.unif(n), ot.unif(n), "square_loss")
        T_gw = procrustes(ZA[:n].T, ((G / G.sum(1, keepdims=True)) @ ZB[:n]).T)
        for name, y in [("agent", p), ("dist", q)]:
            wB, bB = ridge_fit(zB[:half], y[:half])
            out[f"{name}_transfer_gw"] = r2_score(y[half:], zA[half:] @ torch.from_numpy(T_gw).T @ wB + bB)
    except ImportError:
        out["gw"] = "POT not installed"
    out["response_rank_A"] = int(np.linalg.matrix_rank(KA, tol=1e-3 * np.linalg.norm(KA, 2)))
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
