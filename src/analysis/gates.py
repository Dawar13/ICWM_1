"""Turn sweep outputs into pre-registered PASS / FAIL verdicts.

  python src/analysis/gates.py g1_trap
  python src/analysis/gates.py g2_reproduce     # also writes runs/g2/CHOSEN_CONFIG.json by the pre-registered rule
  python src/analysis/gates.py all

Thresholds live in THRESHOLDS below and must match prereg/*.md. Do not edit them after seeing
results; if you must, write a new pre-registration and say so.
Each verdict is also appended to runs/gate_reports.md with the code hash.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.runlog import code_hash  # noqa: E402

THRESHOLDS = {
    # G1 primary (linear trap and warm-up persistence), 5 seeds
    "g1_vanilla_fail_r2": 0.2, "g1_idm_fix_r2": 0.8, "g1_persist_ratio": 0.8, "g1_min_seeds": 4,
    "g1_kill_margin": 0.1,
    # G1 rho (learning-order prediction): fraction of rho levels where the passive ranking predicts the outcome
    "g1_rho_accuracy": 5 / 6,
    # G1 hist: Spearman of action-use escape step with phi (set-mode agent)
    "g1_hist_spearman": 0.8,
    # G2 calibration: whitening rule
    "g2_cov_err_max": 0.15, "g2_zvar_lo": 0.5, "g2_zvar_hi": 2.0,
    # G2 reproduce rule
    "g2_vanilla_fail_r2": 0.5, "g2_idm_fix_r2": 0.8,
    # G2 decisive
    "g2_persist_ratio_mean": 0.8, "g2_persist_ratio_seed": 0.7, "g2_kill_margin": 0.1, "g2_plan_ratio": 0.8,
    # G2 response
    "g2_response_spearman": 0.8,
    # G4
    "g4_ours_ctrl": 0.9, "g4_gw_ctrl_max": 0.3, "g4_unctrl_max": 0.3, "g4_transfer_ratio": 0.7, "g4_kill_ratio": 0.5,
    # Added 8 Nov 2026 with prereg/G1.md part B (after the derivation, before any G1 sweep was run)
    "g1_phase_accuracy": 0.8, "g1_phase_min_valid": 7,   # restated P3 (stability boundary)
    "g1_c_margin": 0.05,       # |c| below this: the theory makes no prediction for that cell (slow or marginal)
    "g1_dose_accuracy": 0.8, "g1_persist_abs": 0.8,       # P4 barrier classifier
}
T = THRESHOLDS


def load_sweep(group, sweep):
    runs = []
    for d in sorted((ROOT / "runs" / group / sweep).glob("*")):
        if not (d / "summary.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())["config"]
        cfg = meta if "train" not in meta else {**meta["env"], **meta["train"]}
        summ = json.loads((d / "summary.json").read_text())
        log = [json.loads(l) for l in open(d / "log.jsonl")]
        runs.append({"dir": d, "cfg": cfg, "final": summ, "log": log})
    if not runs:
        raise SystemExit(f"no finished runs in runs/{group}/{sweep}")
    return runs


def by(runs, *keys):
    out = defaultdict(list)
    for r in runs:
        out[tuple(r["cfg"][k] for k in keys)].append(r)
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def escape_step(log, metric="agent_r2", thr=0.5):
    for e in log:
        if e.get("kind") == "eval" and e.get(metric) is not None and e[metric] >= thr:
            return e["step"]
    return None


def report(name, lines, verdict):
    txt = f"\n## {name}  (code {code_hash()})\n" + "\n".join(lines) + f"\n\n**VERDICT: {verdict}**\n"
    print(txt)
    with open(ROOT / "runs" / "gate_reports.md", "a") as f:
        f.write(txt)


# ------------------------------- G1 -------------------------------
def g1_trap():
    g = by(load_sweep("g1", "g1_trap"), "idm")
    fin = {k[0]: {r["cfg"]["seed"]: r["final"] for r in v} for k, v in g.items()}
    seeds = sorted(fin["none"])
    van = [fin["none"][s]["agent_r2"] for s in seeds]
    alw = [fin["always"][s]["agent_r2"] for s in seeds]
    wu = [fin["warmup"][s]["agent_r2"] for s in seeds]
    pl_v = [fin["none"][s]["pred_loss"] for s in seeds]
    pl_a = [fin["always"][s]["pred_loss"] for s in seeds]
    n_fail = sum(v < T["g1_vanilla_fail_r2"] for v in van)
    n_fix = sum(a >= T["g1_idm_fix_r2"] for a in alw)
    n_persist = sum(w >= T["g1_persist_ratio"] * a for w, a in zip(wu, alw))
    n_kill = sum(w <= v + T["g1_kill_margin"] for w, v in zip(wu, van))
    obj_better = float(np.mean(np.array(pl_a) < np.array(pl_v)))
    lines = [f"seeds {seeds}", f"vanilla agent R2 {np.round(van, 3)}", f"IDM-always   {np.round(alw, 3)}",
             f"warm-up only {np.round(wu, 3)}", f"pred loss vanilla {np.round(pl_v, 3)} vs IDM-always {np.round(pl_a, 3)}",
             f"vanilla fails in {n_fail}, IDM fixes in {n_fix}, warm-up persists in {n_persist}, kill-condition in {n_kill} seeds",
             f"IDM-always reaches a LOWER prediction loss in {obj_better:.0%} of seeds (trap, not objective, if high)"]
    m = T["g1_min_seeds"]
    if n_fail < m or n_fix < m:
        v = "INCONCLUSIVE: regime does not show fail-and-fix; recalibrate with g1_calib (exploratory) and re-register"
    elif n_kill >= 3:
        v = "KILL: warm-up does not persist; the failure is not a trap here (objective territory, Toso et al.)"
    elif n_persist >= m and obj_better >= 0.8:
        v = "PASS: trap that a one-off warm-up escapes, and the escaped solution is better under the JEPA loss"
    else:
        v = "PARTIAL: see numbers"
    report("G1 trap / warm-up persistence (primary)", lines, v)


def _passive_rank_prediction(rk, k):
    """Predict whether plain training keeps the agent: is its passive normalised error among the k best?"""
    errs = [rk["agent_nerr_passive"]] + list(rk["env_nerr"])
    order = np.argsort(errs)
    return bool(0 in order[:k])


def g1_rho():
    runs = [r for r in load_sweep("g1", "g1_rho") if r["cfg"]["idm"] == "none"]
    rows, correct = [], []
    for (rho,), rs in sorted(by(runs, "rho_a").items()):
        rk = rs[0]["log"][0]
        pred = _passive_rank_prediction(rk, rs[0]["cfg"]["k"])
        kept = np.mean([r["final"]["agent_r2"] for r in rs]) >= 0.5
        correct.append(pred == kept)
        rows.append(f"rho_a {rho}: passive nerr {rk['agent_nerr_passive']:.3f}, predicted kept={pred}, "
                    f"observed mean R2={np.mean([r['final']['agent_r2'] for r in rs]):.3f}")
    acc = float(np.mean(correct))
    v = "PASS" if acc >= T["g1_rho_accuracy"] else "FAIL"
    report("G1 learning order: passive ranking predicts who is kept", rows + [f"accuracy {acc:.2f}"], v)


def g1_hist():
    runs = [r for r in load_sweep("g1", "g1_hist") if r["cfg"]["idm"] == "none"]
    lines, verdicts = [], []
    for (h,), rs in sorted(by(runs, "h").items()):
        phis, steps, enc = [], [], []
        for (phi,), rr in sorted(by(rs, "phi").items()):
            esc = []
            for r in rr:
                s = next((e["step"] for e in r["log"] if e.get("kind") == "eval" and e["act_rel_err"] < 0.5), None)
                esc.append(s if s is not None else r["cfg"]["steps"] * 2)  # censored -> 2x budget
            phis.append(phi); steps.append(float(np.mean(esc)))
            enc.append(float(np.mean([r["final"]["agent_r2"] for r in rr])))
        rho = spearman(phis, steps)
        lines.append(f"h={h}: phi {phis} -> action-use escape step {np.round(steps)}; final agent R2 {np.round(enc, 2)}; Spearman {rho:.2f}")
        verdicts.append(rho >= T["g1_hist_spearman"])
    report("G1 history x autocorrelation (deaf predictor)", lines, "PASS" if all(verdicts) else "SEE NUMBERS")


def g1_objective():
    """P3 boundary: warm-up should persist iff the agent is within the top k by ACTION-CONDITIONED error."""
    runs = load_sweep("g1", "g1_objective")
    lines, ok = [], True
    for (noise,), rs in sorted(by(runs, "agent_noise").items()):
        rk = rs[0]["log"][0]
        k = rs[0]["cfg"]["k"]
        errs = [rk["agent_nerr_with_action"]] + list(rk["env_nerr"])
        inside = bool(0 in np.argsort(errs)[:k])
        g = by(rs, "idm")
        alw = {r["cfg"]["seed"]: r["final"]["agent_r2"] for r in g[("always",)]}
        wu = {r["cfg"]["seed"]: r["final"]["agent_r2"] for r in g[("warmup",)]}
        van = {r["cfg"]["seed"]: r["final"]["agent_r2"] for r in g[("none",)]}
        ratios = [wu[s] / max(alw[s], 1e-6) for s in sorted(wu)]
        persists = float(np.mean(ratios)) >= 0.8
        fails = float(np.mean(ratios)) < 0.5
        fit = persists if inside else fails
        ok &= fit
        lines.append(f"agent_noise {noise}: action-conditioned nerr {rk['agent_nerr_with_action']:.3f} vs env {np.round(rk['env_nerr'], 3)} "
                     f"-> inside top-{k}: {inside}; plain {np.mean(list(van.values())):.3f}, always {np.mean(list(alw.values())):.3f}, "
                     f"warm-up/always {np.round(ratios, 2)} -> {'fits P3' if fit else 'DOES NOT FIT P3'}")
    report("G1 trap-vs-objective boundary (P3)", lines, "PASS" if ok else "FAIL (see levels that do not fit)")


# ---- Added with prereg/G1.md part B (theory-derived; registered before any G1 sweep was run) ----
def _theory(cfg):
    from src.analysis.theory_linear import classify
    return classify(cfg)


def _seedmap(rs, key="agent_r2"):
    return {r["cfg"]["seed"]: r["final"][key] for r in rs}


def g1_phase():
    """Restated P3: warm-up persists iff the agent-kept solution is a strict local minimum (c_kept > 0),
    not iff the agent is in the action-conditioned top k. Both rules are scored on the same cells."""
    runs = load_sweep("g1", "g1_phase")
    lines, n_valid, n_s, n_p3, decay_ok, c_ok = [], 0, 0, 0, False, False
    for (rho, noise), rs in sorted(by(runs, "rho_a", "agent_noise").items()):
        th = _theory(rs[0]["cfg"])
        g = by(rs, "idm")
        alw, wu, van = _seedmap(g[("always",)]), _seedmap(g[("warmup",)]), _seedmap(g[("none",)])
        ratios = [wu[s] / max(alw[s], 1e-6) for s in sorted(wu)]
        mr = float(np.mean(ratios))
        obs = "persists" if mr >= 0.8 else ("decays" if mr < 0.5 else "ambiguous")
        valid = float(np.mean(list(alw.values()))) >= T["g1_idm_fix_r2"]
        p_s, p_p3 = th["warmup"], ("persists" if th["inside_action_topk"] else "decays")
        pl_w = np.mean([r["final"]["pred_loss"] for r in g[("warmup",)]])
        pl_n = np.mean([r["final"]["pred_loss"] for r in g[("none",)]])
        if valid:
            n_valid += 1
            n_s += obs == p_s
            n_p3 += obs == p_p3
            decay_ok |= (p_s == "decays" and obs == "decays")
            c_ok |= (th["region"].startswith("C") and obs == "persists")
        lines.append(f"rho_a {rho}, noise {noise}: region {th['region']}, c_kept {th['c_kept']:+.3f}; plain R2 "
                     f"{np.mean(list(van.values())):.3f}, always {np.mean(list(alw.values())):.3f}, warm-up/always "
                     f"{np.round(ratios, 2)} -> {obs}; stability predicts {p_s}, original P3 predicts {p_p3}"
                     f"{'' if valid else ' [INVALID: IDM-always did not fix]'}; pred_loss warm-up {pl_w:.3f} "
                     f"(theory kept {th['pred_loss_soft_kept']:.3f}), plain {pl_n:.3f} (theory evicted {th['pred_loss_soft_evicted']:.3f})")
    acc_s = n_s / max(n_valid, 1)
    lines.append(f"valid cells {n_valid}; stability-rule accuracy {acc_s:.2f}; original-P3 accuracy {n_p3 / max(n_valid, 1):.2f}; "
                 f"a predicted decay cell decayed: {decay_ok}; a multistable (C) cell persisted: {c_ok}")
    ok = n_valid >= T["g1_phase_min_valid"] and acc_s >= T["g1_phase_accuracy"] and decay_ok and c_ok
    report("G1 restated P3: stability boundary (g1_phase)", lines, "PASS" if ok else "FAIL")


def g1_depth():
    """Slow (depth 2, g1_rho) vs fast (depth 1, g1_depth) action pathway under plain training."""
    d1 = load_sweep("g1", "g1_depth")
    d2 = [r for r in load_sweep("g1", "g1_rho") if r["cfg"]["idm"] == "none"]
    lines, ok, discr = [], True, None
    for (rho,), rs1 in sorted(by(d1, "rho_a").items()):
        rs2 = [r for r in d2 if r["cfg"]["rho_a"] == rho]
        th = _theory(rs1[0]["cfg"])
        out = {}
        for name, rs, c, pred in [("depth1", rs1, th["c_evicted_fastV"], th["plain_fastV"]),
                                  ("depth2", rs2, th["c_evicted_slowV"], th["plain_slowV"])]:
            if not rs:
                continue
            obs = "kept" if np.mean([r["final"]["agent_r2"] for r in rs]) >= 0.5 else "evicted"
            p = None if abs(c) < T["g1_c_margin"] else ("evicted" if c > 0 else "kept")
            out[name] = obs
            if p is not None:
                ok &= obs == p
            lines.append(f"rho_a {rho} {name}: c {c:+.3f} -> predicted {p or 'no prediction (|c| < margin)'}; observed {obs} "
                         f"(mean R2 {np.mean([r['final']['agent_r2'] for r in rs]):.3f}, n={len(rs)})")
        if rho == 0.75:
            discr = out.get("depth1") == "kept" and out.get("depth2") == "evicted"
    lines.append(f"discriminating cell rho_a 0.75 (depth 1 kept, depth 2 evicted): {discr}")
    report("G1 slow vs fast action pathway (g1_depth)", lines, "PASS" if ok and discr else "FAIL")


def g1_hist_always():
    """Secondary H-deaf: in IDM-always runs (agent encoded at every phi), action-pathway alignment slows with phi."""
    runs = [r for r in load_sweep("g1", "g1_hist") if r["cfg"]["idm"] == "always"]
    lines, verdicts = [], []
    for (h,), rs in sorted(by(runs, "h").items()):
        phis, steps = [], []
        for (phi,), rr in sorted(by(rs, "phi").items()):
            esc = []
            for r in rr:
                s = next((e["step"] for e in r["log"] if e.get("kind") == "eval" and e["act_rel_err"] < 0.5), None)
                esc.append(s if s is not None else r["cfg"]["steps"] * 2)
            phis.append(phi); steps.append(float(np.mean(esc)))
        rho = spearman(phis, steps)
        lines.append(f"h={h} (IDM-always): phi {phis} -> action-use escape step {np.round(steps)}; Spearman {rho:.2f}")
        verdicts.append(rho >= T["g1_hist_spearman"])
    report("G1 history x autocorrelation, IDM-always runs (secondary H-deaf)", lines, "PASS" if all(verdicts) else "FAIL")


def g1_dose_threshold():
    """P4: a warm-up persists iff, when it ends, the agent's share exceeds the barrier s* and the action pathway is aligned."""
    runs = load_sweep("g1", "g1_dose")
    lines, hit2, hit1 = [], 0, 0
    for r in sorted(runs, key=lambda r: (r["cfg"]["idm_weight"], r["cfg"]["warmup_frac"], r["cfg"]["seed"])):
        c = r["cfg"]
        s_star = _theory(c)["barrier_share"]
        t_end = int(round(c["warmup_frac"] * c["steps"]))
        e = next((e for e in r["log"] if e.get("kind") == "eval" and e["step"] >= t_end), None)
        persist = r["final"]["agent_r2"] >= T["g1_persist_abs"]
        p1 = e["agent_r2"] >= s_star
        p2 = p1 and e["act_rel_err"] < 0.5
        hit1 += p1 == persist
        hit2 += p2 == persist
        lines.append(f"w {c['idm_weight']}, frac {c['warmup_frac']}, seed {c['seed']}: at step {e['step']} R2 {e['agent_r2']:.3f} "
                     f"(s* {s_star:.3f}), act_rel_err {e['act_rel_err']:.2f}; final R2 {r['final']['agent_r2']:.3f} -> "
                     f"{'persists' if persist else 'decays'}; predicted {'persists' if p2 else 'decays'}")
    n = len(runs)
    lines.append(f"accuracy: two-condition rule {hit2}/{n}; R2-only rule {hit1}/{n}")
    report("G1 warm-up dose threshold (P4)", lines, "PASS" if hit2 / n >= T["g1_dose_accuracy"] else "FAIL")


def g1_generic(sweep, keys):
    runs = load_sweep("g1", sweep)
    lines = []
    for k, rs in sorted(by(runs, *keys).items(), key=lambda x: str(x[0])):
        r2 = [r["final"]["agent_r2"] for r in rs]
        mu, lo, hi = boot_ci(r2)
        lines.append(f"{dict(zip(keys, k))}: agent R2 {mu:.3f} [{lo:.3f}, {hi:.3f}] (n={len(r2)}), "
                     f"pred loss {np.mean([r['final']['pred_loss'] for r in rs]):.3f}")
    report(f"{sweep} (descriptive)", lines, "REPORT ONLY")


# ------------------------------- G2 -------------------------------
def g2_calib():
    runs = load_sweep("g2", "g2_calib")
    lines, ok = [], []
    for r in sorted(runs, key=lambda r: r["cfg"]["sigreg_w"]):
        f, d = r["final"], r["cfg"]["d"]
        good = f["cov_err"] < T["g2_cov_err_max"] and T["g2_zvar_lo"] * d <= f["z_var"] <= T["g2_zvar_hi"] * d
        lines.append(f"sigreg_w {r['cfg']['sigreg_w']}: cov_err {f['cov_err']:.3f}, z_var {f['z_var']:.1f} (d={d}) -> {'ok' if good else 'no'}")
        if good:
            ok.append(r["cfg"]["sigreg_w"])
    v = f"CHOOSE sigreg_w = {min(ok)}" if ok else "NONE PASSES: inspect, or use LeWM's exact SIGReg"
    report("G2 calibration (whitening only)", lines, v)


def g2_reproduce():
    runs = load_sweep("g2", "g2_reproduce")
    order = [("set", 2, 16), ("set", 2, 32), ("set", 4, 16), ("set", 4, 32),
             ("integrator", 2, 16), ("integrator", 2, 32), ("integrator", 4, 16), ("integrator", 4, 32)]
    g = by(runs, "agent_mode", "n_dist", "d", "idm")
    lines, chosen = [], None
    for key in order:
        if key + ("none",) not in g or key + ("always",) not in g:
            continue
        v = g[key + ("none",)][0]["final"]["agent_r2"]
        a = g[key + ("always",)][0]["final"]["agent_r2"]
        lines.append(f"{key}: vanilla {v:.3f}, IDM-always {a:.3f}")
        if chosen is None and v < T["g2_vanilla_fail_r2"] and a >= T["g2_idm_fix_r2"]:
            chosen = g[key + ("none",)][0]["cfg"]
    if chosen:
        keep = {k: chosen[k] for k in ("agent_mode", "n_dist", "d", "sigreg_w", "steps") if k in chosen}
        (ROOT / "runs" / "g2" / "CHOSEN_CONFIG.json").write_text(json.dumps(keep, indent=1))
        verdict = f"CHOSEN {keep} (written to runs/g2/CHOSEN_CONFIG.json)"
    else:
        verdict = "NO FAILING CONFIG: pixel arm cannot reproduce the failure; report and follow docs/02 decision rule"
    report("G2 reproduce ladder", lines, verdict)


def g2_trap():
    g = by(load_sweep("g2", "g2_trap"), "idm")
    fin = {k[0]: {r["cfg"]["seed"]: r for r in v} for k, v in g.items()}
    seeds = sorted(fin["none"])
    van = np.array([fin["none"][s]["final"]["agent_r2"] for s in seeds])
    alw = np.array([fin["always"][s]["final"]["agent_r2"] for s in seeds])
    wu = np.array([fin["warmup"][s]["final"]["agent_r2"] for s in seeds])
    lines = [f"seeds {seeds}", f"vanilla {van.round(3)}", f"IDM-always {alw.round(3)}", f"warm-up only {wu.round(3)}",
             f"warm-up / always per seed {(wu / np.maximum(alw, 1e-6)).round(2)}"]
    plan = {}
    for name in ("none", "always", "warmup"):
        vals = []
        for s in seeds:
            pf = list(fin[name][s]["dir"].glob("plan_eval_ckpt_final_s0.json"))
            if pf:
                vals.append(json.loads(pf[0].read_text())["success"])
        if vals:
            plan[name] = float(np.mean(vals))
    if plan:
        lines.append(f"planning success (protocol in plan_eval json): {plan}")
    if van.mean() >= T["g2_vanilla_fail_r2"] or alw.mean() < T["g2_idm_fix_r2"]:
        v = "INCONCLUSIVE: chosen config no longer shows fail-and-fix across seeds"
    elif wu.mean() <= van.mean() + T["g2_kill_margin"]:
        v = "KILL: warm-up does not persist in pixels"
    elif wu.mean() >= T["g2_persist_ratio_mean"] * alw.mean() and (wu >= T["g2_persist_ratio_seed"] * alw).all():
        v = "PASS: warm-up persists in pixels"
        if plan and "warmup" in plan and "always" in plan and plan["warmup"] < T["g2_plan_ratio"] * plan["always"]:
            v += " (but planning does not follow: secondary criterion FAILED)"
    else:
        v = "PARTIAL: see numbers"
    report("G2 decisive: warm-up persistence in pixels", lines, v)


def g2_response():
    runs = load_sweep("g2", "g2_response")
    lines = []
    for (r_,), rs in sorted(by(runs, "agent_r").items()):
        fr = [r["log"][0]["response_fraction"] for r in rs]
        r2 = [r["final"]["agent_r2"] for r in rs]
        lines.append(f"agent_r {r_}: Spearman(response fraction, final agent R2) = {spearman(fr, r2):.2f}")
    for (ne,), rs in sorted(by(runs, "n_echo").items()):
        lines.append(f"n_echo {ne}: Spearman(agent radius, final agent R2) = "
                     f"{spearman([r['cfg']['agent_r'] for r in rs], [r['final']['agent_r2'] for r in rs]):.2f}")
    report("G2 response fraction vs agent size", lines, "REPORT (strengthening)")


def g2_hist():
    runs = load_sweep("g2", "g2_hist")
    lines = []
    for k, rs in sorted(by(runs, "phi", "h", "hist_dropout").items()):
        c = [r["final"]["cas"] for r in rs]; ad = [r["final"]["action_dec"] for r in rs]; a2 = [r["final"]["agent_r2"] for r in rs]
        lines.append(f"phi,h,drop={k}: CAS {np.mean(c):.3f}, action_dec {np.mean(ad):.3f}, agent R2 {np.mean(a2):.3f} (n={len(rs)})")
    report("G2 history x autocorrelation x dropout", lines, "REPORT; apply prereg/G2.md criterion by hand")


def g4_synthetic():
    rows = json.loads((ROOT / "runs" / "g4" / "synthetic.json").read_text())
    lines, ok = [], True
    for an in sorted({r["aniso"] for r in rows}):
        for pn in sorted({r["pred_noise"] for r in rows}):
            rr = [r for r in rows if r["aniso"] == an and r["pred_noise"] == pn]
            o, gw, un = (np.mean([r[k] for r in rr]) for k in ("ours_ctrl", "gw_ctrl", "ours_unctrl"))
            lines.append(f"aniso {an}, pred_noise {pn}: ours(ctrl) {o:.3f}, ours(unctrl) {un:.3f}, GW(ctrl) {gw:.3f}")
            if pn <= 0.05:
                ok &= o >= T["g4_ours_ctrl"] and abs(un) <= T["g4_unctrl_max"]
                if an == 0.0:
                    ok &= gw <= T["g4_gw_ctrl_max"]
    report("G4 synthetic alignment", lines, "PASS" if ok else "FAIL")


GATES = {"g1_trap": g1_trap, "g1_rho": g1_rho, "g1_hist": g1_hist,
         "g1_dose": lambda: g1_generic("g1_dose", ["warmup_frac", "idm_weight"]),
         "g1_init": lambda: g1_generic("g1_init", ["act_init", "act_depth"]),
         "g1_objective": g1_objective,
         "g1_reg": lambda: g1_generic("g1_reg", ["reg", "idm"]),
         "g1_unstable": lambda: g1_generic("g1_unstable", ["idm"]),
         "g1_calib": lambda: g1_generic("g1_calib", ["rho_a", "k", "idm"]),
         "g1_phase": g1_phase, "g1_depth": g1_depth, "g1_hist_always": g1_hist_always,
         "g1_dose_threshold": g1_dose_threshold,
         "g2_calib": g2_calib, "g2_reproduce": g2_reproduce, "g2_trap": g2_trap,
         "g2_response": g2_response, "g2_hist": g2_hist, "g4_synthetic": g4_synthetic}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    for n in (GATES if names == ["all"] else names):
        try:
            GATES[n]()
        except SystemExit as e:
            print(f"{n}: {e}")
