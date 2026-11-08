# 02. Go/no-go protocol

The work is organised as gates, not dates. Each gate has a pre-registration in `prereg/`, a sweep in `src/sweeps.py`, and a verdict from `src/analysis/gates.py`. Verdicts are appended to `runs/gate_reports.md`.

## Stage 0: setup (no GPU)

1. Run `bash setup.sh`. It creates `.venv` inside this folder and needs no sudo.
2. Check that GPU sharing works: `python -c "from src.common.gpu import gpu_report; print(gpu_report())"`. Record the free memory in `notes/daily_log.md`.
3. Verify the key papers. Fetch each one in `notes/lit_table.md` and mark it [V], or [U] if unreachable. Above all:
   - SLIM 2610.03137 (including appendix H);
   - Toso et al. 2610.07540;
   - ActSWM 2607.26712;
   - Shi et al. 2608.04653;
   - AVL-JEPA 2610.03587;
   - Khetarpal et al. 2406.02035;
   - Littwin et al. 2407.03475.
4. **Collision check for #1.** Search arXiv and OpenReview for "learning dynamics", "warm-up", "inverse dynamics schedule", "action-conditioned JEPA saddle" and similar. A paper that already shows warm-up persistence or the trap-vs-objective boundary changes everything: stop and tell the user.
5. Read the LeWM code and fill `notes/lewm_diff.md` (see `src/g3_lewm_ckpt/README.md`).

## Stage 1: G1 linear theory (CPU)

1. **Derive P1 to P4** in the linear model (`docs/01_lead_theory.md`). Write `notes/theory.md`. A fresh subagent checks the derivations without seeing our hopes, only the setup and the claims.
2. **Finalise `prereg/G1.md`.** Update the predictions from the derivation; the thresholds stay as written unless you record why.
3. **Run the sweeps** (`python src/run_sweep.py <name>`):
   - `g1_trap` (primary);
   - `g1_rho`, `g1_objective`, `g1_dose`, `g1_hist`, `g1_init`, `g1_reg`, `g1_unstable`.
4. **Get the verdicts:** `python src/analysis/gates.py g1_trap g1_rho g1_hist g1_dose g1_objective g1_init g1_reg g1_unstable`.

**G1 gate:**

| Outcome | Condition | Action |
|---|---|---|
| **GO** | `g1_trap` PASS, and `g1_objective` shows warm-up persistence failing once the agent is no longer in the action-conditioned top k | Continue |
| **KILL #1** | `g1_trap` KILL (warm-up never persists, even inside the predicted boundary) | Move to #3 |
| **INCONCLUSIVE** | The regime does not show fail-and-fix | Run `g1_calib` (exploratory), write a new pre-registration with the reason, and re-run once. If still inconclusive, report to the user |

## Stage 2: G4 synthetic alignment (CPU; insurance for #3)

1. Run `python src/g4_align/align_synthetic.py --out runs/g4/synthetic.json --seeds 10`.
2. Then run `python src/analysis/gates.py g4_synthetic`.

## Stage 3: G2 pixels (shared GPU)

1. **Timing run first.** Run `python src/run_sweep.py g2_calib --override '{"steps": 500}'`, then delete those outputs. Record steps per second and peak memory, and estimate the cost of each sweep before launching it. If any single run would exceed 2 GPU-hours, ask the user.
2. **Calibrate.** Run `g2_calib` and apply the whitening rule (`gates.py g2_calib`). Put the chosen `sigreg_w` into the `g2_reproduce` base config, or use LeWM's exact SIGReg and weight if `notes/lewm_diff.md` gives them.
3. **Reproduce.** Run `g2_reproduce`. The rule (`gates.py g2_reproduce`) writes `runs/g2/CHOSEN_CONFIG.json`. If no config fails, the pixel arm is NO-GO: report it.
4. **Decisive test.**
   - Run `g2_trap`.
   - Then plan-evaluate every final checkpoint: `python src/g2_pixel/plan_eval.py --run <dir>` for each run.
   - Then run `gates.py g2_trap`.
5. **Strengthening:** `g2_response`, `g2_hist`, `g2_ema`.

**G2 gate:**

| Outcome | Condition | Action |
|---|---|---|
| **GO (pixel-confirmed)** | `g2_trap` PASS | #1 is the lead |
| **PARTIAL** | G1 GO but G2 KILL | #1 survives only as theory plus synthetic, with a pixel negative result. The ceiling drops. Report to the user with both options (#1 reduced, or switch to #3) |
| **KILL #1** | G1 KILL | Already decided at Stage 1 |

## Stage 4: strengthening (only if #1 is GO)

- **G3:** LeWM checkpoints, inference only (`src/g3_lewm_ckpt/README.md`).
- **G4b:** align two tiny pixel models trained with different seeds and h = 1 (`src/g4_align/align_tiny_models.py`). This keeps #3 alive as a possible second submission.

## Decision gate (stop and ask the user)

Fill `notes/decision.md`:
- each gate's verdict, with the numbers and protocol;
- what the paper can claim and what it cannot;
- compute used;
- the recommendation: #1, #1-reduced or #3.

Do not start writing the paper before the user decides.
