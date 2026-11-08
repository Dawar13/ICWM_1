# Daily log (append at the end of every session)

**Format:**
- Date
- What was done
- What was found (gate verdicts with numbers)
- Candidates live or killed
- Compute used (CPU and GPU hours from runs/compute_ledger.csv)
- GPU sharing notes (free memory seen, any waits)
- Next step

---

## 2026-11-08 (kit built in claude.ai)

- **Done:** built this kit. Smoke-tested all code on CPU (see notes/smoke_tests.md).
- **Found (exploratory, not evidence):** in the linear model a trap appeared, and in 2 seeds a warm-up escaped it for good. Synthetic alignment works as designed.
- **Candidates:** #1 lead, #3 fallback, #2 parked.
- **Compute:** CPU only.
- **Next:** Stage 0 in docs/02_gonogo_protocol.md.

## 2026-11-08 (session 1, Claude Code; entry written mid-session)

**Done.**
- **Setup.**
  - Kit copied to the shared A100 at `~/icwm_gonogo`; `setup.sh` gave torch 2.15 cu132 with CUDA.
  - The link from this workstation to the A100 is poor: about half of ssh connections stall, so a retry wrapper is used.
  - Local `.venv` reuses system torch 2.11 CPU, plus pot.
- **GPU check at start:**
  - 55 GB of 82 GB used, 100% utilisation, 6 processes of other users.
  - All of those processes run as the same Unix user `dawar`, so we only ever act on PIDs we started.
  - Later readings: 71 GB used, then 51 GB used at 36%.
- **Host CPU:** 12 vCPUs, 0% idle, run queue about 40. G1 therefore runs on the local workstation CPU instead (prereg/G1.md).
- **Stage 0.**
  - Lit verification and collision search: notes/lit_table.md, notes/collision_check.md.
  - LeWM code diff: notes/lewm_diff.md.
- **Stage 1.**
  - Derivation in notes/theory.md, with closed forms and numeric checks in src/analysis/theory_linear.py.
  - Fresh-subagent referee check is running.
  - prereg/G1.md finalised (code f1da5d74637e), with part B: theory predictions, new sweeps g1_phase and g1_depth, and four added analyses.
- **G1 batch:**
  - All registered sweeps launched locally.
  - The first attempt with 4 processes thrashed (about 9500 pages/s, working sets trimmed to 30 MB). Stopped, partial runs deleted, restarted with 2 processes: about 70 steps/s each.
- **G2:**
  - Timing run: 33 steps/s, 0.16 GB peak, outputs deleted.
  - prereg/G2.md finalised for calibration and reproduce.
  - Stage A (g2_calib, then g2_reproduce) launched on the A100 with `--parallel 2`, because utilisation was low and memory plentiful.
  - g2_trap is on hold.

**Found.**
- **Collision check:** COLLISION FOUND on learning order (P1), from the abstract of an anonymous ICLR 2027 submission, OpenReview t6sUZeEbkt, "When Prediction Chooses the Representation". Its full text is behind a browser check: a human must read it.
  - It is unknown whether it shows a trap (a higher-loss solution) or a constraint switched off with persistence tested.
  - No collision was found for warm-up persistence.
- **Other literature points:**
  - 2603.02263 Social-JEPA is WITHDRAWN.
  - 2610.19375 is by Cheung and He, not Cheung and Zhang.
- **Theory:**
  - The trap-vs-objective boundary is a stability condition, c_kept > 0, not "agent in the action-conditioned top k".
  - The theory predicts that the pre-registered g1_objective gate FAILS at noise 0.6 (multistability) and that g1_hist FAILS (the trap confounds it).
  - P1 is explained by the slow factorised action pathway.
  - The reduced theory reproduces the smoke-test losses (0.342 and 0.170) without fitting.
- **LeWM:**
  - AdaLN-zero action conditioning; full target gradient; SIGReg weight 0.09.
  - Our SIGReg is effectively 3.6 to 14 times weaker.

**Verdicts:** none yet.

**Candidates:** #1 live, pending the user's reading of t6sUZeEbkt; #3 fallback; #2 parked.

**Compute:**
- GPU 0.033 h (timing run).
- CPU: G1 running locally.

**Next:** referee verdict on theory.md; G1 gates as sweeps finish; G2 stage A gates; G4a synthetic; g2_trap after the user's go-ahead.

### 2026-11-08, later the same session

- **G4a synthetic: PASS.**
  - Ours on the controllable subspace: 0.997 at predictor noise 0.05 and 1.000 at 0.
  - Ours on the uncontrollable subspace: 0.12 to 0.18.
  - Gromov-Wasserstein: -0.04 to 0.07, flat with anisotropy.
  - 10 seeds; seed 0 had been seen in the smoke tests. See runs/gate_reports.md.
- **G2 calibration:**
  - All four weights pass the whitening rule (BatchNorm on the latent makes whitening easy), so the rule chose sigreg_w = 0.03.
  - The LeWM-equivalent weight for d = 16, h = 1 would be 0.72, so our regulariser is about 24 times weaker than LeWM's.
- **G2 reproduce: CHOSEN (set, n_dist 2, d 16, sigreg_w 0.03, 15000 steps).**
  - Seed 0, plain: agent_r2 -0.016, CAS 0.0000, pred_loss 0.249 against copy_loss 0.246 (no better than copying).
  - IDM-always: agent_r2 1.000, pred_loss 0.174.
  - So IDM-always reaches a lower prediction loss: in pixels it looks like the linear region A. One seed only, so suggestive, not evidence.
  - BUG in the kit's `g2_reproduce` (src/sweeps.py): run names omit agent_mode, n_dist and d, so all 8 rungs share two directories. Only the first rung ran; the rest were "skipped (done)". This does not change the pre-registered outcome, because the rule picks the first qualifying rung in order and that rung qualified. The other rungs (including the integrator agent) were not run.
  - The fix is deferred until the G1 batch is finished, so G1 runs keep one code hash.
  - The skipped runs added 14 rows of 0.0028 h each to the GPU ledger.
- **G1 batch:** restarted with g1_dose second (it sets the g2_trap warm-up dose), then objective, phase, depth, rho and the rest. Partial runs were deleted before each restart.
- **GPU used so far:** 0.48 h (runs/compute_ledger_gpu.csv).
- **Theory referee verdict:**
  - C1 to C4 and C6 VERIFIED.
  - C5 CONDITIONAL: the slow pathway sets the *speed* of escape, not the landscape; the plateau scales as 1/v0². Recorded in notes/theory.md.
  - prereg/G1.md amendments 1 and 2 added. Amendment 2 is H-ema: warm-up does not persist with an EMA target. It was added before g1_reg ran.
- **G1 g1_trap: PASS (5/5).**
  - Warm-up persists 18000 steps after IDM is switched off.
  - pred_loss matches the theory's pre-registered values: 0.337 vs 0.342 for plain, 0.171 vs 0.170 for warm-up.
- **G1 g1_dose:**
  - Doses from 0.5% at weight 10 upwards persist.
  - P4′ PASS 26/30. The R²-only rule scored 30/30, and the action-pathway condition was refuted (see results_summary).
  - The G2 warm-up dose is kept at the default (prereg/G2.md amendment 1).
