# Full context: ICWM 2027 go/no-go

Read this once at the start. It holds everything a fresh Claude Code session needs; the docs go deeper.

## 1. Mission

- **Who:** Erdős AI Lab (Bengaluru).
- **What:** one strong paper for **ICWM 2027**, the International Conference on World Models (1st edition, San Francisco Bay Area, late May 2027).
- **Organisers:** created by Randall Balestriero (Brown / AMI), Amir Zadeh (Lambda) and Yann LeCun (NYU / AMI), per the icwm.cc About page as quoted by our research run [R].
- **The bar:** spotlight is the minimum, oral is the aim.
- **The real goal:** a result that LeCun and Balestriero would find important, because it is about *their* models (JEPA world models with SIGReg, such as LeWorldModel), and that opens a route to AMI Labs.

### Venue facts

Fetched from icwm.cc/call-for-papers on 7 Nov 2026 [V]:

| Item | Status |
|---|---|
| Submission deadline | **30 Nov 2026**; time zone not stated ("exact deadline times to be confirmed") |
| Reviews released | 11 Jan 2027 |
| Rebuttal | 11 to 18 Jan 2027 |
| Final decisions | 8 Feb 2027 |
| CFP status | Labelled a "working draft"; all dates provisional |
| Not stated | Page limit, tracks, oral/spotlight tiers |

**Re-check the page before submitting.**

## 2. How we got here

- About ten rounds of ideation and collision checks (in our earlier kit, `icwm_kit/docs/01-17`) killed most ideas. The graveyard is in `docs/09_graveyard.md`.
- An extended literature search (8 Nov 2026; report in `references/ai_generated/research_report_2026-11-08.md`, untrusted until verified) changed the plan:
  - **Already published, Aug to Oct 2026:** the "blind or deaf" diagnosis of why JEPA world models ignore the agent, the action-sensitivity probe, and inverse-dynamics fixes.
  - **Still open:** a *learning-dynamics* account of when and why an action-conditioned JEPA world model learns to use its actions, and whether the failure is an **optimisation trap** or a **flaw in the objective**.

## 3. The three candidates (ranked)

### #1 (lead): "Listening Is Learned Last"

A theory of when JEPA world models learn to use their actions.

- **Plain claim:** a JEPA world model can settle into a solution that ignores the agent even when keeping the agent would give a *lower* value of its own loss. It is a training trap, not the true optimum. A short one-off push out of the trap (inverse dynamics as a warm-up, then switched off) is enough, wherever the loss itself prefers the agent.
- **What it explains:**
  - why LeWM-style models fail when little of the frame responds to actions (SLIM, 2610.03137);
  - why inverse dynamics fixes it (seven or more 2026 papers);
  - when the fix can be removed;
  - how this squares with Toso, LeCun et al. (2610.07540), who prove the objective *can* drop controllable modes.
- **Deciding experiment:** warm-up persistence. Train with inverse dynamics briefly, switch it off, and see whether the agent stays encoded.
  - If it stays, we have a general, cheap, surprising result about LeCun and Balestriero's own model.
  - If it decays, the failure is the objective's (Toso territory) and #1 is dead.
- **Do NOT claim** (already published):
  - a new failure mode, the "deaf predictor" or history shortcut (ActSWM 2607.26712; Shi et al. 2608.04653);
  - a new diagnostic (SLIM probe, ACPC 2608.12939, AD-WM 2609.30264);
  - inverse dynamics as a fix (seven or more papers).
- Full detail in `docs/01_lead_theory.md`.

### #3 (fallback, mostly inference): "What You Can Control, You Can Align"

- **The problem:** SIGReg-trained world models are identifiable only up to a rotation, and their latents are close to isotropic Gaussian, so shape-only alignment (optimal transport, Gromov-Wasserstein) cannot find the map between two independently trained models.
- **The method:** shared actions fix the map on the controllable subspace, and only there.
- **Odds:** cheap and clean, but in a crowded area. Spotlight odds are low; it is the fallback if #1 is killed.
- See `docs/04_candidate3_align.md`.

### #2 (separate effort, not in this go/no-go): "Look Before You Leap"

- **The idea:** observability-gated contingency planning, in which probing actions emerge without a curiosity bonus. It targets LeCun's problem of planning with latent variables.
- **Why not now:** it needs a new environment and a mode-latent model. Run it only if a teammate takes it. See `docs/05_candidate2_lbyl.md`.

## 4. What has already been run

See `notes/smoke_tests.md` (exploratory; not evidence). Headlines:

- **Linear model, capacity competition, action-set agent:** plain training drops the agent and ends at a *worse* prediction loss than inverse dynamics always on (0.34 vs 0.17; 2 seeds). An 800-step inverse-dynamics warm-up kept the agent to the end (2 seeds); a 30-step warm-up did not.
- **No trap without competition:** with spare latent capacity, or with a large persistent agent, there was nothing to explain.
- **Synthetic alignment:** action-response Procrustes recovers the controllable map; Gromov-Wasserstein does not.

## 5. Compute

- **Hardware:** a friend's **A100**, shared with 2 or 3 other experiments. Rules in `docs/06_shared_gpu.md`.
- **What goes where:** G1 (linear) and G4-synthetic run on CPU; G2 (tiny pixel model) runs on the GPU with a 6 GB memory cap per process; G3 is inference only.
- **Logging:** log every run (`runs/compute_ledger.csv`, then `notes/compute_log.md`).
- **Ask the user before:**
  - any single run expected to exceed 2 GPU-hours;
  - the go/no-go phase passing 20 GPU-hours in total.
- **No schedule:** we work at agentic speed. Do not plan in days or hours; plan in gates.

## 6. Order of work (details in `docs/02_gonogo_protocol.md`)

1. **Setup.**
   - Create the environment (`setup.sh`).
   - Verify the key papers (`notes/lit_table.md`).
   - Diff against LeWM's real code (`notes/lewm_diff.md`).
2. **G1 linear theory and sweeps (CPU).**
   - Derive the predictions first.
   - Then run the pre-registered sweeps: g1_trap is primary.
3. **G4 synthetic alignment (CPU).** Cheap insurance for #3.
4. **G2 pixels (shared GPU).** Calibrate, reproduce the failure, then run g2_trap, the decisive pixel test.
5. **G3** LeWM checkpoints (inference) and **G4b** tiny-model alignment, as strengthening.
6. **Decision gate:** stop and report to the user with `notes/decision.md` filled.

## 7. Rules that matter most

- **No fabrication.** Every number is measured by us, quoted with its protocol, or a pre-registered threshold. Every citation is fetched first.
- **Pre-register before every sweep** (`prereg/`). The G1 default regime was chosen after smoke tests; say so.
- **Report negative results.** A clean KILL is a good outcome: it saves the deadline for #3.
- **Writing style:** British spelling, no em dashes, no emoji. Plain language first, maths only where it makes a prediction.
