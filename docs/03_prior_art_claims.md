# 03. Prior art, and what we may and may not claim

**Status tags:**
- [V] = fetched and read by us;
- [S] = seen in search results or the research report; re-verify before citing;
- [U] = unverified.

Mark each [V] in `notes/lit_table.md` once fetched.

## Directly about action blindness in JEPA world models (all 2026)

| Paper | Status | Use |
|---|---|---|
| Keeping JEPA World Models Plannable When Little of the Frame Moves (Strohm, Wagner, Schwab, Huber), 2610.03137 | [V] abstract on 7 Nov; [S] details | The phenomenon. LeWM succeeds on 3 of 1200 SLIM trials. Inverse dynamics takes success from 0.003 to 0.348. Response fraction matters; object size does not. Appendix H: the SIGReg weight is not the cause. The repair goes through the encoder |
| Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models (Toso, LeCun, Anderson, Bounou), 2610.07540 | [V] | Objective-level result: the loss can be minimised with controllable unstable modes collapsed; endpoint inverse dynamics |
| ActSWM, 2607.26712 | [S] | Already names the history shortcut: the predictor "becoming insensitive to the conditioning action sequence" |
| Overcoming Statistical Bias in Action-Controllable World Models (Shi et al.), 2608.04653 | [S] | Already names the shortcut; ARC and Drift Energy metrics |
| AVL-JEPA, 2610.03587 | [S] | "Causal dynamics information collapse"; static Bayes-risk argument |
| CF-JEPA 2610.00727; AD-WM 2609.30264; Action-Consequence Alignment 2610.04539; Delta-JEPA 2606.31232; No Gaussian Required 2608.17542; Toward Physically Grounded JEPA World Models 2609.03565 | [S] | Fixes that stay on throughout training |
| Sensorimotor World Models (Ivashkov, Balestriero, Schölkopf), 2606.20104 | [V] earlier | Inverse dynamics keeps controllable degrees of freedom; theory left for future work |
| PhyLatent, 2608.05720 | [V] abstract | "Counterfactual Dynamics Collapse"; "do not isolate a unique internal cause" |
| Spectral-Target, 2609.04264 | [S] | "Physical representation laziness". Check the body before quoting "control ball" |
| One-Step Prediction Fidelity Is a Poor Proxy (Cheung and Zhang), NeurIPS 2026 WM-PAI workshop; 2610.19375 | [V] abstract | One-step error vs planning: r = -0.010 over 15 checkpoints |
| The Intervention Gap (Vakalis), 2608.29998 | [S] | Leaves the mechanism open |
| ACPC, 2608.12939 | [V] abstract | Diagnostic for perturbation robustness; not our claim |

## Learning dynamics and theory to build on

| Paper | Status | Use |
|---|---|---|
| On the Stepwise Nature of SSL (Simon et al.), 2303.15438 | [S] | Stepwise eigenmode learning; no actions |
| How JEPA Avoids Noisy Features (Littwin et al.), 2407.03475 | [S] | JEPA implicit bias; no actions |
| A Unifying Framework for Action-Conditional Self-Predictive RL (Khetarpal et al.), 2406.02035, AISTATS 2025 | [V] | Fixed points of BYOL-AC with per-action predictors; no ordering or timing |
| Saxe et al., deep linear dynamics; saddle-to-saddle literature | [U] | Background; cite precisely |
| Klindt, LeCun, Balestriero, 2605.26379 | [S] | Identifiability up to rotation (needed for #3) |
| Zhang et al., On the Identifiability of Controlled World Models, 2607.22430 | [S] | When action effects are identifiable |
| Wen et al., Fighting Copycat Agents, NeurIPS 2020, 2010.14876 | [V] search | Analogy for the history shortcut |
| CAER, 2608.30897 | [V] abstract | Video world models under-train sparse action effects |
| AC-State (Lamb et al.) 2207.08229; Efroni et al. (Ex-BMDP); Multistep Inverse Is Not All You Need 2403.11940 | [S] | Inverse models recover controllable state (RL theory) |

## Claims

| We may claim, if the gates pass | We may NOT claim |
|---|---|
| A learning-dynamics account of when action use is acquired: passive ranking early, action-conditioned ranking at the optimum | A new failure mode (taken) |
| The trap vs objective boundary, with a falsifiable warm-up persistence test | A new diagnostic or probe (taken) |
| That a short warm-up suffices inside the boundary, and the minimum dose | Inverse dynamics as a new fix (taken) |
| A reconciliation of Toso et al. (objective level) with SLIM and the inverse-dynamics papers (empirical) | Anything about LeWM at scale beyond what G3 measures |
| | Any number we did not measure under our protocol |
