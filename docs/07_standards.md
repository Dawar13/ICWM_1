# 07. Experiment standards

## Pre-registration (non-negotiable)

- Before every sweep, its file in `prereg/` must state:
  - hypothesis;
  - metric;
  - seeds;
  - levels;
  - thresholds;
  - what counts as PASS, KILL and INCONCLUSIVE.
- **Templates:**
  - G1, G2 and G4 are pre-filled. Finalise each and add the date before running.
  - Create `prereg/G3.md` yourself before G3.
- **Thresholds** in `src/analysis/gates.py` (THRESHOLDS) must equal those in `prereg/`.
- **Changing a sweep after seeing results** requires a new pre-registration file with a reason, and the paper must say so.
- **Disclosure:** the G1 default regime was chosen after the smoke tests in `notes/smoke_tests.md`. The pre-registrations must say this.

## Reproducibility

- Every run directory holds:
  - `meta.json` (full config, config hash, code hash, host);
  - `log.jsonl` (curves);
  - `summary.json` (final metrics);
  - `stdout.txt`.
- Never edit run outputs by hand. Figures come only from scripts (put plotting code in `src/analysis/`).

## Statistics

- Seeds per pre-registration (5 for G1 primary, 3 for most others).
- Report the mean with 95% bootstrap intervals (`gates.boot_ci`) and per-seed values for primary gates.
- Use paired comparisons (same seed, same data) wherever possible.
- Censored outcomes ("never escaped within the budget") are reported as censored, not dropped.

## Protocol discipline

- Planning numbers always carry their protocol: episodes, horizon, samples, elites, iterations, budget, tolerance and seed (`plan_eval.py` writes them).
- Never compare our numbers with a published number unless every protocol element matches. Quote published numbers only as context.

## Honesty

- Negative and null results go in `notes/results_summary.md` and, later, the appendix.
- A KILL is a successful outcome of this phase.
- Limitations are written as specific statements ("fails when ...").

## Logs

- `notes/daily_log.md`: append at the end of every session. Record what was done, what was found, the gate verdicts, the compute used and the next step.
- `notes/compute_log.md`: copy the totals from `runs/compute_ledger.csv`.
