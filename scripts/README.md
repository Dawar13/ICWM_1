# Launch scripts

These are the scripts that actually launched the sweeps, kept for provenance.

| Script | Where it ran | What it does |
|---|---|---|
| `run_g1_batch.sh` | Local workstation CPU | Runs every registered G1 sweep in order, 2 processes at a time. Finished runs are skipped, so it is safe to restart. Delete partial run folders (no `summary.json`) before restarting, because `log.jsonl` is opened in append mode |
| `g2_stage_a.sh` | Shared A100 host, `~/icwm_gonogo` | Runs g2_calib, applies the whitening rule, then runs g2_reproduce with the chosen `sigreg_w` |

The ssh retry wrapper used to reach the A100 is deliberately not in the repository, because it contains the key path and the host address.
