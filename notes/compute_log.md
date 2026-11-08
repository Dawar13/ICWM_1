# Compute log

Copy the totals from `runs/compute_ledger.csv` after each sweep. GPU = the shared A100.

| Date | Sweep / run | Device | Runs | Wall hours (sum) | Running GPU total | Note |
|---|---|---|---|---|---|---|
| 2026-11-08 | smoke tests (kit build) | CPU | about 40 | small | 0 | not in this machine's ledger |
| 2026-11-08 | g2_calib timing run (500 steps, 4 runs, outputs deleted) | GPU (shared A100) | 4 | 0.033 | 0.033 | 33 steps/s, 0.16 GB peak |
| 2026-11-08 | G1 batch, first attempt (4 parallel, thrashed, stopped; partial runs deleted) | CPU (local) | 0 finished | about 0.3 (not in ledger) | 0.033 | restarted with 2 parallel |
