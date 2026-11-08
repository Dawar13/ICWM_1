#!/usr/bin/env bash
# G1 batch on the local CPU (prereg/G1.md). Sweeps run in order; each skips finished runs, so it is safe to restart.
cd "/c/Users/djtde/Downloads/ICWM_1"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in g1_trap g1_dose g1_objective g1_phase g1_depth g1_rho g1_hist g1_reg g1_init g1_unstable; do
  echo "=== $s $(date)"
  .venv/Scripts/python src/run_sweep.py $s --parallel 2 | grep -vE "^started"
done
echo "=== ALL G1 DONE $(date)"
