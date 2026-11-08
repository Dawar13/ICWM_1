#!/usr/bin/env bash
# G2 stage A on the shared A100 (prereg/G2.md): calibrate SIGReg by the whitening rule, then the reproduce ladder.
set -u
cd ~/icwm_gonogo
PY=.venv/bin/python
echo "=== g2_calib $(date)"
$PY src/run_sweep.py g2_calib --parallel 2
$PY src/analysis/gates.py g2_calib | tee runs/g2/calib_gate.txt
W=$(grep -oP 'CHOOSE sigreg_w = \K[0-9.]+' runs/g2/calib_gate.txt | tail -1)
if [ -z "$W" ]; then echo "=== STAGE A STOPPED: no sigreg_w passes the whitening rule $(date)"; exit 1; fi
echo "chosen sigreg_w=$W"
echo "=== g2_reproduce $(date)"
$PY src/run_sweep.py g2_reproduce --parallel 2 --override "{\"sigreg_w\": $W}"
$PY src/analysis/gates.py g2_reproduce | tee runs/g2/reproduce_gate.txt
echo "=== STAGE A DONE $(date)"
