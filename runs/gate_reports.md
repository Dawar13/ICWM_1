
## G4 synthetic alignment  (code f1da5d74637e)
aniso 0.0, pred_noise 0.0: ours(ctrl) 1.000, ours(unctrl) 0.180, GW(ctrl) -0.041
aniso 0.0, pred_noise 0.05: ours(ctrl) 0.997, ours(unctrl) 0.121, GW(ctrl) -0.041
aniso 0.0, pred_noise 0.2: ours(ctrl) 0.956, ours(unctrl) 0.122, GW(ctrl) -0.041
aniso 0.5, pred_noise 0.0: ours(ctrl) 1.000, ours(unctrl) 0.180, GW(ctrl) 0.056
aniso 0.5, pred_noise 0.05: ours(ctrl) 0.997, ours(unctrl) 0.121, GW(ctrl) 0.056
aniso 0.5, pred_noise 0.2: ours(ctrl) 0.956, ours(unctrl) 0.122, GW(ctrl) 0.056
aniso 1.0, pred_noise 0.0: ours(ctrl) 1.000, ours(unctrl) 0.180, GW(ctrl) 0.072
aniso 1.0, pred_noise 0.05: ours(ctrl) 0.997, ours(unctrl) 0.121, GW(ctrl) 0.072
aniso 1.0, pred_noise 0.2: ours(ctrl) 0.956, ours(unctrl) 0.122, GW(ctrl) 0.072

**VERDICT: PASS**

<!-- The verdicts below were produced on the A100 host (~/icwm_gonogo/runs/gate_reports.md) and copied here on 8 Nov 2026. -->

## G2 calibration (whitening only)  (code f1da5d74637e)
sigreg_w 0.03: cov_err 0.080, z_var 30.4 (d=32) -> ok
sigreg_w 0.1: cov_err 0.040, z_var 30.2 (d=32) -> ok
sigreg_w 0.3: cov_err 0.040, z_var 31.4 (d=32) -> ok
sigreg_w 1.0: cov_err 0.034, z_var 31.2 (d=32) -> ok

**VERDICT: CHOOSE sigreg_w = 0.03**

## G2 reproduce ladder  (code f1da5d74637e)
('set', 2, 16): vanilla -0.016, IDM-always 1.000

**VERDICT: CHOSEN {'agent_mode': 'set', 'n_dist': 2, 'd': 16, 'sigreg_w': 0.03, 'steps': 15000} (written to runs/g2/CHOSEN_CONFIG.json)**

## G1 trap / warm-up persistence (primary)  (code f1da5d74637e)
seeds [0, 1, 2, 3, 4]
vanilla agent R2 [-0.002 -0.    -0.001 -0.003  0.   ]
IDM-always   [0.997 0.997 0.997 0.998 0.998]
warm-up only [0.997 0.997 0.997 0.997 0.998]
pred loss vanilla [0.347 0.339 0.339 0.341 0.32 ] vs IDM-always [0.16  0.165 0.154 0.168 0.168]
vanilla fails in 5, IDM fixes in 5, warm-up persists in 5, kill-condition in 0 seeds
IDM-always reaches a LOWER prediction loss in 100% of seeds (trap, not objective, if high)

**VERDICT: PASS: trap that a one-off warm-up escapes, and the escaped solution is better under the JEPA loss**

## g1_dose (descriptive)  (code f1da5d74637e)
{'warmup_frac': 0.005, 'idm_weight': 10.0}: agent R2 0.997 [0.997, 0.997] (n=3), pred loss 0.170
{'warmup_frac': 0.005, 'idm_weight': 3.0}: agent R2 0.332 [-0.002, 0.997] (n=3), pred loss 0.283
{'warmup_frac': 0.01, 'idm_weight': 10.0}: agent R2 0.997 [0.997, 0.997] (n=3), pred loss 0.170
{'warmup_frac': 0.01, 'idm_weight': 3.0}: agent R2 0.997 [0.997, 0.997] (n=3), pred loss 0.170
{'warmup_frac': 0.02, 'idm_weight': 10.0}: agent R2 0.997 [0.997, 0.997] (n=3), pred loss 0.170
{'warmup_frac': 0.02, 'idm_weight': 3.0}: agent R2 0.997 [0.997, 0.997] (n=3), pred loss 0.170
{'warmup_frac': 0.05, 'idm_weight': 10.0}: agent R2 0.997 [0.997, 0.997] (n=3), pred loss 0.170
{'warmup_frac': 0.05, 'idm_weight': 3.0}: agent R2 0.997 [0.997, 0.997] (n=3), pred loss 0.170
{'warmup_frac': 0.1, 'idm_weight': 10.0}: agent R2 0.997 [0.997, 0.997] (n=3), pred loss 0.170
{'warmup_frac': 0.1, 'idm_weight': 3.0}: agent R2 0.997 [0.997, 0.997] (n=3), pred loss 0.170

**VERDICT: REPORT ONLY**

## G1 warm-up dose threshold (P4)  (code f1da5d74637e)
w 3.0, frac 0.005, seed 0: at step 100 R2 0.269 (s* 0.383), act_rel_err 1.00; final R2 -0.002 -> decays; predicted decays
w 3.0, frac 0.005, seed 1: at step 100 R2 0.347 (s* 0.383), act_rel_err 1.00; final R2 -0.000 -> decays; predicted decays
w 3.0, frac 0.005, seed 2: at step 100 R2 0.997 (s* 0.383), act_rel_err 0.99; final R2 0.997 -> persists; predicted decays
w 3.0, frac 0.01, seed 0: at step 200 R2 0.997 (s* 0.383), act_rel_err 0.08; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.01, seed 1: at step 200 R2 0.997 (s* 0.383), act_rel_err 0.04; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.01, seed 2: at step 200 R2 0.997 (s* 0.383), act_rel_err 0.03; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.02, seed 0: at step 400 R2 0.997 (s* 0.383), act_rel_err 0.01; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.02, seed 1: at step 400 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.02, seed 2: at step 400 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.05, seed 0: at step 1000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.05, seed 1: at step 1000 R2 0.997 (s* 0.383), act_rel_err 0.01; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.05, seed 2: at step 1000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.1, seed 0: at step 2000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.1, seed 1: at step 2000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 3.0, frac 0.1, seed 2: at step 2000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.005, seed 0: at step 100 R2 0.997 (s* 0.383), act_rel_err 0.98; final R2 0.997 -> persists; predicted decays
w 10.0, frac 0.005, seed 1: at step 100 R2 0.997 (s* 0.383), act_rel_err 0.99; final R2 0.997 -> persists; predicted decays
w 10.0, frac 0.005, seed 2: at step 100 R2 0.997 (s* 0.383), act_rel_err 0.98; final R2 0.997 -> persists; predicted decays
w 10.0, frac 0.01, seed 0: at step 200 R2 0.997 (s* 0.383), act_rel_err 0.01; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.01, seed 1: at step 200 R2 0.997 (s* 0.383), act_rel_err 0.01; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.01, seed 2: at step 200 R2 0.997 (s* 0.383), act_rel_err 0.01; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.02, seed 0: at step 400 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.02, seed 1: at step 400 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.02, seed 2: at step 400 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.05, seed 0: at step 1000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.05, seed 1: at step 1000 R2 0.997 (s* 0.383), act_rel_err 0.01; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.05, seed 2: at step 1000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.1, seed 0: at step 2000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.1, seed 1: at step 2000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
w 10.0, frac 0.1, seed 2: at step 2000 R2 0.997 (s* 0.383), act_rel_err 0.00; final R2 0.997 -> persists; predicted persists
accuracy: two-condition rule 26/30; R2-only rule 30/30

**VERDICT: PASS**
