# Code map

Run everything from the kit root (`icwm_gonogo/`) with the virtual environment active.

| Command | What it does | Device |
|---|---|---|
| `python src/g1_linear/sim.py --out runs/test --cfg '{"steps": 2000}'` | One linear run | CPU |
| `python src/run_sweep.py g1_trap` | A registered sweep (see `src/sweeps.py`) | CPU for g1_*, GPU for g2_* |
| `python src/run_sweep.py g2_calib --override '{"steps": 500}'` | Timing run: delete its outputs afterwards | GPU |
| `python src/analysis/gates.py g1_trap` | Pre-registered verdict, appended to `runs/gate_reports.md` | CPU |
| `python src/g2_pixel/train.py --out runs/x --cfg '{...}'` | One pixel run | GPU |
| `python src/g2_pixel/plan_eval.py --run runs/g2/g2_trap/<run>` | CEM-MPC planning check (protocol written to JSON) | GPU |
| `python src/g4_align/align_synthetic.py --out runs/g4/synthetic.json --seeds 10` | G4a | CPU |
| `python src/g4_align/align_tiny_models.py --a <runA> --b <runB>` | G4b (h=1 models) | GPU |

## Metrics logged during training

| Metric | Meaning |
|---|---|
| `agent_r2` | Linear-probe R² of the agent's true state from z_t (simulator ground truth). Primary |
| `env_r2` / `dist_r2` | The same for the autonomous factors or distractors |
| `action_dec` | R² of a_t from (z_t, z_{t+1}) |
| `cas` | Counterfactual action sensitivity: the variance of the prediction across alternative actions, over the variance of the true next latent |
| `act_rel_err` | G1 only: distance of the learned action pathway from the true action effect in the current latent |
| `pred_loss` vs `copy_loss` | Is the predictor better than copying z_t forward? |
| `cov_err`, `z_var` | Is the anti-collapse regulariser working? |
| `rankings` | G1, first log line: each factor's normalised one-step error, passive and with the action. The theory's key input |

## Known differences from LeWM

Our pixel model is a stand-in, not LeWM. Fill `notes/lewm_diff.md` from LeWM's code before drawing pixel conclusions. `sigreg_epps_pulley` is reconstructed from memory of the LeJEPA paper's minimal code, so diff it against LeWM's implementation.
