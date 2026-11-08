# G3. Inference on the public LeWM checkpoints (no training)

No code is shipped here on purpose: we have not seen LeWM's loading API, and guessing it would be fabrication. Claude Code writes `src/g3_lewm_ckpt/probe.py` after reading the real code.

## Step 0: find the code and the data

- Checkpoints on Hugging Face: `quentinll/lewm-pusht`, `quentinll/lewm-tworooms`, `quentinll/lewm-reacher`, `quentinll/lewm-cube`. Read each model card for the code repository and the dataset.
- LeWorldModel paper: arXiv 2603.19312. The training platform is "stable-worldmodel" (find the repository from the paper or the model card).
- Write `notes/lewm_diff.md` with what LeWM actually does, compared with our tiny model (`src/g2_pixel/model.py`):
  - predictor conditioning (AdaLN? zero-init?);
  - initialisation of the action pathway;
  - SIGReg form and weight;
  - gradient into the target or not;
  - projector and BatchNorm;
  - history length;
  - frame skip;
  - the action autocorrelation of the training data.

## Step 1: measure (inference only; about 1000 states per environment)

For each checkpoint, on states from its own dataset:

1. **Action autocorrelation φ of the training data:** the lag-one correlation of logged actions, per action dimension.
2. **Counterfactual action sensitivity (CAS)** with the full history versus a truncated history (the history replaced by copies of the latest frame's latent). The research report's prediction P2: CAS rises when history is cut, wherever φ is high.
3. **Action decodability** from (z_t, z_t+1) with a ridge probe. If the dataset logs proprioception or agent position, also add an agent-state probe.

## Status

**Exploratory.** Pre-register the comparison in `prereg/G3.md` before running. It cannot kill candidate #1 on its own, because LeWM checkpoints are not trained under our controlled conditions. It can strengthen the paper ("the effect is visible in released models").

## Shared-GPU note

Inference only. Batch at most 64 states and check free memory first (`src/common/gpu.py`).
