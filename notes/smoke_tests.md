# Smoke tests run while building the kit (8 Nov 2026, CPU)

**Status:** exploratory. These are not evidence for the paper.
- They used one or two seeds and were not pre-registered.
- Some used truncated budgets.

They are recorded so that the regime choice is disclosed honestly: the G1 default config was chosen *after* these tests. The pre-registered G1 runs (5 seeds, full budget) are the real test.

## G1 linear model (`src/g1_linear/sim.py`)

| Setting | What happened | Reading |
|---|---|---|
| Large persistent agent (g_a 0.3, rho_a 0.97), k 4 | Agent learned immediately, with or without inverse dynamics | Nothing to explain |
| Small persistent agent, obs noise 0.1, k 4 | Agent evicted, even with inverse dynamics (weight 1) | Information or capacity limit, not a trap |
| Small persistent agent, k 6 (spare capacity) | Agent learned (R² about 0.8, limited by noise); inverse dynamics irrelevant | Whitening fills spare dimensions with whatever signal exists |
| Action-set agent (rho_a 0), low noise, k 5 (spare capacity) | Agent learned by step 2000, regardless of inverse dynamics | Same: no competition, no failure |
| **Action-set agent, low noise, k 4 (competition)** | **Plain training: agent R² 0.00, prediction loss 0.34 (2 seeds). Inverse dynamics always on (weight 10): R² 1.00, loss 0.17. Inverse dynamics for the first 10% of 8000 steps only: R² 1.00 at the end, loss 0.175 (2 seeds)** | **Looks like a trap: gradient descent settles at a worse value of the same loss, and a one-off push escapes it for good** |
| Same, inverse dynamics weight 1 | Always on: R² 0.10; warm-up: decays to 0 | The fix must be strong enough to establish the agent |
| Same, action pathway depth 1 instead of 2 | Still evicted | The trap does not need a deep action pathway; it comes from encoder-predictor coupling under capacity competition |
| Same, init of action pathway 0.1 or 0.3 | Still evicted | Init scale alone did not escape (to be confirmed in g1_init) |
| Same, EMA target, or SIGReg (Epps-Pulley) instead of the covariance penalty | Still evicted | Not specific to one anti-collapse choice (to be confirmed in g1_reg) |
| Unstable agent, closed-loop data, warm-up | Partial (R² 0.45) | Different regime; see g1_unstable |

**Pipeline test** (300 steps, truncated budget, deleted afterwards):
- The gates script ran end to end.
- A 30-step warm-up did *not* fully persist (warm-up final R² 0.23 to 0.45), whereas the 800-step warm-up above did. **So the dose matters.** The g1_dose sweep measures it.
- The learning-order rule "plain training keeps the agent if and only if its *passive* (action-free) predictability ranks within the top k" was correct at 6 of 6 rho levels. This was at a truncated budget and is not evidence.

## Early theoretical reading (hypothesis, not a result)

In the linear model:
- Early training ranks factors by **passive** predictability.
- The predictor's action pathway can only grow along latent directions that already carry the agent.
- So an agent whose motion is mostly set by actions looks like noise and loses the capacity competition.
- It loses even though, once the action is used, it would be the *most* predictable factor.
- This is a coupling trap. Escaping it once (inverse-dynamics warm-up) lands in a better basin that the loss itself then holds.

It refines the research report's P1:
- What matters may be the agent's passive predictability relative to its competitors, under limited capacity, rather than "response energy" alone.
- A persistent integrator agent (position under velocity control) was not trapped in the linear model.
- So the pixel failures in SLIM may need a different explanation. One candidate: a small sprite's *pixel pattern* decorrelates quickly when it moves, even though its position is persistent. G2 tests this.

## G2 pixel model (`src/g2_pixel/`)

- **Collapse at the start.** With the default CNN init, latents started near-constant, and a per-sample-mean Epps-Pulley regulariser could not lift them.
- **Fixes applied:**
  - a batch-normalised projector output (LeWM has a projector with batch normalisation; check this);
  - the LeJEPA-style regulariser (trapezoid integral times batch size).
- **After the fixes:** with sigreg_w 1.0 latents were non-collapsed (z_var about d), and the agent probe rose to R² 0.58 within 300 CPU steps. With sigreg_w 0.1 it had not risen by step 300.
- **SIGReg weight:** calibrate it with the pre-registered whitening rule (g2_calib).

## G4 synthetic alignment (`src/g4_align/align_synthetic.py`), seed 0 only

**Controllable subspace (cosine to the true map):**

| Method | Cosine |
|---|---|
| Action-response Procrustes | 1.00 (exact predictors); 0.95 (20% noise) |
| Gromov-Wasserstein | about 0.1 (isotropic) and -0.06 (anisotropic) |
| Plain OT | -0.08 to -0.23 |

**Uncontrollable complement:** action-response Procrustes gave -0.2 to 0.2 (chance), as the theory predicts.

**Paired oracle:** needed about 8 pairs to match action-response Procrustes in this 8-dimensional world.
