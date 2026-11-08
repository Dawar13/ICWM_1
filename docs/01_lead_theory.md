# 01. Lead candidate: "Listening Is Learned Last"

## Plain language (read this first)

A JEPA world model has two parts that must learn together:
- an **encoder** that decides what to keep from the image;
- a **predictor** that guesses the next latent from the current one and the action.

**The trap:**
- Early in training, the action pathway is weak. So the encoder judges each feature only by how predictable it is *on its own*, without the action.
- The agent's motion is mostly caused by actions, so at that point it looks like noise. The encoder spends its limited room on things that move predictably by themselves (balls, backgrounds).
- The predictor's action pathway can only learn along directions the encoder already keeps. So it never gets the chance to show that the agent is perfectly predictable once you know the action.
- Each part waits for the other. Training settles in a solution that ignores the agent, even though keeping the agent would give a lower loss.

**The way out:** a brief inverse-dynamics loss (predict the action from consecutive latents) forces the encoder to keep the agent for a while. Once the predictor has learned to use the action, the main loss itself prefers to keep the agent, so the push can be switched off.

**Where it should fail:** if the loss really does prefer other features, even with the action (for example, a noisy agent, or too little room), the warm-up will decay. Then the fix must stay on. That boundary separates "trap" from "objective" and reconciles our claim with Toso, LeCun et al.

## Formal setup (linear; `src/g1_linear/sim.py` implements exactly this)

**World:**
- Agent state s^a, with s^a_{t+1} = ρ_a s^a_t + b a_t + σ_a ε.
- Environment factors s^e_i: autonomous AR(1) processes with persistence ρ_e,i and unit variance.
- Actions are AR(1) with lag-one correlation φ.
- Observation x = G_a s^a + G_e s^e + noise, with orthonormal mixing columns scaled by the gains g_a and g_e.

**Model:**
- Encoder z = W x, with k latent dimensions.
- Predictor ẑ_{t+1} = Σ_j A_j z_{t-j} + V a_t, where V is either one matrix or a two-layer product.
- Anti-collapse: covariance-to-identity penalty (the Gaussian case of SIGReg), the Epps-Pulley SIGReg form, or an EMA target.
- Optional inverse-dynamics head â_t = U [z_t; z_{t+1}].

**Key quantity:** the normalised one-step error of each factor, with and without the action (logged as `rankings` at the start of every run).
- **Passive ranking:** the order the encoder "sees" early.
- **Action-conditioned ranking:** the order the loss prefers at its optimum, once V is learned.

## Predictions

Derive these properly in the linear model before running. Write each derivation in `notes/theory.md`, and have a fresh subagent check it. The statements below are hypotheses.

| ID | Prediction | Sweep | Notes |
|---|---|---|---|
| **P1 (learning order)** | Plain training keeps the agent if and only if its *passive* predictability ranks within the top k. Action-explained predictability does not count early | `g1_rho` | Smoke: 6/6 at a truncated budget. The research report's version said escape time depends on "action-response energy"; our smoke tests suggest passive ranking is the better variable. Test both |
| **P2 (deaf predictor)** | Where past latents already predict the current action's effect (high φ), the action pathway's alignment with the true effect is learned more slowly | `g1_hist`, `g2_hist` | For an action-set agent, z_t already carries a_{t-1}, so this applies at h = 1. For an integrator agent it needs h ≥ 2. The research report's form: escape slows by about 1/(1-φ²). Check the functional form; do not assume it |
| **P3 (persistence boundary: the core claim)** | A warm-up that establishes the agent persists after it is switched off **if and only if** the agent is within the top k by *action-conditioned* predictability | `g1_trap` (inside the boundary), `g1_objective` (agent noise pushes it across), `g1_unstable` (Toso regime) | Otherwise it decays to the plain-training solution. Persistence is judged at the full budget, since decay outside the boundary may be slow; truncated pipeline tests could not show it. If the warm-up also persists *outside* the boundary, both solutions are stable minima (multistability). Then restate P3 as "persists whenever the agent-kept fixed point is stable", and derive that stability condition in `notes/theory.md` |
| **P4 (dose)** | There is a minimum warm-up dose (steps × weight) needed to escape | `g1_dose` | Smoke: 30 steps was not enough, 800 was |
| **P5 (remedy specificity)** | History dropout restores action use where P2 bites. Action-pathway init scale does little (smoke) | `g1_init`, `g2_hist` | Report honestly if the research report's init prediction fails |
| **P6 (pixels; SLIM reconciliation)** | In pixels, what matters is how predictable the agent's *pixel features* are, and how much of the frame responds, not object size as such | `g2_response` | Our hypothesis: a small sprite's pixel pattern decorrelates when it moves, so it looks passively unpredictable even when its position is persistent |

## Why this is new (and what is not)

| Work | Covers | Leaves open (our contribution) |
|---|---|---|
| SLIM 2610.03137 | Failure when little of the frame moves; inverse-dynamics fix; repair goes through the encoder; "the SIGReg weight does not cause the collapse" (appendix H) | No theory; fix kept on; why response fraction matters and size does not |
| Toso, LeCun et al. 2610.07540 | Zero-loss encoders can drop controllable unstable modes (objective level) | Which solution gradient descent reaches; trap vs objective boundary |
| ActSWM 2607.26712; Shi et al. 2608.04653 | Name the history or statistical shortcut | No dynamics law; no link to φ |
| AVL-JEPA 2610.03587 | "Causal dynamics information collapse", static Bayes-risk argument | Static, not dynamic |
| Khetarpal et al. 2406.02035 (AISTATS 2025) | Fixed points of action-conditional BYOL | No learning order; no controllable vs uncontrollable; no SIGReg; no warm-up |
| Simon et al. 2303.15438; Littwin et al. 2407.03475 | Stepwise learning order in passive SSL/JEPA | No actions |
| CAER 2608.30897 | Video world models: background dominates the gradient; reweighting | Generative pixels, not what a JEPA representation keeps |

**Framing for the paper:** "a theory of when and why, with a falsifiable warm-up prediction and a trap-versus-objective boundary". Not "a new failure", not "a new diagnostic", not "a new fix".

## Risks to watch

- **Linear theory may not transfer to the nonlinear pixel model.** G2 is the test. Predictions are ordinal, so test them as orderings.
- **SIGReg may keep redistributing variance after the warm-up,** so the agent decays. That outcome is a KILL in G2: report it.
- **A concurrent learning-dynamics paper may appear.** Search arXiv and OpenReview (ICLR 2027 submissions) before writing.
- **Calibration confound.** We chose the G1 regime after smoke tests. The defence is that P1 and P3 are *predicted* boundaries, tested across regimes, not one cherry-picked setting.
