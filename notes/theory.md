# Theory notes (linear model)

Written 8 Nov 2026, before any G1 sweep was run. Code: `src/analysis/theory_linear.py` (closed forms, numeric checks, predictions for every sweep cell). Checked by a fresh subagent: see the verdict at the end.

## Plain language

If the predictor is always as good as it can be for the current encoder, and the latent is whitened, the JEPA loss only cares about *which* factors the encoder keeps. Keeping a factor costs its one-step error once the action is known. So the best encoder keeps the k most predictable factors, counting the action.

Gradient descent can still get stuck. Moving a little of the agent into a slot that holds an environment factor blends two signals that move differently (the agent jumps with each action; a ball drifts smoothly). A single linear predictor cannot follow both, so the blend costs extra error. If that blending cost is larger than what the agent would save, the "agent dropped" solution is a genuine local minimum: a trap. The same blending cost also protects the "agent kept" solution, so once a warm-up has put the agent in, it can stay even where the loss would prefer it out. The true boundary between "trap" and "objective" is therefore a stability condition, not "is the agent in the top k".

A second point explains why the agent is learned last. When the action pathway starts near zero and is factorised (V = V2 V1), it cannot grow until the encoder already carries the agent. Until then, the agent is judged by its passive (action-free) predictability. This is P1.

## Setup and assumptions

The model is `src/g1_linear/sim.py`:
- factors f = (agent, env_1..env_m);
- observation x = G f + small noise;
- encoder z = W x with k latents;
- predictor ẑ_{t+1} = A z_t + V a_t;
- loss: squared prediction error plus a covariance penalty;
- the target also receives gradient.

**Assumptions:**
- **A1.** Integrator agent: s_{t+1} = ρ_a s_t + b a_t + σ ε, with i.i.d. unit-variance actions (φ = 0).
- **A2.** History h = 1.
- **A3.** Hard whitening: Cov(z) = I. The real penalty is soft; see the correction below.
- **A4.** "Fast predictor": (A, V) are at their least-squares optimum for the current encoder.
- **A5.** Observation noise is negligible (0.01 in the default config).

**Normalised factors.**
- Each factor is scaled to unit variance.
- Factor i then follows f̃_i(t+1) = d_i f̃_i(t) + β_i a_t + noise, with d_i² + β_i² + nerr_i = 1.
- For an environment factor: d = ρ_e, β = 0, nerr = 1 - ρ_e².
- For the agent: d = ρ_a, β² = b²(1 - ρ_a²)/(b² + σ²) and nerr_a = σ²(1 - ρ_a²)/(b² + σ²). Here nerr_a is the action-conditioned normalised error. The passive error is 1 - ρ_a².

## Result 1: reduced loss

Under A1 to A5, z = P f̃, where P is k × n with orthonormal rows. Let Π = PᵀP, D = diag(d) and b = (β_i).
- The best predictor is A = P D Pᵀ and V = P b, because a_t is independent of f̃(t).
- The loss is

  L(P) = k − tr(Π D Π D) − bᵀ Π b.

- At a coordinate subset K (the encoder reads whole factors): **L(K) = Σ_{i∈K} nerr_i**.
- The global minimum keeps the k factors with the smallest action-conditioned error.

**Default regime** (ρ_a 0, σ 0, env ρ = 0.99, 0.97, 0.95, 0.9, k 4):

| Solution | Hard whitening | With the soft-penalty correction |
|---|---|---|
| Agent dropped | 0.3665 | 0.342 |
| Agent kept | 0.1765 | 0.170 |

These match the smoke tests (0.34 and 0.17) without fitting.

**Soft-penalty correction.**
- With penalty λ(var − 1)² per direction, the variance settles at 1 − nerr/(2λ).
- The logged prediction loss is then Σ(nerr − nerr²/(2λ)).

## Result 2: which subsets are traps

The pair coefficient for keeping X and dropping Y is

  **c(X, Y) = (d_X − d_Y)² + nerr_Y − nerr_X.**

- **Hessian.** At a coordinate subset K, the Hessian of L on the Grassmannian is diagonal in the pair rotations (X ∈ K, Y ∉ K), with entries 2 c(X, Y).
- **Exact profile.** Along a single rotation with s = sin²θ, L(s) − L(0) = c s − (d_X − d_Y)² s².
- **Interpretation:**
  - nerr_Y − nerr_X is the gain or loss from swapping X for Y;
  - (d_X − d_Y)² is the cost of blending two factors with different dynamics into one slot.
- **K is a strict local minimum iff c(X, Y) > 0 for every pair.** It is a trap if, in addition, it is not the global minimum.

**Corollaries:**
1. **No traps among passive factors.** If X and Y are both environment factors with 0 < ρ_X < ρ_Y, then c(X, Y) = −2ρ_X(ρ_Y − ρ_X) < 0. Environment competition always sorts itself out.
2. **Agent dropped is stable** iff, for every kept environment factor j, (ρ_j − ρ_a)² > nerr_j − nerr_a.
   - Default: c = 2ρ_j² − 1 > 0 for all kept j (all ρ_j ≥ 0.9). So it is a genuine local minimum, a **trap** (region A).
3. **Agent kept is stable** iff, for the dropped factor Y, (ρ_a − ρ_Y)² > nerr_a − nerr_Y.
   - For ρ_a = 0 this is 1 − nerr_a > 0, which **always holds**.
   - So a warm-up should persist for an action-set agent even when its action-conditioned error is worse than the dropped environment factor's (region C, multistable).
4. **Restated P3 (persistence boundary).** A warm-up persists iff c_kept = min over dropped Y of [(ρ_a − ρ_Y)² + nerr_Y − nerr_a] > 0.
   - The original P3 boundary ("agent in the action-conditioned top k") is right only when ρ_a ≈ ρ_Y.
   - Four regions:

| Region | Agent globally better? | Agent-dropped stable? | Agent-kept stable? | Plain training | Warm-up |
|---|---|---|---|---|---|
| A trap | yes | yes | yes | fails | persists |
| B no trap | yes | no | yes | escapes (if V is fast) | persists |
| C multistable | no | yes | yes | fails | persists (locks in a worse loss) |
| D objective | no | yes | no | fails | decays (Toso et al. territory) |

Numeric checks (`theory_linear.py check`):
- **Hessian:** finite-difference Hessian against 2c on 5 configurations, max error 1.8e-7.
- **Loss formula:** Monte Carlo least-squares residual from the real simulator against Σ nerr, within 0.001 in all 5.
- **1-D profile:** exact to 3e-16.

## Result 3: slow action pathway (P1)

**Setting.** The default V = V2 V1 with init std 1e-3 makes the action pathway a degenerate saddle at V = 0:
- the gradient on V is proportional to how much agent the encoder already carries;
- the gradient that would bring the agent into the encoder is proportional to V.

**Claim.**
- While V ≈ 0, the agent's error term is its passive error 1 − ρ_a², and β² drops out.
- Then c_V0(j, agent) = 2ρ_j(ρ_j − ρ_a).
- So the agent can enter only if ρ_a > ρ_j for some kept j, which is **exactly the passive ranking (P1)**.
- With a fast pathway (depth 1, V tracks its optimum within about 10 steps at lr 0.05), the full c applies instead. Region-B cells then escape the saddle.

**Escape rate.**
- Near a saddle, the agent direction in W grows at about 2 · lr · g̃_a² · |c| per step, with g̃_a = g_a · std(agent).
- This is slow because g_a = 0.2.
- Cells with |c| < 0.05 get no prediction (pre-registered margin).

**Prediction (g1_depth, plain training, ρ_a = 0.75):**
- depth 2 (slow V) keeps the agent out;
- depth 1 (fast V) brings it in.

This separates "listening is learned last because the action pathway lags" from any rule based on passive predictability alone.

## Result 4: dose (P4), approximate

**Barrier.** After a warm-up that has established V, the 1-D profile has its barrier at s* = c_ev/(2(ρ_j − ρ_a)²). For the default regime, s* = 0.383 for the env-0.9 slot.

**Prediction.** A warm-up persists iff, when it ends:
- the agent's R² exceeds s*; and
- the action pathway is aligned (act_rel_err < 0.5).

**Caveats.**
- This is approximate: the separatrix in the full space is not exactly the 1-D one.
- A short warm-up may end before V has grown, and then the agent decays even above s*.

## Result 5: action autocorrelation (P2), outside A1, qualitative

**Plain training.**
- With φ > 0 and ρ_a = 0, the agent is passively AR(1) with coefficient φ.
- With a slow V, the passive rule predicts the agent is evicted for φ < 0.9, marginal at 0.9, and kept at 0.95.
- So in plain runs, the "action-use escape step" is censored at low φ and finite at high φ. That is a *negative* trend: the pre-registered g1_hist criterion (Spearman ≥ 0.8) is predicted to **fail**. The cause is the trap, not evidence against P2.

**IDM-always runs.**
- With IDM always on, the agent is encoded at every φ.
- V must separate a_t from the correlated history; gradient descent on correlated regressors slows like 1/(1 − φ).
- Prediction: escape step increases with φ (secondary criterion g1_hist_always).

## Result 6: other sweeps (no sharp prediction)

- **g1_init:** the default regime is in region A for both fast and slow pathways. Prediction: plain fails at every init scale and depth.
- **g1_reg:**
  - Epps-Pulley SIGReg on Gaussian latents acts like whitening, so we expect the same as `cov`.
  - The EMA target (stop-gradient) changes the dynamics; Khetarpal et al. 2406.02035 is the relevant fixed-point analysis.
  - Descriptive only.
- **g1_unstable:** the action depends on the state (closed loop), which violates A1. Descriptive only.

## Limits of this theory

- It is a reduced landscape (A3, A4). A strict local minimum of the reduced loss is also a local minimum of the full loss (the full loss is at least the reduced loss, with equality at the optimal predictor), but escape *times* depend on the real parametrisation.
- The soft penalty, SGD noise and finite budgets blur the boundary near c = 0.
- Only coordinate subsets are analysed. Non-coordinate stationary points can exist when factors are degenerate (equal d and β); none of our cells are degenerate except ρ_a = 0.9 against env 0.9, which is flagged as marginal.

## Subagent check

A fresh subagent saw only the setup and claims C1 to C6, with no notes or docs, and derived each one. Its scripts are in the session scratchpad (`checker/`). Verdict, 8 Nov 2026:

| Claim | Verdict | Condition or correction |
|---|---|---|
| C1 reduced loss | VERIFIED | Needs exactly one action-driven factor; otherwise ββᵀ has off-diagonal terms. Monte Carlo matches to about 1e-3 |
| C2 Hessian and pair coefficient | VERIFIED | Diagonality and criticality need the single-agent condition. The 1-D profile is exact |
| C3 corollaries | VERIFIED | (c) needs the kept environment factors to have strictly larger ρ than the dropped ones; ties give c = 0 |
| C4 soft penalty | VERIFIED | Gives 0.3418 and 0.1698, confirmed by free optimisation. Curvatures are multiplied by (1 − nerr_X/2), so their signs are unchanged |
| C5 slow pathway | **CONDITIONAL** | See below |
| C6 barrier | VERIFIED | s* = 0.3827, barrier height 0.1186. A grid search over paths found the lowest barrier at 0.1184 |

**C5 correction (Result 3 restated).**
- The formula c_V0 = 2ρ_j(ρ_j − ρ_a) is right.
- But "the agent can enter only if ρ_a > ρ_j" is a statement about **speed**, not about the landscape:
  - if the full c(j, agent) < 0 while c_V0 > 0, a cubic coupling makes the point a saddle;
  - the agent then enters after a plateau that grows roughly like 1/v0², where v0 is the init scale of V1 and V2.
- Referee's simulation: plateau ≈ 11k steps at v0 = 3e-2, 53k at 1e-2, about 550k at 3e-3.
- At our act_init = 1e-3 the plateau is likely millions of steps, far beyond the 20k budget. So the registered within-budget predictions (g1_rho, g1_depth, g1_phase "plain fails") are unchanged.
- If c(j, agent) > 0 (region A), the point is a local minimum whatever the pathway's parametrisation.
- "Listening is learned last" should therefore be stated as: **the plateau before action use scales like 1/v0² when the agent is not passively competitive.** This is a falsifiable scaling law that could be tested later.

**Hidden assumptions flagged by the referee:**
1. **The target receives gradient.** With a stop-gradient target (EMA), the linearised encoder dynamics become 2d_j(d_j − d_i), with no β. The action plays no role, and an action-set agent (d = 0) is never preferred. LeWM trains end to end (`notes/lewm_diff.md`), so the theory applies to LeWM. It also yields a new prediction for the EMA arm of g1_reg, registered as amendment 2 in prereg/G1.md before g1_reg ran.
2. **One action-driven factor, with i.i.d. actions independent of the state.** φ ≠ 0 and unstable_cl break C1 and C2. These cases were already flagged as outside the theory.
3. **(A, V) fast relative to W.**
4. **g_a = 0.2** does not change the landscape, but it makes the agent direction about 25 times slower. Observation noise raises the agent's nerr to about 0.0025.
5. **Non-coordinate stationary points** exist (the C6 saddles). No non-coordinate local minima were found in random restarts. This is proved for k = n − 1 and only numerical otherwise.
