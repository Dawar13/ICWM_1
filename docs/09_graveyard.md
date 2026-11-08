# 09. Graveyard

Ideas already killed. Do not resurrect one without new evidence, and if you do, write the new evidence here first.

**Format for new entries:** `| idea | source (us, Gemini, other) | date killed | reason, with the paper IDs that kill it |`

| Idea | Source | Killed | Why it is dead |
|---|---|---|---|
| Gaussian-smoothed log-barrier "Weierstrass homotopy" planning | Gemini round 1, idea 1 | 2 Nov | CEM/MPPI already follow a Gaussian-smoothed objective; randomised smoothing for contact (Suh, Pang, Tedrake) and interior-point MPC exist; the guardrail head is SLS². Its "94.5% to 2.8%" number stitched two protocols together |
| SIGReg-density out-of-distribution barrier | Gemini round 1, idea 2 | 2 Nov | Done in SLS². Also flawed: Gaussian mass in high dimensions sits on a thin shell, so off-manifold predicted points can look typical |
| Min-max "least action" latent planning | Gemini round 1, idea 3 | 2 Nov | Robust or minimax MPC. Nature is not an adversary, so it is over-conservative. The useful core moved to Candidate A as recourse planning |
| Surprise-based subgoal discovery ("temporal renormalisation") | Gemini round 1, idea 4 | 2 Nov | Event segmentation theory, THICK, Director, H-JEPA, Mind the Gap |
| CEM-to-policy "hippocampal consolidation" | Gemini round 1, idea 5 | 2 Nov | Guided policy search, POPLIN, TD-MPC2 prior, LePlanner, GC-IDM, LeFlow |
| Dataset-graph geodesic or Dijkstra cost ("Eikonal" planner) | Gemini round 2, idea 1 | 4 Nov | Search on the Replay Buffer (2019), Beyond Euclidean Proximity 2605.22164, RC-aux, Temporal-Distance JEPA, VQ-WM. Also: if you plan on the dataset graph, the world model does little |
| Event-triggered or conformal-tube replanning | Us, then Gemini round 2 | 4 Nov | AdaReP 2606.23079, CERT-Replan 2610.09302, ARC-Bench 2609.05461 |
| Pure failure-attribution diagnostic paper | Us (2 Nov report) | 4 Nov | Partly done (Beyond Euclidean Proximity, Planning Limits, ARC-Bench, VIScore), and not a method paper |
| Attribution + geodesic cost + conformal tube bundle | Gemini round 2, idea 1 | 4 Nov | A bundle of three graveyard items. Its target numbers (34% to 92%, 18% to 50%) were invented, not measured |
| Scenario tree with a tuned separation bonus β | Gemini round 2, idea 2 | 4 Nov | Branching at a fixed step assumes free identification; the β bonus is an untuned heuristic. The principled version lives on as Candidate A |
| Pullback-metric preconditioned CEM | Gemini round 2, idea 3 | 4 Nov | Essentially full-covariance CEM, CMA-ES or Gauss-Newton shaping; weak delta. The calibration half lives on as Candidate C |

## Lessons from the graveyard (avoid these patterns)

- Renaming a known method with grand words: Weierstrass, renormalisation, hippocampal, Eikonal.
- Stacking components so the paper looks large.
- Writing targets as if they were results.
- Comparing against a baseline number from a different protocol.
- Citing papers or numbers that were never fetched.

## Added 7 to 8 Nov 2026

| Idea | Source | Killed | Why it is dead |
|---|---|---|---|
| "The world model is lying to your planner" (prediction vs decision utility; planner exploitation) | GPT, 7 Nov | 7 Nov | Lambert et al. 2020, Objective Mismatch (2002.04523); MOPO/MOReL; in 2026 JEPA: Cheung and Zhang (NeurIPS 2026 workshop), Agrawal et al. (NeurIPS 2026, LeCun co-author), Singh 2608.12959 |
| Mechanism transfer across physics | GPT, 7 Nov | 7 Nov | Established field (context-conditioned dynamics, meta-learning, sim-to-real); no specific open failure; does not fit compute |
| "World model as scientist" (active experimentation) | GPT, 7 Nov | 7 Nov | Denil et al., ICLR 2017 (1611.01843); Bayesian experimental design; active inference. Only the narrow #2 survives |
| Gradient-based planning in JEPA world models | Us, 7 Nov | 7 Nov | Agrawal, Kalra, Parthasarathy, LeCun, Bounou, Izmailov, Goldblum (NeurIPS 2026); Parallel Stochastic Gradient-Based Planning (ICML 2026); FF-JEPA 2606.09311 |
| "Blind or Deaf" as a new failure mode or diagnostic | Us, 7 Nov | 8 Nov | Deaf mechanism named by ActSWM 2607.26712 and Shi et al. 2608.04653; probes by SLIM 2610.03137, ACPC 2608.12939, AD-WM 2609.30264. The learning-dynamics part survives as candidate #1 |
| Label-free action-use diagnostic predicting planning | Us, 7 Nov | 8 Nov | Same papers as above |
| Latent cost geometry fixes | Several | 7 Nov | The Objective Is the Bottleneck 2608.12959; Temporal-Distance JEPA 2607.25337; Beyond Euclidean Proximity; RC-aux |
