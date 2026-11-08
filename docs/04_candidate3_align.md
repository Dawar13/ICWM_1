# 04. Candidate #3: "What You Can Control, You Can Align" (fallback)

## Plain language

- **The problem:**
  - Two world models trained separately store the same world in different coordinates.
  - With SIGReg, each model's coordinates are fixed only up to a rotation (Klindt, LeCun, Balestriero 2605.26379), and the stored data look the same from every angle.
  - So any method that only looks at the shape of the data (optimal transport, Gromov-Wasserstein) cannot tell which rotation is right.
- **The way out:** both models must agree on what each action does. Match their predicted responses to the same actions, and the rotation is recovered on everything the actions can reach, and only there.

## Method

1. **Responses:** for each model separately, on its own states, compute the mean k-step response to each unit action. This gives the empirical controllability matrix (Krylov matrix).
2. **Map:** orthogonal Procrustes between the two response matrices gives the map. Also report a general linear map, since real models need not be exact rotations.
3. **Corollary to test:** error near zero on the controllable subspace, chance on its complement.

## Prior art

| Work | Status | Relation |
|---|---|---|
| Dynamics Cycle-Consistency (Zhang, Xiao, Efros, Pinto, Wang), ICLR 2021 oral, 2012.09811 | [S] | Unpaired alignment through dynamics, with no identifiability statement |
| Social-JEPA, 2603.02263 | [V] abstract | Fits the map on shared states (paired). arXiv shows v1 with no withdrawal notice (checked 7 Nov) |
| Platonic Representation Hypothesis on World Models, 2608.23720 | [S] | Stitches DINO-WM models; no actions as the signal |
| Xu and Kanan, More Data Cannot Break a Symmetry, 2608.27651 | [S] | Gromov-Wasserstein is flat over the symmetry orbit. Cite it; do not claim that half |
| HGA, 2608.28840 | [U] | Verify before citing |

## Gates

- **G4a synthetic** (`src/g4_align/align_synthetic.py`), PASS if all hold:
  - ours (controllable) ≥ 0.9 at predictor noise ≤ 0.05;
  - |ours (uncontrollable)| ≤ 0.3;
  - Gromov-Wasserstein (controllable) ≤ 0.3 in the isotropic case.
- **G4b tiny models** (`src/g4_align/align_tiny_models.py`) on two G2 checkpoints with different seeds and h = 1:
  - PASS if agent-probe transfer ≥ 0.7 × within-model R²;
  - KILL if below 0.5×.
  - Distractor transfer should be much weaker (the prediction). Report it either way.
- Pre-registration: `prereg/G4.md`.

## Risks

- "Obvious once stated."
- LeWM latents may not be isotropic enough for Gromov-Wasserstein to fail. Measure isotropy (eigenvalue spread of the latent covariance) on the real checkpoints before claiming it.

**Odds (judgement, from the research report):** acceptance about 40 to 55%, spotlight about 10 to 20%.
