# ICWM 2027: three ranked paper candidates, each checked against existing work

**Copy of the extended research report, 8 Nov 2026. AI-generated: verify before use. See README.md.**

## Bottom line

Lead with a reworked version of A. Between August and October 2026 other groups published the "blind or deaf" diagnosis, the action-sensitivity probe and inverse-dynamics fixes, so a diagnosis paper is no longer new. What is still open is a learning-dynamics account of when an action-conditioned JEPA world model learns to use its actions at all. Its sharpest claim can be tested: if the failure is an optimisation trap rather than a flaw in the objective, a short inverse-dynamics warm-up should be enough.

## TL;DR

- **#1 (lead): "Listening Is Learned Last".** A saddle-escape theory of action use in JEPA world models. It predicts from two measurable quantities when the encoder will encode the controlled agent and when the predictor will use its action input:
  - the fraction of the frame that responds to actions;
  - how strongly actions are correlated over time.

  It also predicts that a brief inverse-dynamics warm-up, switched off afterwards, is enough for stable agents. I found no learning-dynamics paper on this question. The diagnostic and the "deaf predictor" observation are already published.
- **#2: "Look Before You Leap".** Observability-gated contingency planning in a latent world model, aimed at LeCun's problem of planning when latent variables stand for what cannot be observed. I found no JEPA paper in which information-gathering actions emerge without an exploration bonus. The closest work, EpicWorldModel (NeurIPS 2026), uses an explicit variance bonus. The idea is real but conceptually close to classical dual and contingency control, and it is harder to run in 23 days.
- **#3: "What You Can Control, You Can Align".** Under SIGReg, label-free alignment of two independently trained world models can only be identified up to a rotation. Shared actions recover that rotation on the controllable subspace, and only there. The result is clean and cheap (mostly inference), but the area is crowded: Dynamics Cycle-Consistency, Social-JEPA and identifiability work all sit nearby. Treat it as a strong fallback, not a lead.
- **Overall judgement:** none of the three gives a credible above-50% chance of a spotlight. #1 is the only one where a spotlight is plausible, if the kill test passes cleanly.

## Key findings

### About the venue (verified only in part)

- The ICWM "About" page lists the creators as Randall Balestriero (Brown / AMI), Amir Zadeh (Lambda) and Yann LeCun (NYU / AMI), with the conference in "Late May 2027, San Francisco Bay Area".
- I could not fetch the Call for Papers page. The public icwm.cc pages list only "Late May 2027, San Francisco Bay Area", and aideadlines.org lists the ICWM 2027 dates as "TBA".
- So the 30 November deadline, the review dates and the missing tier and page-limit details come from your own check and are not publicly confirmed by a second source.
- **Kit note:** we fetched the CFP page ourselves on 7 Nov; see CONTEXT.md.

### The "action-blind world model" area filled up between August and October 2026

Each of the following is verified on arXiv:

- **Keeping JEPA World Models Plannable When Little of the Frame Moves** (Strohm, Wagner, Schwab, Huber; 2610.03137, 2 Oct 2026):
  - LeWM "succeeds on 3 of 1200 SLIM trials";
  - their probes locate the failure in the encoder, and "rollouts are no better than copying the current latent forward";
  - an inverse-dynamics loss raises success "from 0.003 to 0.348 ± 0.016", and "Controls attribute the repair to the gradient into the encoder";
  - "A cheap action-sensitivity probe, computable without environment access, acts as an empirical necessary condition";
  - "doubling the object radius changes nothing, whereas making more of the frame respond to actions lifts the vanilla model from 0.005 to 0.530";
  - its appendix H is titled "The SIGReg weight does not cause the collapse".
- **ActSWM (2607.26712)** already names the deaf mechanism: "as the latent context becomes more informative, the predictor can extrapolate state progression from visual history while becoming insensitive to the conditioning action sequence."
- **Overcoming Statistical Bias in Action-Controllable World Models** (Shi et al., 2608.04653) names the shortcut: "different actions may produce similar futures, while motion may persist even under zero action."
- **Action-discriminative and action-consistency fixes are now numerous:**
  - Delta-JEPA (2606.31232);
  - AD-WM (2609.30264): "recovery discourages the predictor from ignoring action information";
  - CF-JEPA (2610.00727): "the baseline latents lose the agent";
  - Action-Consequence Alignment (2610.04539): "nearby alternatives can receive lower prediction errors despite producing physical outcomes farther from the recorded target";
  - No Gaussian Required (2608.17542);
  - Toward Physically Grounded JEPA World Models (2609.03565);
  - AVL-JEPA (2610.03587, under review at ICLR 2027): names "causal dynamics information collapse" and backs it with a static Bayes-risk argument.
- **Theory papers cover the objective but not the training dynamics:**
  - Toso, LeCun et al. (2610.07540): "next step prediction combined with anti-collapse regularization does not guarantee that controllable unstable modes are preserved: the training loss can be minimized while these modes are collapsed";
  - I Act Therefore I Am (2609.31161), on identifiability;
  - Zhang et al. (2607.22430): "The gap is not whether actions can be inserted into latent predictors, but when action-conditioned prediction makes the action effect itself identifiable."
- **The open hole:**
  - A search of arXiv for July to November 2026 found no paper analysing the learning dynamics of action use in action-conditioned JEPA world models.
  - It also found no follow-up extending Littwin et al. (2407.03475) or Khetarpal et al. (2406.02035, AISTATS 2025) to the controllable-versus-uncontrollable question.
  - The Intervention Gap paper (Vakalis, 2608.29998) found that LeWM checkpoints on Cheetah "capture the current task query" while their "imagined five-step effects are worse than predicting no effect". It explicitly leaves open "which mechanism produces rotation with excess gain in one family and attenuation in another".

### Evidence that one-step loss is the wrong guide (supports all three candidates)

- Cheung and He (2610.19375, NeurIPS 2026 WM-PAI workshop) report "r = -0.010" between one-step error and planning success across fifteen checkpoints.
- You et al. (2607.10362) find "the single-step validation error is essentially uncorrelated with control success".
- An independent reproduction of LeWM on TwoRoom (2608.10145) found that "the two most accurate models plan worse than a random-action control at the longer horizon".

### Planning under hidden variables in JEPA world models is open but has started

- EpicWorldModel (2610.05996, accepted at NeurIPS 2026) trains a stochastic flow-matching JEPA and scores plans as "mean goal MSE-βU".
- "Disabling the exploration term (β=0) costs 5–7 points of success".
- Its stochastic predictor alone reaches "71.7% versus 45.0%" for LeWM.
- No contingency, dual-control or emergent-probing planner for JEPA world models was found.

## Candidate 1 (lead): "Listening Is Learned Last: a saddle-escape theory of action use in JEPA world models"

**1. Claim.** A JEPA world model learns to use its actions only after it escapes a training trap. The trap's depth can be predicted from how much of the scene the agent moves and how predictable the agent's own past motion is. A one-off push out of the trap (an inverse-dynamics warm-up, then switched off) should be enough wherever the controlled dynamics are stable.

**2. Evidence that it is open.**
- The failure is documented and repeatable: LeWM gets 3 of 1200 on SLIM; "the baseline latents lose the agent" (CF-JEPA); the predictor becomes "insensitive to the conditioning action sequence" (ActSWM).
- Every existing fix keeps an extra loss on for the whole of training (SMWM 2606.20104, Delta-JEPA, AD-WM, EP-IDM 2610.07540, SLIM's IDM loss). None asks whether the bad solution is a trap in training or the true optimum of the loss.
- The objective-level theory shows that collapsed solutions can reach zero loss (2610.07540). It does not say which solution gradient descent reaches, or when.
- SLIM observes that object size does not matter and frame response does, and that SIGReg weight is not the cause. Nobody has explained this mechanistically.
- Vakalis (2608.29998) explicitly leaves the mechanism open.
- Simon et al. (2303.15438) and Littwin et al. (2407.03475) cover learning order in passive SSL, with no actions.
- Khetarpal et al. (2406.02035, AISTATS 2025) analyse BYOL-AC with an ODE framework. They characterise convergence and the distinctions between the BYOL-Π and BYOL-AC limiting solutions, but say nothing about learning order or controllable versus uncontrollable features.

**3. Candidate causes, and why it is unsolved.**
- **The coupling trap (blind).** The predictor's action pathway receives gradient only in proportion to how much agent state the encoder already carries. The encoder in turn gains from encoding the agent only in proportion to how much the predictor already uses actions. Both start near zero, so the origin is a saddle. The escape rate scales with the action-induced change in the input. That is exactly the "response fraction" SLIM found to matter, and it explains why doubling the object radius does nothing.
- **The history shortcut (deaf).** With frame history, the predictor can explain the agent's motion by extrapolating velocity. The part left for the action input to explain shrinks to roughly (1 - φ²) of its variance, where φ is the lag-one autocorrelation of the actions. Strongly correlated behaviour data (typical of demonstrations) should therefore slow or prevent escape. This is the world-model analogue of the copycat problem (Wen et al., 2010.14876). It also relates to the "stand-in" effect of autocorrelated designs (2611.01666), which concerns diagnostics, not training.
- **Competition under SIGReg.** SIGReg fills every latent dimension with something. Fast, highly predictable uncontrolled content (backgrounds, distractors) wins those dimensions first. This is consistent with Spectral-Target (Zhu et al., 2609.04264), whose abstract names "physical representation laziness": latents that "do not collapse but nonetheless fail to represent key physical properties". The "control ball" phrase is not in its abstract, so check the body before quoting it.
- **Why it is unsolved.** Every paper treats this as a question about the objective and fixes it by changing the objective, so whether training dynamics or the objective is to blame has never been separated experimentally.

**4. Method.**
- **Theory.** Use a linear action-conditioned JEPA: agent state (driven by actions), environment state (autonomous) and noise in the observation; a linear encoder; a predictor with a state matrix, an action matrix and optionally a history term; a whitening constraint in place of SIGReg. From the gradient flow, derive:
  - **P1 (escape time):** time to learn agent directions scales as log(1/initialisation scale) divided by the action-response energy, not the agent's pixel area;
  - **P2 (history penalty):** with history length ≥ 2, escape time grows by roughly 1/(1 - φ²);
  - **P3 (persistence):** once escaped, a stable or marginally stable agent stays in a stable minimum; an unstable mode (Toso et al.'s CartPole regime) does not.
- **Remedies:** an inverse-dynamics warm-up for the first few per cent of steps; the initialisation scale of the action pathway; history dropout or action-decorrelated minibatches. All are tested against IDM-throughout as the upper baseline.
- **A scoped label-free monitor:** action decodability and counterfactual action sensitivity, used only to time the transition. It is not claimed as new.
- **Why it should work:** the coupling-trap structure is standard in two-layer linear models with small initialisation. The new part is that the action input splits the problem into a jointly learned part (controllable) and an alone-learned part (autonomous), with different escape laws.

**Kit note:** our smoke tests suggest the passive-ranking version of P1 may fit better than "response energy"; see `notes/smoke_tests.md`.

**5. Delta over the closest work.**

| Work | What it covers | What it leaves open |
|---|---|---|
| SLIM (2610.03137) | Failure when little of the frame moves; IDM fix; probe | No theory; IDM kept on throughout; why response matters and area does not |
| Toso, LeCun et al. (2610.07540) | Zero-loss encoders can drop controllable unstable modes; endpoint IDM; linear proofs | Which solution gradient descent reaches; learning order; stable vs unstable persistence |
| ActSWM (2607.26712), Shi et al. (2608.04653) | Name the history or statistical shortcut; add action-separation or counterfactual losses | No dynamics law; no link to φ; no test of when it bites |
| AVL-JEPA (2610.03587) | Names the collapse; static Bayes-risk argument | Static; no escape-time or persistence predictions |
| Littwin et al. (2407.03475); Simon et al. (2303.15438) | Stepwise learning order in passive JEPA/SSL | No actions, no controllability, no SIGReg |

**6. Collision search.**
- The deaf mechanism is already named, so it must not be claimed as a discovery.
- The label-free diagnostic is essentially taken.
- Inverse-dynamics fixes are crowded.
- Nothing was found on learning dynamics, escape times, warm-up sufficiency or action-autocorrelation laws. This covered arXiv only; OpenReview-only ICLR 2027 submissions were not searched.
- **Framing:** "a theory of when and why, with a falsifiable warm-up prediction".

**7. Kill test (as proposed by the report).**
- Simulate the linear model (P1, P2).
- Use a procedural 64×64 environment with a controlled dot and an uncontrolled ball, with an adjustable response fraction.
- Compare vanilla, IDM throughout, and IDM warm-up only (first 5%), three seeds each.
- **Decisive persistence test:** if warm-up-only decays back to vanilla, the trap claim is dead.
- Then sweep φ × history and run inference on the public quentinll/lewm checkpoints (CAS with full versus truncated history).

**8. Appeal.** It is about LeWM and their own design principle. It answers whether SIGReg is the culprit (consistent with SLIM appendix H), and reconciles Toso et al. with the empirical success of IDM. Citation potential is high.

**9. Risks.**
- Linear theory may not transfer to ViTs.
- "Just stepwise SSL plus actions."
- "IDM fixes are known."
- Warm-up-only may fail because SIGReg redistributes variance.
- A concurrent OpenReview paper may appear.

**10. Odds (judgement).** Acceptance about 65 to 75%; spotlight about 25 to 35%; oral about 8 to 12%, conditional on the kill test passing and the theory matching at least P1 and P3.

## Candidate 2: "Look Before You Leap"

**Claim and evidence.**
- If a planner may branch its plan only once its own world model predicts that the hidden possibilities will look different, it learns by itself to make "experiments" that reveal them, with no curiosity bonus.
- **Supporting sources:**
  - Dawid and LeCun (2306.02572) on latent variables for uncertainty and hierarchical planning;
  - LeCun's position paper (OpenReview BZ5a1r-kVsf; quote seen only in search text);
  - EpicWorldModel (2610.05996) still needs a hand-weighted variance bonus.

**Method.**
- A LeWM-scale encoder with a predictor conditioned on a discrete latent mode, trained with a winner-takes-all loss.
- A shared opening plus one continuation per mode; continuations may differ only once predicted embeddings are further apart than 2q_{1-α} (a split-conformal quantile).
- The cost is the prior-weighted goal cost.

**Collision search.** No contingency, dual-control or emergent-probing planner was found for JEPA world models; the control ideas themselves are classical.

**Kill test.**
- **Environment:** hidden-mode TwoRoom.
- **Comparison:** against CEM on a mode-averaged cost and against variance-bonus CEM.
- **Kill if** it does not beat variance-bonus CEM, or does not visit the vantage strip more than chance.

**Risks.**
- Mode collapse.
- The tight timeline.
- The stochastic predictor may do the work (needs an ablation).

**Odds (judgement).** Acceptance 45 to 55%; spotlight 15 to 25%.

## Candidate 3: "What You Can Control, You Can Align"

**Claim and method.**
- Two SIGReg models cannot be aligned from their latents alone (rotation symmetry). Shared actions fix the rotation on the controllable subspace, and only there.
- **Method:** Procrustes on mean multi-horizon action responses (an empirical Krylov matrix); inference only.

**Prior art.**
- Klindt, LeCun, Balestriero (2605.26379): "the learned representation must be a rotation of the true latents".
- Social-JEPA (2603.02263) fits W on shared states.
- Dynamics Cycle-Consistency (2012.09811) has no identifiability statement.

**Collisions.**
- Platonic Representation Hypothesis on World Models (2608.23720) stitches DINO-WM models without actions.
- Xu and Kanan (2608.27651): GW is "exactly flat over the entire group orbit". Cite it.
- HGA (2608.28840) is UNVERIFIED.

**Kill.** Kill if action-based alignment does not clearly beat Gromov-Wasserstein on the controllable subspace.

**Odds (judgement).** Acceptance 40 to 55%; spotlight 10 to 20%.

## Comparison

| Criterion | #1 | #2 | #3 |
|---|---|---|---|
| Novelty after collision search | High for dynamics theory; diagnosis taken | Medium-high in JEPA; classical in control | Medium; crowded |
| Real method | Theory plus derived remedies | Planner plus mode-latent model | Closed-form estimator |
| Compute fit | Good | Tight | Very good |
| Scoop risk by 30 Nov | Medium (check OpenReview) | Low-medium | Medium |
| Judgement: acceptance / spotlight | 65-75% / 25-35% | 45-55% / 15-25% | 40-55% / 10-20% |

## Recommendation

Lead with #1 and run its kill test first. The persistence test (warm-up on, then off) is the deciding experiment. Do not frame the paper as "a new failure mode" or "a new diagnostic". If the kill test fails, switch to #3. Treat #2 as a separate submission only with a collaborator.

## Caveats

- Odds are judgement; ICWM has no published acceptance rate or tier structure.
- **Not re-verified in the report:**
  - the ICWM CFP page;
  - the Spectral-Target "control ball" quote;
  - CAER;
  - HGA;
  - LeCun's talk quotes;
  - the OpenReview position-paper quote.
- OpenReview-only ICLR 2027 submissions were not searched.
- All compute figures are planning estimates.
