# Collision check: "Listening Is Learned Last"

**Date:** 8 Nov 2026. **Scope:** Stage 0 novelty check for the lead idea in `docs/01_lead_theory.md`.

## What counts as a collision

- **(a) Warm-up persistence.** An auxiliary inverse-dynamics (or action-prediction) loss is used only early in training and then removed, and action use or agent encoding persists afterwards.
- **(b) Trap vs objective boundary.** A learning-dynamics account of when an action-conditioned self-predictive, JEPA or BYOL-style world model learns to use actions. This includes saddles, plateaus, local minima, learning order or multistability.

## Sources searched and queries run

**arXiv** (export.arxiv.org API, abstract-field search, sorted by relevance):
1. `"inverse dynamics" AND "world model" AND (warm OR warmup OR "warm-up" OR anneal OR annealing OR schedule OR curriculum)`
2. `"learning dynamics" AND "action-conditioned" AND (JEPA OR "world model" OR "self-predictive")`
3. `JEPA AND "world model" AND (saddle OR plateau OR "local minimum" OR "local minima" OR "training dynamics" OR "gradient dynamics" OR "gradient flow")`
4. `"inverse dynamics" AND auxiliary AND (warm-up OR warmup OR "early training" OR "switched off" OR "turned off" OR removed OR annealed OR anneal)` (0 hits)
5. `"self-predictive" AND action AND ("learning dynamics" OR "training dynamics" OR ODE OR "gradient flow")` (only hit: Khetarpal et al.)
6. `"world model" AND ("ignores actions" OR "ignore actions" OR "ignore the action" OR "action-blind" OR "insensitive to actions" OR "action insensitive" OR "action-insensitive")`
7. `"inverse dynamics" AND (JEPA OR "latent world model" OR "self-predictive") AND (theory OR theoretical OR linear OR dynamics)` (21 hits, all screened)
8. `controllable AND uncontrollable AND ("learning dynamics" OR "training dynamics" OR "learned first" OR "learning order" OR "gradient descent")`
9. `"action-conditioned" AND linear AND (JEPA OR "joint-embedding") AND (dynamics OR saddle OR incremental OR stepwise)`
10. `"auxiliary loss" AND ("warm-up" OR warmup OR "only early" OR "early phase" OR "critical period") AND representation AND (persist OR persists OR removed OR removal)` (0 hits)
11. `BYOL AND action AND ("learning dynamics" OR "training dynamics" OR saddle OR "critical point" OR multistab*)` (0 hits)
12. `"inverse dynamics" AND (pretraining OR pre-training OR "pre-train") AND ("world model" OR "latent dynamics") AND representation`
13. `"critical learning periods" AND (linear OR multimodal OR pathway)`
14. `"world model" AND action AND ("optimization trap" OR "optimisation trap" OR "bad local" OR "spurious local" OR "poor local" OR trapped)`
15. `predictor AND action AND encoder AND ("chicken-and-egg" OR "co-adaptation" OR "mutual dependence" OR bootstrap) AND "world model"`
16. `"latent world model" AND ("warm-up" OR warmup OR "warm start") AND action`
17. `"world model" AND ("action pathway" OR "action conditioning") AND ("learning order" OR "learned late" OR "learned last" OR delayed OR dormant)`
18. `"self-predictive" AND (controllable OR "agent-centric" OR exogenous) AND (dynamics OR theory)`
19. `"inverse dynamics" AND ("early stopping" OR "two-stage" OR "two stage") AND JEPA` (0 hits) and `auxiliary AND "inverse dynamics" AND persist` (0 hits)
20. Phrases from the OpenReview hit below, to look for an arXiv version: `"predictable nuisances"`, `"gain capacity early"`, `"world model" AND "easy-to-predict"`, `"action-conditioned consequences"`, and its title (none found)

**OpenReview, ICLR 2027** (api2.openreview.net `notes/search`, group `ICLR.cc/2027/Conference`, up to 50 results per term, 727 unique submissions screened by title and abstract). Terms: inverse dynamics world model; action-conditioned JEPA; learning dynamics world model actions; JEPA world model collapse action; world model ignores actions; inverse dynamics warm-up; auxiliary loss warm-up; self-predictive learning dynamics action; saddle action-conditioned; action-blind world model; LeWorldModel; SIGReg; inverse dynamics auxiliary schedule; action sensitivity world model; BYOL-AC; controllable uncontrollable representation learning dynamics; linear JEPA dynamics; implicit bias JEPA; stepwise learning self-supervised; world model action shortcut history; inverse dynamics pretraining then removed; auxiliary loss annealing representation; critical learning period world model; encoder predictor co-adaptation action; easy-to-predict features suppress.
**Limit:** OpenReview's search endpoint worked, but its note, PDF and attachment endpoints and its web pages all returned a browser check (HTTP 403). So for OpenReview-only submissions, only the title, abstract, TLDR and keywords were read.

**Semantic Scholar** (graph API paper search). The service rate-limited us (HTTP 429). Only "linear JEPA action conditioning dynamics", "action-conditioned self-predictive learning saddle" and "auxiliary loss curriculum removed after warmup representation persists" (0 hits) returned results. The other suggested queries got no answer, so Semantic Scholar coverage is partial.

**Google Scholar** could not be queried directly. As a stand-in, we ran general web searches: "learning dynamics action-conditioned JEPA world model encoder ignores agent"; "inverse dynamics warm-up world model auxiliary loss switched off representation persists"; "openreview ICLR 2027 action-conditioned world model learning dynamics saddle inverse dynamics"; "self-predictive representation learning actions saddle point plateau gradient dynamics controllable features learned late"; "world model learns to use actions late training dynamics encoder predictor coupling theory 2026"; "critical period early training auxiliary inverse dynamics loss removed later representation retains controllable state"; "JEPA world model multistability bistable"; "scholar action-conditioned world model learning dynamics inverse dynamics warmup ablation 2026"; and the exact OpenReview title.

**Full-text greps.** We searched the full texts of SLIM, Toso et al., ActSWM, Shi et al., AVL-JEPA, Khetarpal et al., Littwin et al., SMWM, AC-MTM, DA-LeWM, Learning Intrinsic Dynamics, Delta-JEPA, Toward Physically Grounded JEPA and LeWM. The search terms were warm-up, anneal, schedule, curriculum, switch or turn off, disable, saddle, plateau, local minimum, learning or training dynamics, and learning order. None of these papers turns its IDM or action loss off partway through training. The only warm-ups we found are learning-rate warm-ups.

## Hits

| ID | Title (short) | Class | One-line reason |
|---|---|---|---|
| OpenReview t6sUZeEbkt | When Prediction Chooses the Representation: Learning Dynamics and Planning in World Models (anonymous, ICLR 2027 submission, record created 12 Sep 2026) | COLLISION (probable; judged from abstract, TLDR and keywords only) | It gives a learning-dynamics account of learning order in jointly trained action-conditioned world models: "Predictive training creates a competition: easy-to-predict features gain capacity early, accelerate their own dynamics learning, and suppress harder features that determine which actions succeed". It adds "Constraints on marginal latent distributions cannot rule out this failure" and recommends that "constraints should become silent after forming planning-compatible geometry". Primary area: learning theory; keywords include "training dynamics". This takes our P1 and the "theory of when and why" framing. It is unknown whether it also shows a trap vs objective boundary, or tests a warm-up that is switched off |
| 2610.07540 | Toso, LeCun, Anderson, Bounou | PARTIAL | Objective level only: an existence result (Lemma 2) plus guarantees for exact EP-IDM. The IDM is on at constant weight throughout; there is no training dynamics, no schedule, and no account of which minimum gradient descent reaches. They state the question ("whether standard JEPA training can learn them in the first place") but answer it only at the objective level. Left for us: dynamics, the trap, warm-up persistence |
| 2610.03137 | SLIM (Strohm et al.) | PARTIAL | Phenomenon, response-fraction dose, SIGReg not the cause, repair through the encoder. The IDM is on throughout; only its weight is swept. They assert an objective-level reading ("can be minimised by an action-insensitive encoder") without testing it. Left for us: the theory, the trap vs objective test, warm-up persistence |
| 2406.02035 | Khetarpal et al. (BYOL-AC) | PARTIAL | Fixed points of passive (BYOL-Π) vs action-conditioned (BYOL-AC) objectives correspond to our two rankings. But the predictor is assumed optimal at every step, there is no learning order and no controllable vs uncontrollable split, and they say stability of sub-optimal critical points is future work. Left for us: the coupling trap, the order, the warm-up |
| OpenReview OAiyzoDnYk | Understanding Feature Learning Dynamics of SIGReg via BHEP Statistics | PARTIAL | Learning dynamics of SIGReg: features emerge one after another with growing delays. No actions. Left for us: action conditioning, controllable vs autonomous factors, the warm-up |
| OpenReview ZuWTHE4KvU | Temporal Action Structure Controls Representation Identifiability in Predictive World Models | PARTIAL (for P2 only) | Shows at the objective level that the temporal structure of actions decides which representation the loss prefers. Not training dynamics; not about dropping the agent |
| 2610.03587 | AVL-JEPA | PARTIAL | Static Bayes-risk argument for why predictability beats causal relevance. Two-stage training, but the action loss stays on. Left for us: the dynamics |
| 2607.26712; 2608.04653 | ActSWM; Shi et al. (CoCo) | BACKGROUND | Name the history and statistical shortcut; their fixes stay on; no dynamics law, no φ |
| 2407.03475; 2303.15438 | Littwin et al.; Simon et al. | BACKGROUND | Learning order in passive JEPA and SSL; no actions |
| 2212.03319 | Tang et al., self-predictive RL | BACKGROUND | Self-predictive learning dynamics under a fixed policy; no action conditioning |
| 2308.12221; 2210.04643 | Kleinman, Achille, Soatto (critical learning periods) | BACKGROUND | Early-training effects persist in deep linear networks; competition between sources. No actions or world models. Likely reviewer framing for our persistence result: cite and contrast |
| 1312.6120 | Saxe, McClelland, Ganguli | BACKGROUND | Plateaus and transitions in deep linear networks |
| 2603.19312 | LeWorldModel | BACKGROUND (supports us) | The action pathway is zero-initialised so that "action conditioning impacts the predictor training progressively" |
| 2609.23881; 2608.24044 | MotionJEPA; JEPA-x | BACKGROUND | Slow-feature bias, and co-adaptation to easy-to-predict transitions; fixes stay on; no dynamics law |
| 2608.06706 | Dueling World Models | BACKGROUND | Action-blindness while the loss still falls; readout fix; no dynamics |
| 2606.30068 | Pendharkar | BACKGROUND | Objective-level discarding of exogenous control-relevant features |
| 2608.18746 | DA-LeWM | BACKGROUND | An IDM speeds convergence, measured empirically; IDM on throughout |
| 2606.20104; 2608.17542; 2606.31232; 2609.03565; 2610.21612; 2609.30264; 2610.00727 | SMWM; AC-MTM; Delta-JEPA; Physically Grounded JEPA; Learning Intrinsic Dynamics; AD-WM; CF-JEPA | BACKGROUND | IDM or action-recovery losses kept on for all of training; full-text greps of the first five found no schedule or switch-off |
| OpenReview 3AZ8FMQEX4 | Inverse-Conditioned World Models (IC-LeWM) | BACKGROUND | Inverse co-training and an action-Jacobian regulariser, both on throughout |
| OpenReview enLPR30GmE; 2510.26433; 2603.29844 | JEVA; CoLA-World; DIAL | BACKGROUND | Warm-up stages exist, but they align latent-action models or VLA modules. They are not an IDM loss used to make a JEPA encoder keep the agent |
| OpenReview LtrgTUjYVy | Beyond Spread (regulariser switched mid-training) | BACKGROUND | Passive JEPA; after a regulariser switch, the final probe is "only partially plastic"; no actions |
| 1612.07307 | Shelhamer et al., Loss is its own Reward | BACKGROUND | Self-supervised pre-training, including inverse dynamics, before RL; not a world-model persistence test |
| 2609.32966 | Self-Confirming Superposition Traps in RL | BACKGROUND | Trap concept in the RL loop between policy and data; not world-model encoders |

## Notes for the go/no-go

- **(a) Warm-up persistence:** no collision found. No paper we could read uses an IDM or action loss only early and then removes it in an action-conditioned JEPA or self-predictive world model. t6sUZeEbkt's advice that "constraints should become silent" comes close in spirit. Whether it tests a switch-off is unknown without the PDF.
- **(b) Learning dynamics of action use:** t6sUZeEbkt appears to cover the learning-order part (our P1), with a theory paper framing (primary area "learning theory"). The explicit trap vs objective boundary (P3: the agent kept only when it is in the top k by action-conditioned predictability, and persistence after switch-off) was not found in any paper we could read. Its status in t6sUZeEbkt is unknown.
- **Action needed before deciding:** read the t6sUZeEbkt PDF in a logged-in browser (https://openreview.net/forum?id=t6sUZeEbkt). Check whether it: (i) shows that the suppressed solution has higher loss than one that keeps the decision-relevant features (a trap); (ii) runs a constraint that is switched off or decays, and tests persistence; (iii) uses SIGReg or LeWM; (iv) has a linear model. If (i) and (ii) are both absent, the lead can be repositioned around the trap vs objective boundary and the warm-up persistence test, citing t6sUZeEbkt for learning order.
- **AI report claims checked:** the report said it found no learning-dynamics paper on this question. It also said it had not searched OpenReview. The OpenReview search contradicts the first claim. All arXiv IDs in our lit table resolve correctly; see `notes/lit_table.md` for the Social-JEPA withdrawal and the Cheung and He author fix.

**Verdict: COLLISION FOUND** (on (b), learning order, from OpenReview t6sUZeEbkt, judged from its abstract, TLDR and keywords only; no collision found for (a) warm-up persistence)
