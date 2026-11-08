# 08. LeCun's stated open problems (from the earlier kit; candidate letters A, B, C there refer to old candidates)

Quotes come from user-supplied transcripts in `references/transcripts/`. They are automatic transcripts, so wording may be slightly off. Before quoting in the paper, find a citable source (a paper, an official video or slides) and cite that.

**Transcript files (in references/transcripts/):**
- `harvard_cmsa_talk.txt`: talk at Harvard CMSA (Geometry of Machine Learning); he is still at Meta.
- `world_models_talk_post_meta.txt`: a recent world-models talk in which he mentions leaving Meta and founding AMI Labs. Includes the guardrail Q&A.
- `lex_fridman_podcast.txt`: his third Lex Fridman podcast appearance.

## Problems, quotes and current status

| # | Problem | Quote (abridged) | Source | Lane status (old kit) | Candidate |
|---|---|---|---|---|---|
| P1 | Hierarchical planning with **learned** levels of abstraction | "How do we train a system like this to learn the appropriate level of abstractions? ... completely unsolved, it's wide open." / "Nobody knows how to train a system to learn the appropriate multiple levels of representation so that hierarchical planning works." | Harvard; Lex | Hierarchy saturated, but learned levels still open | Seed S5 |
| P2 | Planning when the world model has latent variables | "the planning problem becomes even more complicated ... because you don't know the value of latent and you have to plan in the context of uncertainty" | Harvard | Model side crowded; planner side open | **A** |
| P3 | Non-smooth, discontinuous cost in action space | Going left or right around the desk: "a very small change, yet it's going to result in a discontinuous change in cost ... major issues in optimization here that are not really solved" | Harvard | Planner lane saturated | none (graveyard idea 1) |
| P4 | Guardrails in representation space | Q: "How do I even get a constraint like don't bump into the wall into this representation space?" A: "a very small head on top of your representation ... a very small number of samples ... basically that requires two samples" | Post-Meta talk Q&A | Guardrails saturated (SLS²); the sample-complexity claim is untested | Seed S3; **B** stress-tests it |
| P5 | Information content estimates are only upper bounds | "every empirical measure that we have are all upper bounds. So, what do we do? We come up with a good upper bound and we cross our fingers." | Post-Meta talk | Not researched | none yet |
| P6 | Why EMA distillation does not collapse | "Somehow this works, somehow this doesn't collapse. Why? I don't know, it's kind of mysterious." | Harvard | Partly addressed (theory papers); LeWM avoids EMA | none |
| P7 | Intrinsic safety by construction | "a system constructed this way can be made intrinsically safe because it has to ... optimize the guardrail objective with every output it produces" / "you can't jailbreak a system like this" | Post-Meta talk | The claim assumes the world model is right; untested under search pressure | **B** |
| P8 | Control of complex systems that cannot be written as equations | "a turbojet or a chemical plant or a patient ... learn a energy model of the whole system" / AMI: "control of industrial processes ... anything that is high-dimensional, continuous, and noisy" | Post-Meta talk | AMI's focus; deployment issues under-studied | **C**, seeds S1, S2 |
| P9 | Sample efficiency and zero-shot tasks | "learn to drive in about 20 hours"; the 10-year-old clearing the table without training | Both talks | Broad | none directly |
| P10 | Learning by experiment | 8 to 9 month-olds throw toys off the high chair: "They're doing the experiment that gravity actually applies to everything." | Post-Meta talk | Active probing in JEPA planners appears open | **A** |
| P11 | Planning when information arrives later | "I don't have the information. I don't know ... how long I'm going to have to wait before a taxi stops ... I have to do hierarchical planning." | Post-Meta talk | Links hierarchy to information; supports recourse planning | **A** |
| P12 | Switching between deliberate and reactive control | "There's a point in this hierarchy where ... I may not actually need to plan formally ... revert to system one" | Harvard | Amortisation saturated; the switching rule is less studied | none (graveyard idea 5) |
| P13 | Latent variables as unperceived information | "the latent variable ... is supposed to represent all the information about the world that you don't perceive yet" | Lex | Same as P2 | **A** |

## V-JEPA 2 stated limitations (from the paper; verified [V])

| Limitation | Where | Candidate |
|---|---|---|
| Sensitivity to camera position; inferred action axes rotate with the camera | Section 4.3, Appendix B.4 | **C** |
| Long-horizon planning: error accumulation and search-space size | Section 4.3 | Saturated lanes |
| Image goals only | Section 4.3 | Language-goal heads exist (Keeping JEPA World Models Plannable) |
| 16 s per action; future work suggests a feed-forward policy initialiser or gradient planning | Section 4.2 / Table 3 | Amortisation saturated; seed S1 (latency) |
| Pick-and-place needs hand-given sub-goals | Section 4.2 | Hierarchy lane |

## Framing advice

- LeCun prefers energy-based framing, joint-embedding models, planning by optimisation, and minimal reinforcement learning.
- He dislikes generative pixel prediction, probabilistic normalisation where it is unnecessary, and hype.
- Frame the paper as making his architecture work better, or testing one of his claims honestly. Do not frame it as an argument against him.
- Candidate B challenges P7 constructively: show the failure, then fix it within his framework.
