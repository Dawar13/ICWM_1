# 05. Candidate #2: "Look Before You Leap" (separate effort; not part of this go/no-go)

## Plain language

When a hidden fact decides the right plan (the door is at the top or the bottom), let the planner write one opening move plus one continuation per possibility. Continuations may differ only once the world model predicts that the possibilities will *look* different, by more than a calibrated error margin (2 q_{1-α}, a split-conformal quantile). Probing ("go and look") then emerges by itself, because it is the only way to unlock specialised continuations. No curiosity bonus is needed.

## Why it is not in this go/no-go

It needs a new hidden-mode environment, a mode-latent predictor (winner-takes-all over K modes), and the planner. That is more than the other two candidates combined. Run it only if a teammate takes it on.

## Closest work

| Work | Status | Relation |
|---|---|---|
| EpicWorldModel, 2610.05996, NeurIPS 2026 | [S] | Stochastic flow-matching JEPA with a variance bonus β. The stochastic predictor alone gave a large gain over LeWM, so an ablation is required |
| Dual control (Feldbaum); dual stochastic MPC (Arcari et al.); contingency MPC | [U] | Classical; cite precisely |
| Denil et al., ICLR 2017, 1611.01843 | [V] search | RL agents that learn to experiment |
| UWM-JEPA, 2605.25313 | [S] | Belief carried through rollouts |

## Kill test (from the research report)

- **Environment:** hidden-mode TwoRoom; the door is at the top or the bottom and is visible only from a vantage strip.
- **Comparison:** the gated contingency planner against CEM on a mode-averaged cost, against a variance-bonus CEM, and against a probe-then-plan rule; three seeds.
- **KILL** if it does not beat the variance-bonus baseline, or does not visit the vantage strip more often than chance.
