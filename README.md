# ICWM 2027 go/no-go kit

**Purpose:** decide, with pre-registered experiments, whether our lead paper idea holds before we write anything.

**The lead idea, "Listening Is Learned Last":** JEPA world models can get stuck in a training trap where they ignore the agent they control, even though keeping it would lower their own loss. If so, a short inverse-dynamics warm-up, switched off afterwards, should be enough to escape. The fallback is "What You Can Control, You Can Align".

## Quick start

1. Unzip on the A100 machine. `cd icwm_gonogo`.
2. Run `bash setup.sh`. It creates `.venv` in this folder and needs no sudo.
3. Open the folder in VS Code with Claude Code, and paste the first prompt from `START_PROMPT.md`.

## What is inside

- `CLAUDE.md`: rules and autonomy, auto-loaded by Claude Code.
- `CONTEXT.md`: the full story.
- `docs/`: the theory, the protocol, the prior art, the GPU-sharing rules and the standards.
- `prereg/`: pre-registrations with thresholds that match `src/analysis/gates.py`.
- `src/`: code (all smoke-tested on CPU while building the kit):
  - `g1_linear/sim.py`: linear action-conditioned JEPA (CPU);
  - `g2_pixel/`: GPU-rendered sprite world, tiny LeWM-like JEPA, training with probes, CEM planning evaluation;
  - `g3_lewm_ckpt/README.md`: the spec for inference on the public LeWM checkpoints (code to be written after reading LeWM);
  - `g4_align/`: action-identified alignment (synthetic, and between two tiny models);
  - `run_sweep.py` and `sweeps.py`: polite sweep runner and registry;
  - `analysis/gates.py`: pre-registered PASS / FAIL verdicts.
- `notes/`: logs and templates, including `smoke_tests.md` (what was already tried, honestly labelled as not evidence).
- `references/`: LeCun transcripts, and the 8 Nov research report (AI-generated; verify before use).
