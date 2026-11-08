# Prompts for Claude Code

## First session (paste this)

> Read CLAUDE.md and CONTEXT.md, then docs/01_lead_theory.md, docs/02_gonogo_protocol.md and docs/06_shared_gpu.md. We are running the go/no-go for ICWM 2027 on a shared A100: two or three other people's experiments are running on it, so follow docs/06 strictly. Work at agentic speed; do not plan in days or hours.
>
> Do Stage 0 now:
> 1. Run `bash setup.sh` and the GPU check.
> 2. Verify the key papers in notes/lit_table.md, and run the collision search for candidate #1 (learning dynamics, warm-up persistence, trap vs objective), including OpenReview ICLR 2027 submissions.
> 3. Read the real LeWM code and fill notes/lewm_diff.md.
>
> Then continue through Stage 1:
> 1. Derive P1 to P4 in notes/theory.md, with a fresh subagent checking them.
> 2. Finalise prereg/G1.md.
> 3. Run the G1 sweeps on CPU and get the verdicts with src/analysis/gates.py.
>
> Then run Stage 2 (G4 synthetic) and Stage 3 (G2 pixels, starting with the timing run).
>
> Stop at the decision gate with notes/decision.md filled, or earlier if CLAUDE.md says to ask. Keep chat updates to one line each; put the details in notes/.

## Later sessions

> Read CLAUDE.md, the last entry of notes/daily_log.md and the tail of runs/gate_reports.md. Check nvidia-smi. Continue from the current stage in docs/02_gonogo_protocol.md.

## If candidate #1 is killed

> Candidate #1 is killed (see runs/gate_reports.md). Read docs/04_candidate3_align.md. Finish G4a, then train two h=1 tiny models with IDM-always (seeds 0 and 1) on the chosen G2 config and run G4b. Then measure latent isotropy on the public LeWM checkpoints (inference only). Fill notes/decision.md for #3 and stop.
