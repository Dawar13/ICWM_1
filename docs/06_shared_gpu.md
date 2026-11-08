# 06. Working on a shared A100 (a friend's machine, 2 or 3 other jobs running)

The other people's experiments come first. Every rule below exists to make sure we never slow down, crash or evict them.

## Before anything

- Run `nvidia-smi`. Note the free memory and the utilisation in `notes/daily_log.md`.
- **Never** kill, renice, signal or otherwise touch a process that is not ours. Not even a stuck one: tell the user.
- **No sudo, no system-wide installs.** Everything goes into `.venv` inside this folder (`setup.sh`).
- Keep disk use inside this folder. Delete our own large checkpoints when a gate is decided. Check `df -h` before big sweeps.

## Memory

- Every GPU script calls `src/common/gpu.py:pick_device`, which:
  - caps this process at `max_mem_gb` (default 6 GB) with `torch.cuda.set_per_process_memory_fraction`;
  - always leaves at least 2 GB free for others;
  - refuses to start, falling back to CPU, if less than 4 GB is free.
- The tiny pixel model needs well under 2 GB at batch 128. Do not raise `max_mem_gb` without a reason recorded in the daily log.
- On an out-of-memory error, halve the batch and retry. Never take more memory.

## Compute

- **G1 and G4-synthetic run on CPU.** The sweep runner hides the GPU from them and limits each process to 2 threads.
- **G2 runs one at a time by default** (`--parallel 1`). Before each launch the runner waits until at least `--min_free_gb` (default 10 GB) is free.
  - Increase parallelism only if utilisation is low and memory is plentiful.
  - Record the decision in the daily log.
- **Run long sweeps under `nohup` or `tmux`,** logging to `runs/.../stdout.txt`. Runs are resumable: finished runs are skipped.
- **Use bf16 autocast** (already in the code); A100s support it.

## Throughput honesty

On a shared GPU, wall-clock is noisy:
- log steps per second (already in `log.jsonl`);
- never compare methods by wall-clock;
- compare by steps and by number of predictor calls.

## Budget (ask the user before crossing it)

- Any single run expected to exceed **2 GPU-hours**.
- The go/no-go phase as a whole exceeding **20 GPU-hours** (sum of `wall_h` in `runs/compute_ledger.csv` for GPU rows).
