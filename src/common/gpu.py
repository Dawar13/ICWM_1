"""Polite use of a shared GPU.

The A100 is shared with two or three other people's experiments. Rules enforced here:
- never take more than a capped slice of memory (per-process memory fraction);
- refuse to start if free memory is below a floor, instead of evicting others;
- never touch other processes.
"""
import os
import subprocess

import torch


def gpu_report() -> str:
    """Return a short nvidia-smi summary (memory, utilisation, processes). Safe if nvidia-smi is missing."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        return "GPU: " + out.stdout.strip() + "\nprocesses: " + (procs.stdout.strip() or "none listed")
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi unavailable ({exc})"


def pick_device(max_mem_gb: float = 8.0, min_free_gb: float = 4.0, headroom_gb: float = 2.0,
                allow_cpu: bool = True) -> torch.device:
    """Pick the GPU if enough memory is free, and cap this process's share of it.

    max_mem_gb: hard cap for this process.
    min_free_gb: refuse to start (or fall back to CPU) if less than this is free.
    headroom_gb: always leave at least this much free for the other users.
    """
    if not torch.cuda.is_available():
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("CUDA not available")
    dev_index = int(os.environ.get("LOCAL_GPU_INDEX", "0"))
    torch.cuda.set_device(dev_index)
    free, total = torch.cuda.mem_get_info(dev_index)
    free_gb, total_gb = free / 1e9, total / 1e9
    if free_gb < min_free_gb:
        msg = (f"Only {free_gb:.1f} GB free of {total_gb:.1f} GB on the shared GPU "
               f"(floor {min_free_gb} GB). Not starting, to avoid hurting other jobs.")
        if allow_cpu:
            print(msg + " Falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError(msg)
    cap_gb = max(0.5, min(max_mem_gb, free_gb - headroom_gb))
    frac = min(0.95, cap_gb / total_gb)
    torch.cuda.set_per_process_memory_fraction(frac, dev_index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Using cuda:{dev_index}, capped at {cap_gb:.1f} GB ({frac:.2%} of {total_gb:.1f} GB); "
          f"{free_gb:.1f} GB was free at start.")
    return torch.device(f"cuda:{dev_index}")


def peak_mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0
