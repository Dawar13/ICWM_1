"""Run a registered sweep politely.

  python src/run_sweep.py g1_trap                 # G1 runs on CPU, a few processes in parallel
  python src/run_sweep.py g2_trap --parallel 1    # G2 on the shared GPU, one run at a time
  python src/run_sweep.py g2_trap --dry           # list runs without starting them

- Skips runs that already have summary.json (safe to restart).
- For GPU sweeps, checks free GPU memory before each launch and waits (does not evict anyone).
- Appends one line per run to runs/compute_ledger.csv (copy the totals into notes/compute_log.md).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sweeps import SWEEPS  # noqa: E402


def free_gpu_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return float(out[int(os.environ.get("LOCAL_GPU_INDEX", "0"))]) / 1024
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=sorted(SWEEPS))
    ap.add_argument("--parallel", type=int, default=None, help="default: 2 for g1 (CPU), 1 for g2 (GPU)")
    ap.add_argument("--min_free_gb", type=float, default=10.0, help="wait until this much GPU memory is free")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--override", default="{}", help="JSON applied to every config (e.g. steps for a timing run)")
    a = ap.parse_args()
    is_gpu = a.sweep.startswith("g2_")
    par = a.parallel or (1 if is_gpu else 2)
    override = json.loads(a.override)
    runs = [(n, s, {**c, **override}) for n, s, c in SWEEPS[a.sweep]()]
    out_root = ROOT / "runs" / a.sweep.split("_")[0] / a.sweep
    print(f"{a.sweep}: {len(runs)} runs -> {out_root}")
    if a.dry:
        for n, s, c in runs:
            print(" ", n)
        return
    ledger = ROOT / "runs" / "compute_ledger.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not ledger.exists()
    lf = open(ledger, "a", newline="")
    w = csv.writer(lf)
    if new:
        w.writerow(["sweep", "run", "device", "start", "end", "wall_h", "status"])
    procs = []
    queue = [r for r in runs if not (out_root / r[0] / "summary.json").exists()]
    print(f"{len(queue)} to run, {len(runs) - len(queue)} already done")
    while queue or procs:
        for p in procs[:]:
            if p[0].poll() is not None:
                end = time.time()
                w.writerow([a.sweep, p[1], "gpu" if is_gpu else "cpu", int(p[2]), int(end),
                            round((end - p[2]) / 3600, 4), p[0].returncode])
                lf.flush()
                procs.remove(p)
        while queue and len(procs) < par:
            if is_gpu:
                f = free_gpu_gb()
                if f is not None and f < a.min_free_gb:
                    print(f"waiting: only {f:.1f} GB free on the shared GPU")
                    break
            n, script, cfg = queue.pop(0)
            od = out_root / n
            od.mkdir(parents=True, exist_ok=True)
            logf = open(od / "stdout.txt", "w")
            env = dict(os.environ)
            if not is_gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            env.setdefault("OMP_NUM_THREADS", "2")  # stay polite on a shared machine
            env.setdefault("MKL_NUM_THREADS", "2")
            cmd = ["nice", "-n", "10", sys.executable, str(ROOT / script), "--out", str(od), "--cfg", json.dumps(cfg)]
            procs.append((subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT, env=env), n, time.time()))
            print("started", n)
        time.sleep(2 if not is_gpu else 10)
    lf.close()
    print("done")


if __name__ == "__main__":
    main()
