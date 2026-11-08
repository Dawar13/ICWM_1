"""Run directories, config hashing and JSON-lines logging."""
import hashlib
import json
import os
import socket
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]


def code_hash() -> str:
    """Hash of every .py file under src/, so each result is tied to the exact code."""
    h = hashlib.sha256()
    for p in sorted(CODE_ROOT.rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


class RunLog:
    def __init__(self, out_dir: str, cfg: dict):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = {"config": cfg, "config_hash": config_hash(cfg), "code_hash": code_hash(),
                "host": socket.gethostname(), "start_unix": time.time()}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self.f = open(self.dir / "log.jsonl", "a")
        self.t0 = time.time()

    def log(self, **row):
        row["wall_s"] = round(time.time() - self.t0, 2)
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def finish(self, **summary):
        summary["wall_s"] = round(time.time() - self.t0, 2)
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.f.close()


def already_done(out_dir: str) -> bool:
    return (Path(out_dir) / "summary.json").exists()
