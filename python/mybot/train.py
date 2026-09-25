"""Fit a LinearPolicy from GameLogger jsonl files.

    python -m mybot.train
    python -m mybot.train --logs logs --out models/policy.npz --epochs 60

The output is a numpy archive. Copy it to `bwapi-data/read/policy.npz` (tournament)
or set `BWBOT_MODEL`. Inference does not import torch.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM
from .logger import default_log_dir, load_decisions
from .model import N_BUILD, N_TACTIC, LinearPolicy, fit, softmax

log = logging.getLogger("mybot.train")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit mybot LinearPolicy from jsonl game logs.")
    ap.add_argument("--logs", type=Path, default=None, help="folder of *.jsonl (default: BWBOT_LOG_DIR or ./logs)")
    ap.add_argument("--out", type=Path, default=Path("models/policy.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.2)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s")
    folder = args.logs or default_log_dir()
    xs, yt, yb, wins, games = load_decisions(folder)
    if not xs:
        log.error("no decisions in %s — play some games with logging on first", folder)
        return 1
    X = np.asarray(xs, dtype=np.float32)
    if X.shape[1] != FEATURE_DIM:
        log.error("feature width %d != FEATURE_DIM %d (old logs?)", X.shape[1], FEATURE_DIM)
        return 1
    y_t = np.asarray(yt, dtype=np.int64)
    y_b = np.asarray(yb, dtype=np.int64)
    log.info("loaded %d decisions from %d games (%d wins) in %s", len(X), games, wins, folder)

    model = fit(X, y_t, y_b, epochs=args.epochs, lr=args.lr)
    pt = softmax(X @ model.tactic_w.T + model.tactic_b)
    pb = softmax(X @ model.build_w.T + model.build_b)
    acc_t = float((pt.argmax(1) == y_t).mean())
    acc_b = float((pb.argmax(1) == y_b).mean())
    log.info("train acc tactic=%.3f build=%.3f (classes %d/%d)", acc_t, acc_b, N_TACTIC, N_BUILD)
    model.save(args.out)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
