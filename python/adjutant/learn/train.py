"""Train learned components from recorder logs.

    python -m adjutant.learn.train strategy --logs runs/explore1 runs/explore2 --out models/strategy.npz
    python -m adjutant.learn.train engage   --logs runs/... --out models/engage.npz
    python -m adjutant.learn.train tactics  --logs runs/... --out models/tactics.npz
    python -m adjutant.learn.train micro    --logs runs/... --out models/micro.npz

strategy: every `strategy/ctx` row (context + active template) is one sample labelled with the
game result; each game's rows share weight 1 so long games do not dominate, and the validation
split is by game.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from blackboard.recorder import iter_games
from blackboard.train import fit

from .features import CTX_SPEC, strategy_input, strategy_spec

log = logging.getLogger("adjutant.train")


def strategy_dataset(dirs: Sequence[Path], templates: Optional[Sequence[str]] = None, bot: Optional[str] = None):
    games = []
    skipped = defaultdict(int)
    seen: set[str] = set()
    for g in iter_games(*dirs):
        if bot and not str(g["header"].get("bot", "")).startswith(bot):
            skipped["bot"] += 1
            continue
        rows = [r for r in g["rows"] if r.get("slot") == "strategy" and r.get("k") == "ctx"]
        good = [r for r in rows if len(r.get("x", ())) == CTX_SPEC.dim]
        if len(good) < len(rows):
            skipped["feature_width"] += len(rows) - len(good)
        if not good:
            skipped["no_rows"] += 1
            continue
        games.append((g["header"].get("game_id", g["path"]) + g["header"].get("side", ""), bool(g["end"]["won"]), good))
        seen.update(r["template"] for r in good)
    names = list(templates) if templates else sorted(seen)
    index = {t: i for i, t in enumerate(names)}
    X, y, groups, w = [], [], [], []
    stats: dict[str, list[int]] = {t: [0, 0, 0] for t in names}      # rows, games, wins
    for gid, won, rows in games:
        rows = [r for r in rows if r["template"] in index]
        if not rows:
            continue
        for t in {r["template"] for r in rows}:
            stats[t][1] += 1
            stats[t][2] += int(won)
        for r in rows:
            X.append(strategy_input(np.asarray(r["x"], np.float32), index[r["template"]], len(names)))
            y.append(float(won))
            groups.append(gid)
            w.append(1.0 / len(rows))
            stats[r["template"]][0] += 1
    if not X:
        return None, names, stats, dict(skipped), 0
    return (np.stack(X), np.asarray(y, np.float32), np.asarray(groups), np.asarray(w, np.float32)), names, stats, \
        dict(skipped), len(games)


def train_strategy(args) -> int:
    data, names, stats, skipped, n_games = strategy_dataset([Path(p) for p in args.logs], args.templates, args.bot)
    print(f"games {n_games}  skipped {skipped}")
    for t, (rows, games, wins) in stats.items():
        print(f"  {t:16s} rows {rows:5d}  games {games:4d}  win {wins / max(1, games):.1%}")
    if data is None:
        print("no strategy rows found (play Explore games first)", file=sys.stderr)
        return 1
    X, y, groups, w = data
    spec = strategy_spec(names)
    model, rep = fit(X, y, task="binary", hidden=args.hidden, labels=("win",), spec=spec, weights=w, groups=groups,
                     epochs=args.epochs, l2=args.l2, lr=args.lr, seed=args.seed,
                     meta={"templates": names, "games": n_games})
    out = model.save(Path(args.out))
    print(f"{rep}\nsaved {out}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="adjutant.learn.train")
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("strategy", help="per-template win model for LearnedStrategy")
    s.add_argument("--logs", nargs="+", required=True, help="log dirs / files (searched recursively)")
    s.add_argument("--out", default="models/strategy.npz")
    s.add_argument("--templates", nargs="*", help="template order (default: all seen, sorted)")
    s.add_argument("--bot", help="only games whose header bot starts with this (e.g. adjutant)")
    s.add_argument("--hidden", type=int, nargs="*", default=[])
    s.add_argument("--epochs", type=int, default=300)
    s.add_argument("--l2", type=float, default=1e-3)
    s.add_argument("--lr", type=float, default=1e-2)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("belief", help="LearnedBelief heads from truth rows (complete-map-information games)")
    s.add_argument("--logs", nargs="+", required=True)
    s.add_argument("--out", default="models", help="directory for belief_*.npz")
    s.add_argument("--hidden", type=int, nargs="*", default=[32])
    s.add_argument("--epochs", type=int, default=200)
    s.add_argument("--l2", type=float, default=1e-4)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("engage", help="engagement predictor (win probability) from engagement/fight rows")
    s.add_argument("--logs", nargs="+", required=True)
    s.add_argument("--out", default="models/engage.npz")
    s.add_argument("--hidden", type=int, nargs="*", default=[16])
    s.add_argument("--epochs", type=int, default=300)
    s.add_argument("--l2", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("tactics", help="tactics value model from tactics/decision rows")
    s.add_argument("--logs", nargs="+", required=True)
    s.add_argument("--out", default="models/tactics.npz")
    s.add_argument("--horizon", type=int, default=45, help="seconds of value trade per decision")
    s.add_argument("--win-weight", type=float, default=0.3, help="weight of the game result in the target")
    s.add_argument("--hidden", type=int, nargs="*", default=[32])
    s.add_argument("--epochs", type=int, default=300)
    s.add_argument("--l2", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("micro", help="RLMicro Q model (fitted Q iteration) from micro/step transitions")
    s.add_argument("--logs", nargs="+", required=True)
    s.add_argument("--out", default="models/micro.npz")
    s.add_argument("--gamma", type=float, default=0.8)
    s.add_argument("--iters", type=int, default=4)
    s.add_argument("--hidden", type=int, nargs="*", default=[32])
    s.add_argument("--epochs", type=int, default=60)
    s.add_argument("--l2", type=float, default=1e-4)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.what == "micro":
        from .micro import train_micro
        return train_micro(args)
    if args.what in ("engage", "tactics"):
        from .combat import train_engage, train_tactics
        return (train_engage if args.what == "engage" else train_tactics)(args)
    if args.what == "strategy":
        return train_strategy(args)
    if args.what == "belief":
        from .belief import train_belief
        return train_belief(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
