"""Combat learning: the engagement predictor and the tactics value model.

Engagement: every `engagement/fight` row carries `x` = engage_features(...) taken when the fight
opened (the Lanchester sides and estimate plus game context), labelled with `won` (we destroyed
more value than we lost). `LearnedEngagement` replaces the Lanchester win probability with the
model's.

Tactics: `tactics/decision` rows (every few seconds while a main squad exists) carry
`x` = tactics_features(ctx, option, target info) for the option the main squad was given, and the
running lost-value totals. The label is the value trade over the next `horizon_s` seconds,
(enemy value lost - own value lost) / (own army value + 200), plus `win_weight` x (+1 win / -1 loss).
`LearnedTactics` scores every option with the model and picks the best.

    python -m adjutant.learn.train engage  --logs runs/a runs/b --out models/engage.npz
    python -m adjutant.learn.train tactics --logs runs/a runs/b --out models/tactics.npz
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from blackboard.models import FeatureSpec
from blackboard.recorder import iter_games

from .. import engage as E

ENGAGE_VERSION = 1
_e_names = ("own_hp", "enemy_hp", "own_value", "enemy_value", "log_ratio", "lanchester_p", "own_left",
            "enemy_left", "own_rate", "enemy_rate", "own_air", "enemy_air", "own_aa", "enemy_aa",
            "own_cloak", "enemy_cloak", "own_static", "enemy_static", "n_own", "n_enemy",
            "minute", "b_army", "own_army", "hidden_supply")
ENGAGE_SPEC = FeatureSpec("engage", ENGAGE_VERSION, _e_names)

OPTIONS = ("hold", "contain", "attack", "attack_base", "retreat")
STANCES = ("hold", "defend", "contain", "attack", "all_in")
TACTICS_VERSION = 1
_t_names = (("minute", "own_army", "b_army", "global_p", "global_log_ratio", "own_bases", "enemy_bases",
             "workers", "threat", "tank_frac", "air_frac")
            + tuple(f"stance_{s}" for s in STANCES) + tuple(f"opt_{o}" for o in OPTIONS)
            + ("has_target", "target_dist", "target_p", "target_value", "target_static"))
TACTICS_SPEC = FeatureSpec("tactics", TACTICS_VERSION, _t_names)


def _log_ratio(r: float) -> float:
    if not math.isfinite(r) or r > 1e5:
        return 5.0
    if r <= 0:
        return -5.0
    return max(-5.0, min(5.0, math.log(r)))


def _frac(side: E.Side, pred) -> float:
    hp = side.hp
    if hp <= 0:
        return 0.0
    return sum(m.hp for m in side.members if pred(m)) / hp


def engage_features(own: E.Side, enemy: E.Side, est: E.Estimate, ctx: dict) -> np.ndarray:
    e_supply = sum(m.info.supply for m in enemy.members if not m.info.building)
    x = [
        math.log1p(own.hp), math.log1p(enemy.hp), math.log1p(own.value), math.log1p(enemy.value),
        _log_ratio(est.ratio), est.win_prob, est.own_left, est.enemy_left,
        math.log1p(100 * est.own_rate), math.log1p(100 * est.enemy_rate),
        _frac(own, lambda m: m.info.flyer), _frac(enemy, lambda m: m.info.flyer),
        _frac(own, lambda m: m.info.air is not None), _frac(enemy, lambda m: m.info.air is not None),
        _frac(own, lambda m: m.cloaked and not enemy.detects), _frac(enemy, lambda m: m.cloaked and not own.detects),
        _frac(own, lambda m: m.info.building), _frac(enemy, lambda m: m.info.building),
        len(own) / 20, len(enemy) / 20,
        ctx.get("frame", 0) / (24 * 60) / 20, ctx.get("b_army", 0.0) / 100, ctx.get("own_army", 0.0) / 100,
        max(0.0, ctx.get("b_army", 0.0) - e_supply) / 50,
    ]
    return np.asarray(x, np.float32)


def tactics_features(ctx: dict, option: str, target: Optional[dict]) -> np.ndarray:
    """ctx: context shared by all options; target: {dist (px), p, value, static} or None."""
    stance = [float(ctx.get("stance") == s) for s in STANCES]
    opt = [float(option == o) for o in OPTIONS]
    diag = max(1.0, ctx.get("map_diag", 4096.0))
    if target is None:
        tgt = [0.0, 0.0, 1.0, 0.0, 0.0]
    else:
        tgt = [1.0, target["dist"] / diag, target["p"], math.log1p(target["value"]) / 8,
               math.log1p(target["static"]) / 8]
    x = [ctx.get("frame", 0) / (24 * 60) / 20, ctx.get("own_army", 0.0) / 100, ctx.get("b_army", 0.0) / 100,
         ctx.get("global_p", 0.5), _log_ratio(ctx.get("global_ratio", 1.0)) / 5, ctx.get("own_bases", 1) / 4,
         ctx.get("enemy_bases", 1) / 4, ctx.get("workers", 0) / 60, ctx.get("threat", 0.0),
         ctx.get("tank_frac", 0.0), ctx.get("air_frac", 0.0)] + stance + opt + tgt
    return np.asarray(x, np.float32)


# ---------------------------------------------------------------------------- datasets
def engage_dataset(dirs: Sequence[Path]):
    X, y, groups = [], [], []
    skipped: dict[str, int] = defaultdict(int)
    n_games = 0
    for g in iter_games(*dirs, finished_only=False):
        n_games += 1
        gid = g["header"].get("game_id", g["path"]) + str(g["header"].get("side", ""))
        for r in g["rows"]:
            if r.get("slot") != "engagement" or r.get("k") != "fight":
                continue
            snap = r.get("snap") or {}
            x = snap.get("x")
            if x is None or snap.get("fh") != ENGAGE_SPEC.hash or len(x) != ENGAGE_SPEC.dim:
                skipped["no_features"] += 1
                continue
            if r.get("own_lost_value", 0) + r.get("enemy_lost_value", 0) <= 0:
                skipped["no_losses"] += 1      # nothing died: neither a win nor a loss
                continue
            X.append(x)
            y.append(float(r["won"]))
            groups.append(gid)
    return np.asarray(X, np.float32).reshape(-1, ENGAGE_SPEC.dim), np.asarray(y, np.float32), \
        np.asarray(groups), dict(skipped), n_games


def tactics_labels(rows: list[dict], final_lost: Optional[list[float]], won: Optional[bool], horizon: int,
                   win_weight: float) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for i, r in enumerate(rows):
        f = int(r["f"])
        later = next((q for q in rows[i + 1:] if int(q["f"]) >= f + horizon), None)
        end = later["lost"] if later is not None else final_lost
        if end is None:
            out.append(None)
            continue
        d_own = float(end[0]) - float(r["lost"][0])
        d_enemy = float(end[1]) - float(r["lost"][1])
        trade = max(-2.0, min(2.0, (d_enemy - d_own) / (float(r.get("value", 0.0)) + 200.0)))
        res = 0.0 if won is None else (1.0 if won else -1.0)
        out.append(trade + win_weight * res)
    return out


def tactics_dataset(dirs: Sequence[Path], horizon_s: int = 45, win_weight: float = 0.3):
    X, y, groups, opts = [], [], [], []
    skipped: dict[str, int] = defaultdict(int)
    n_games = 0
    for g in iter_games(*dirs, finished_only=False):
        rows = [r for r in g["rows"] if r.get("slot") == "tactics" and r.get("k") == "decision"]
        if not rows:
            continue
        n_games += 1
        good = [r for r in rows if r.get("fh") == TACTICS_SPEC.hash and len(r.get("x", ())) == TACTICS_SPEC.dim]
        skipped["width"] += len(rows) - len(good)
        final = next((r["lost"] for r in g["rows"] if r.get("slot") == "tactics" and r.get("k") == "final"), None)
        won = None if g.get("end") is None else bool(g["end"].get("won"))
        gid = g["header"].get("game_id", g["path"]) + str(g["header"].get("side", ""))
        for r, lab in zip(good, tactics_labels(good, final, won, horizon_s * 24, win_weight)):
            if lab is None:
                skipped["no_label"] += 1
                continue
            X.append(r["x"])
            y.append(lab)
            groups.append(gid)
            opts.append(r["opt"])
    return np.asarray(X, np.float32).reshape(-1, TACTICS_SPEC.dim), np.asarray(y, np.float32), \
        np.asarray(groups), opts, dict(skipped), n_games


# ---------------------------------------------------------------------------- trainers
def train_engage(args) -> int:
    from blackboard.train import fit
    X, y, groups, skipped, n_games = engage_dataset([Path(p) for p in args.logs])
    print(f"games {n_games}  fights {len(X)}  won {y.mean() if len(y) else 0:.1%}  skipped {skipped}")
    if len(X) < 20:
        print("need at least 20 logged fights with features (play games with the Engagement slot)",
              file=sys.stderr)
        return 1
    base = float(((X[:, _e_names.index("lanchester_p")] > 0.5) == (y > 0.5)).mean())
    m, rep = fit(X, y, task="binary", hidden=args.hidden, labels=("won",), spec=ENGAGE_SPEC, groups=groups,
                 epochs=args.epochs, l2=args.l2, seed=args.seed, meta={"games": n_games, "lanchester_acc": base})
    print(f"lanchester accuracy {base:.3f}\n{rep}\nsaved {m.save(Path(args.out))}")
    return 0


def train_tactics(args) -> int:
    from blackboard.train import fit
    X, y, groups, opts, skipped, n_games = tactics_dataset([Path(p) for p in args.logs], args.horizon,
                                                            args.win_weight)
    print(f"games {n_games}  decisions {len(X)}  skipped {skipped}")
    for o in OPTIONS:
        sel = np.array([p == o for p in opts], bool)
        if sel.any():
            print(f"  {o:12s} n {int(sel.sum()):5d}  mean target {float(y[sel].mean()):+.3f}")
    if len(X) < 50:
        print("need at least 50 tactics decisions (play games with the Tactics slot)", file=sys.stderr)
        return 1
    m, rep = fit(X, y, task="regression", hidden=args.hidden, labels=("value",), spec=TACTICS_SPEC, groups=groups,
                 epochs=args.epochs, l2=args.l2, seed=args.seed,
                 meta={"games": n_games, "horizon_s": args.horizon, "win_weight": args.win_weight})
    print(f"{rep}\nsaved {m.save(Path(args.out))}")
    return 0
