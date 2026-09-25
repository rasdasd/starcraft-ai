"""Reinforcement-learned micro (experimental): fitted Q iteration over logged unit transitions.

`RLMicro` picks one of `ACTIONS` for every combat unit that has enemies in reach, once per
`decide_frames`, and logs `micro/step` transitions for a sample of units:
    x  = unit state features (MICRO_STATE, without the action)
    a  = action index
    r  = local damage trade until the unit's next decision: hit points + shields the enemies that were
         near it lost (all of it if they died) minus what our nearby units lost, weighted by unit
         value per hit point, / 100
    x2 = next state features, or null when the unit died or left the fight (terminal)
Training is offline and off-policy: Q(s, a) is an MLP over state + action one-hot, refit
`iters` times on r + gamma * max_a' Q(s', a'). Collect with epsilon exploration, train, collect
again with the new model, repeat.

    python -m adjutant.learn.train micro --logs runs/a runs/b --out models/micro.npz
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from blackboard.models import FeatureSpec
from blackboard.recorder import iter_games

ACTIONS = ("focus", "nearest", "kite", "back", "stay")
ORDERS = ("attack", "defend", "hold", "other")
MICRO_VERSION = 1
STATE_NAMES = ("hp", "shields", "cooldown", "ranged", "range", "flyer", "value", "melee_threats",
               "threats", "n_enemy", "n_own", "near_dist", "near_range", "near_melee", "near_hp",
               "own_hp_local", "enemy_hp_local", "hp_ratio", "target_dist", "target_hp",
               "dest_dist") + tuple(f"order_{o}" for o in ORDERS)
MICRO_STATE = len(STATE_NAMES)
MICRO_SPEC = FeatureSpec("micro", MICRO_VERSION, STATE_NAMES + tuple(f"act_{a}" for a in ACTIONS))


def sa(X: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """State rows + action one-hots -> Q model inputs."""
    X = np.asarray(X, np.float32).reshape(-1, MICRO_STATE)
    return np.concatenate([X, np.eye(len(ACTIONS), dtype=np.float32)[np.asarray(actions, int)]], axis=1)


def all_actions(x: np.ndarray) -> np.ndarray:
    """One state -> the Q inputs for every action (len(ACTIONS) rows)."""
    X = np.repeat(np.asarray(x, np.float32).reshape(1, -1), len(ACTIONS), axis=0)
    return sa(X, np.arange(len(ACTIONS)))


def micro_dataset(dirs: Sequence[Path]):
    X, A, R, X2, done, groups = [], [], [], [], [], []
    skipped: dict[str, int] = defaultdict(int)
    n_games = 0
    for g in iter_games(*dirs, finished_only=False):
        rows = [r for r in g["rows"] if r.get("slot") == "micro" and r.get("k") == "step"]
        if not rows:
            continue
        n_games += 1
        gid = g["header"].get("game_id", g["path"]) + str(g["header"].get("side", ""))
        for r in rows:
            x, x2 = r.get("x"), r.get("x2")
            if r.get("fh") != MICRO_SPEC.hash or x is None or len(x) != MICRO_STATE or \
                    (x2 is not None and len(x2) != MICRO_STATE):
                skipped["width"] += 1
                continue
            X.append(x)
            A.append(int(r["a"]))
            R.append(float(r["r"]))
            X2.append(x2 if x2 is not None else [0.0] * MICRO_STATE)
            done.append(x2 is None)
            groups.append(gid)
    f = np.float32
    return (np.asarray(X, f).reshape(-1, MICRO_STATE), np.asarray(A, int), np.asarray(R, f),
            np.asarray(X2, f).reshape(-1, MICRO_STATE), np.asarray(done, bool), np.asarray(groups),
            dict(skipped), n_games)


def fitted_q(X, A, R, X2, done, groups=None, gamma: float = 0.8, iters: int = 4, hidden=(32,), epochs: int = 60,
             l2: float = 1e-4, seed: int = 0, meta: dict | None = None, log=print):
    from blackboard.train import fit
    Xa = sa(X, A)
    k = len(ACTIONS)
    X2all = np.concatenate([sa(X2, np.full(len(X2), a)) for a in range(k)], axis=0) if len(X2) else Xa
    y = R.copy()
    model = rep = None
    for it in range(iters):
        model, rep = fit(Xa, y, task="regression", hidden=hidden, labels=("q",), spec=MICRO_SPEC, groups=groups,
                         epochs=epochs, l2=l2, seed=seed + it,
                         meta=dict(meta or {}, gamma=gamma, iters=iters, iteration=it))
        q2 = np.asarray(model.predict(X2all), np.float32).reshape(k, -1).max(axis=0)
        y = R + gamma * np.where(done, 0.0, q2)
        log(f"iter {it}: {rep}  mean target {float(y.mean()):+.3f}")
    return model, rep


def train_micro(args) -> int:
    X, A, R, X2, done, groups, skipped, n_games = micro_dataset([Path(p) for p in args.logs])
    print(f"games {n_games}  transitions {len(X)}  terminal {float(done.mean()) if len(done) else 0:.1%}  "
          f"skipped {skipped}")
    for i, a in enumerate(ACTIONS):
        sel = A == i
        if sel.any():
            print(f"  {a:8s} n {int(sel.sum()):6d}  mean reward {float(R[sel].mean()):+.3f}")
    if len(X) < 200:
        print("need at least 200 micro transitions (play games with the RLMicro slot)", file=sys.stderr)
        return 1
    model, rep = fitted_q(X, A, R, X2, done, groups, gamma=args.gamma, iters=args.iters, hidden=args.hidden,
                          epochs=args.epochs, l2=args.l2, seed=args.seed, meta={"games": n_games})
    print(f"saved {model.save(Path(args.out))}")
    return 0
