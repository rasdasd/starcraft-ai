"""Build training arrays from recorder logs."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from .recorder import iter_games


def rows(dirs: Iterable[Path], slot: str, kind: str, finished_only: bool = True) -> Iterator[tuple[dict, dict]]:
    """(game, row) for every recorded row of `slot`/`kind`."""
    for g in iter_games(*dirs, finished_only=finished_only):
        for r in g["rows"]:
            if r.get("slot") == slot and r.get("k") == kind:
                yield g, r


def outcome(game: dict) -> Optional[bool]:
    end = game.get("end")
    return None if end is None else bool(end.get("won"))


def matrix(pairs: Iterable[tuple[dict, dict]], x_key: str, label: Callable[[dict, dict], Optional[float]],
           feature_hash: Optional[str] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, groups). Rows whose label is None or whose feature hash differs are skipped."""
    xs, ys, gs = [], [], []
    for g, r in pairs:
        if feature_hash is not None and r.get("fh") not in (None, feature_hash):
            continue
        y = label(g, r)
        if y is None or x_key not in r:
            continue
        xs.append(r[x_key])
        ys.append(y)
        gs.append(g["header"].get("game_id", g["path"]))
    if not xs:
        return np.zeros((0, 0), np.float32), np.zeros(0, np.float32), np.zeros(0, object)
    return np.asarray(xs, np.float32), np.asarray(ys), np.asarray(gs, object)


def pair_sides(dirs: Iterable[Path]) -> dict[str, dict[str, dict]]:
    """game_id -> {side: game} for self-play logs (both players share BWBOT_GAME_ID)."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for g in iter_games(*dirs, finished_only=False):
        h = g["header"]
        out[h.get("game_id", g["path"])][h.get("side") or g["path"]] = g
    return dict(out)
