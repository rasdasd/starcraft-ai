"""Slot-tagged JSONL game recorder.

One file per game: a header (game id, seed, map, profile, slot implementations, feature versions),
then rows `{"t": "rec", "slot": ..., "k": kind, "f": frame, ...}` written by components, then a
footer with the result. Trainers join rows with the footer (and, in self-play, with the opponent's
truth rows) by game id.

Location: `BWBOT_LOG_DIR`, else `bwapi-data/write/adjutant-logs` when the tournament write folder
exists, else `./logs/adjutant`. `BWBOT_LOG=0` disables it. `BWBOT_GAME_ID` pins the game id
(the self-play harness sets it so both players' logs share it).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional, TextIO

import numpy as np

log = logging.getLogger("blackboard.recorder")

FORMAT_VERSION = 1


def default_log_dir() -> Path:
    env = os.environ.get("BWBOT_LOG_DIR")
    if env:
        return Path(env)
    if Path("bwapi-data/write").is_dir():
        return Path("bwapi-data") / "write" / "adjutant-logs"
    return Path("logs") / "adjutant"


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return round(float(o), 5)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


class Recorder:
    def __init__(self, directory: Optional[Path] = None, enabled: Optional[bool] = None) -> None:
        self.directory = Path(directory) if directory is not None else default_log_dir()
        self.enabled = (os.environ.get("BWBOT_LOG", "1") != "0") if enabled is None else enabled
        self.game_id = ""
        self.path: Optional[Path] = None
        self.rows = 0
        self._fh: Optional[TextIO] = None
        self._last: dict[tuple[str, str], int] = {}

    @property
    def active(self) -> bool:
        return self._fh is not None

    def start(self, header: dict) -> None:
        self.close()
        self.rows = 0
        self._last.clear()
        self.game_id = os.environ.get("BWBOT_GAME_ID") or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        side = os.environ.get("BWBOT_SIDE", "")
        self.path = self.directory / f"{self.game_id}{'-' + side if side else ''}.jsonl"
        self._fh = self.path.open("w", encoding="utf-8")
        self._write({"t": "header", "v": FORMAT_VERSION, "game_id": self.game_id, "side": side,
                     "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **header})

    def record(self, slot: str, kind: str, frame: int, every: int = 0, **data: Any) -> bool:
        """Write a row; with `every` > 0, at most one row per (slot, kind) per `every` frames."""
        if self._fh is None:
            return False
        last = self._last.get((slot, kind))
        if every > 0 and last is not None and frame - last < every:
            return False
        self._last[(slot, kind)] = frame
        self._write({"t": "rec", "slot": slot, "k": kind, "f": int(frame), **data})
        self.rows += 1
        return True

    def finish(self, won: bool, frame: int, **extra: Any) -> None:
        if self._fh is None:
            return
        self._write({"t": "end", "won": bool(won), "f": int(frame), "rows": self.rows, **extra})
        log.info("recorded %d rows -> %s (%s)", self.rows, self.path, "WIN" if won else "LOSS")
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _write(self, row: dict) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(row, separators=(",", ":"), default=_default) + "\n")
        self._fh.flush()


# ---------------------------------------------------------------------------- reading
def read_game(path: Path) -> dict:
    """{"header": {...}, "rows": [...], "end": {...} or None} for one log file."""
    header: dict = {}
    end: Optional[dict] = None
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # truncated last line of a crashed game
            t = r.get("t")
            if t == "header":
                header = r
            elif t == "end":
                end = r
            elif t == "rec":
                rows.append(r)
    return {"header": header, "rows": rows, "end": end, "path": str(path)}


def iter_games(*dirs: Path, finished_only: bool = True):
    for d in dirs:
        d = Path(d)
        files = [d] if d.is_file() else sorted(d.rglob("*.jsonl"))
        for p in files:
            g = read_game(p)
            if g["header"] and (g["end"] is not None or not finished_only):
                yield g
