"""Per-game JSONL of (features, tactic, build) plus a win/loss footer.

Logs land in `BWBOT_LOG_DIR`, else `bwapi-data/write/bwbot-logs` (tournament write
folder), else `./logs`. Sampling keeps files small: every `every` decisions, and
always when the tactic changes.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, TextIO

from .features import encode
from .model import encode_build
from .policy import Build, Intent
from .state import State
from .tactics import from_intents

log = logging.getLogger("mybot.logger")


def default_log_dir() -> Path:
    env = os.environ.get("BWBOT_LOG_DIR")
    if env:
        return Path(env)
    write = Path("bwapi-data") / "write" / "bwbot-logs"
    if Path("bwapi-data/write").is_dir():
        return write
    return Path("logs")


class GameLogger:
    def __init__(self, directory: Optional[Path] = None, every: int = 6) -> None:
        self.directory = Path(directory) if directory is not None else default_log_dir()
        self.every = max(1, every)
        self._fh: Optional[TextIO] = None
        self._n = 0
        self._last_tactic: Optional[int] = None
        self._path: Optional[Path] = None
        self.enabled = os.environ.get("BWBOT_LOG", "1") != "0"

    def on_start(self, map_name: str, policy: str) -> None:
        self.close()
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self._path = self.directory / f"{stamp}-{policy}.jsonl"
        self._fh = self._path.open("w", encoding="utf-8")
        self._n = 0
        self._last_tactic = None
        self._write({"event": "start", "map": map_name, "policy": policy, "t": stamp})

    def record(self, s: State, intents: list[Intent]) -> None:
        if self._fh is None:
            return
        tactic = int(from_intents(intents, s))
        self._n += 1
        if self._n % self.every != 0 and tactic == self._last_tactic:
            return
        self._last_tactic = tactic
        build = next((int(i.unit_type) for i in intents if isinstance(i, Build)), None)
        self._write({
            "event": "dec",
            "frame": s.frame,
            "x": encode(s).tolist(),
            "tactic": tactic,
            "build": encode_build(build),
        })

    def finish(self, won: bool, frame: int, extra: Optional[dict] = None) -> None:
        if self._fh is None:
            return
        row = {"event": "end", "won": bool(won), "frame": frame, "decisions": self._n}
        if extra:
            row.update(extra)
        self._write(row)
        log.info("logged %d decisions -> %s (%s)", self._n, self._path, "WIN" if won else "LOSS")
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _write(self, row: dict) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._fh.flush()


def load_decisions(directory: Path) -> tuple[list[list[float]], list[int], list[int], int, int]:
    """Return (X, y_tactic, y_build, wins, games) from a folder of jsonl logs."""
    xs, yt, yb = [], [], []
    wins = games = 0
    for path in sorted(Path(directory).glob("*.jsonl")):
        games += 1
        won = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "dec":
                xs.append(row["x"])
                yt.append(int(row["tactic"]))
                yb.append(int(row["build"]))
            elif row.get("event") == "end":
                won = bool(row.get("won"))
        wins += int(won)
    return xs, yt, yb, wins, games
