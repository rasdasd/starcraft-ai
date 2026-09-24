"""Win-rate summaries over harness results (`runs/*/results.jsonl`).

    python -m adjutant.learn.report runs/sp20260923-101500 [more runs...] [--player adjutant@explore]

Tables: per player overall, per player vs opponent, per map, per opening template, and game-end
reasons. Rates come with 95% Wilson intervals; draws count as half a win.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


def load(paths: Iterable[str]) -> list[dict]:
    rows = []
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else sorted(p.rglob("results.jsonl"))
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class Tally:
    __slots__ = ("wins", "losses", "draws")

    def __init__(self) -> None:
        self.wins = self.losses = self.draws = 0

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.n if self.n else 0.0

    def add(self, outcome: float) -> None:
        if outcome > 0.5:
            self.wins += 1
        elif outcome < 0.5:
            self.losses += 1
        else:
            self.draws += 1

    def fmt(self) -> str:
        lo, hi = wilson(self.wins + 0.5 * self.draws, self.n)
        return f"{self.rate:6.1%} [{lo:5.1%}-{hi:5.1%}]  {self.wins}-{self.losses}-{self.draws}"


def perspectives(rows: list[dict], player: Optional[str] = None):
    """Yield (me, opponent, row, outcome 1/0/0.5) for each side of each judged game."""
    for r in rows:
        if r.get("reason") == "crash":
            continue
        for me, op in (("a", "b"), ("b", "a")):
            if player and r[me]["name"] != player:
                continue
            w = r.get("winner")
            yield r[me], r[op], r, (0.5 if w is None else float(w == me))


def summarize(rows: list[dict], player: Optional[str] = None) -> dict[str, dict]:
    tables: dict[str, dict] = {k: defaultdict(Tally) for k in ("player", "matchup", "map", "template")}
    reasons: dict[str, int] = defaultdict(int)
    for r in rows:
        reasons[r.get("reason", "?")] += 1
    for me, op, r, outcome in perspectives(rows, player):
        name = me["name"]
        tables["player"][name].add(outcome)
        tables["matchup"][f"{name} vs {op['name']}"].add(outcome)
        tables["map"][f"{name} on {Path(r['map']).stem}"].add(outcome)
        if me.get("template"):
            tables["template"][f"{name} / {me['template']}"].add(outcome)
    return {**{k: dict(v) for k, v in tables.items()}, "reasons": dict(reasons)}


def print_report(rows: list[dict], player: Optional[str] = None, out=sys.stdout) -> None:
    s = summarize(rows, player)
    print(f"\n{len(rows)} games; end reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(s["reasons"].items())),
          file=out)
    for key, title in (("player", "Player"), ("matchup", "Matchup"), ("map", "Map"), ("template", "Opening template")):
        table = s[key]
        if not table:
            continue
        print(f"\n{title}:", file=out)
        width = max(len(k) for k in table)
        for k, t in sorted(table.items(), key=lambda kv: (-kv[1].n, kv[0])):
            print(f"  {k:<{width}}  {t.fmt()}", file=out)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run directories or results.jsonl files")
    ap.add_argument("--player", help="only this player's perspective")
    args = ap.parse_args(argv)
    rows = load(args.paths)
    if not rows:
        print("no results found")
        return 1
    print_report(rows, args.player)
    return 0


if __name__ == "__main__":
    sys.exit(main())
