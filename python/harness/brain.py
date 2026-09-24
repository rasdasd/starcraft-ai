"""Run any bot like `python -m bwbot.run`, and write a result file when the game ends.

    BWBOT_RESULT=/tmp/a.json python -m harness.brain adjutant --port 9100 --games 1 --no-gui --max-frames 28800

The result (win flag, end frame, our score) lets the harness judge games that hit the frame limit
by score, for any bot, without touching the bot's code.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import bwbot.run as bwrun


def score(obs, game) -> dict:
    """BWAPI's in-game score when the engine keeps one; OpenBW leaves it at 0 until the end, so
    otherwise a material score: resources gathered + value of what we still own + 2x value killed."""
    me = obs.me
    bw = {"unit": me.unit_score, "kill": me.kill_score, "building": me.building_score, "razing": me.razing_score}
    if any(bw.values()):
        return bw
    ut = game.unit_types
    value = (ut["mineral_price"] + ut["gas_price"]).astype(np.int64)
    n = min(len(value), len(me.all_unit_count))
    owned = int((me.all_unit_count[:n] * value[:n]).sum()) if n else 0
    k = min(len(value), len(me.killed_unit_count))
    killed = int((me.killed_unit_count[:k] * value[:k]).sum()) if k else 0
    return {"gathered": me.gathered_minerals + me.gathered_gas, "owned": owned, "killed": 2 * killed}


def enemy_score(obs) -> int:
    """Sum of the enemies' BWAPI scores (0 under OpenBW). Lets a one-sided harness (published
    opponent, no result file of its own) judge frame-limit games."""
    total = 0
    for p in obs.game.enemies:
        ps = obs.players.get(p.id)
        if ps is not None:
            total += ps.unit_score + ps.kill_score + ps.building_score + ps.razing_score
    return total


def _wrap(bot):
    state = {"frame": 0, "score": {}, "t0": time.time(), "decisions": 0, "max_ms": 0.0}
    on_frame, on_end = bot.on_frame, bot.on_end

    def frame(obs, act):
        t = time.perf_counter()
        on_frame(obs, act)
        state["max_ms"] = max(state["max_ms"], (time.perf_counter() - t) * 1000)
        state["decisions"] += 1
        state["frame"] = obs.frame_count
        state["obs"] = obs

    def end(won):
        try:
            on_end(won)
        finally:
            last = state.pop("obs", None)
            if last is not None:
                state["score"] = score(last, bot.game)
                state["enemy_total"] = enemy_score(last)
            path = os.environ.get("BWBOT_RESULT")
            if path:
                sc = state["score"]
                row = {"won": bool(won), "frame": state["frame"], "score": sc, "total": sum(sc.values()),
                       "enemy_total": state.get("enemy_total", 0), "decisions": state["decisions"], "max_ms": round(state["max_ms"], 2),
                       "wall_s": round(time.time() - state["t0"], 1), "bot": getattr(bot, "name", type(bot).__name__)}
                rec = getattr(bot, "recorder", None)
                if rec is not None and getattr(rec, "game_id", ""):
                    row["log"] = str(rec.path) if rec.path else None
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(row), encoding="utf-8")

    bot.on_frame, bot.on_end = frame, end
    return bot


def main(argv=None) -> int:
    orig = bwrun.load_bot
    bwrun.load_bot = lambda spec: _wrap(orig(spec))
    return bwrun.main(argv)


if __name__ == "__main__":
    sys.exit(main())
