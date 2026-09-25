"""Goliath - a Terran agent that techs to Goliaths and fights with them.

Run it against a visible game from the repo root:

    python run.py --bot goliath
    python run.py --bot goliath --speed 42 --race Terran

or directly (shim + StarCraft already running):

    cd python && .venv\\Scripts\\python -m bwbot.run goliath
"""
from __future__ import annotations

from bwbot import Actions, Observation, UnitType

from mybot.bot import MyBot
from mybot.state import State

from .policy import GoliathPolicy

GOLIATH = UnitType.Terran_Goliath
FACTORY = UnitType.Terran_Factory
ARMORY = UnitType.Terran_Armory


class Goliath(MyBot):
    """MyBot wired to GoliathPolicy, plus Goliath repairs and a Goliath-focused HUD."""

    def __init__(self) -> None:
        super().__init__(policy=GoliathPolicy())

    def on_frame(self, obs: Observation, act: Actions) -> None:
        super().on_frame(obs, act)
        if self.state is not None:
            self._goliath_hud(self.state, act)

    def on_workers_ready(self, s: State, act: Actions) -> None:
        self._repair_goliaths(s, act)

    def _repair_goliaths(self, s: State, act: Actions) -> None:
        goliaths = s.obs.my_completed(GOLIATH)
        if len(goliaths) == 0:
            return
        max_hp = int(s.game.unit_types["max_hit_points"][int(GOLIATH)])
        hurt = goliaths[goliaths["hit_points"] < max_hp * 3 // 4]
        if len(hurt) == 0:
            return
        target = hurt[int(hurt["hit_points"].argmin())]
        mx, my = s.main_tile[0] * 32, s.main_tile[1] * 32
        if (int(target["x"]) - mx) ** 2 + (int(target["y"]) - my) ** 2 > (25 * 32) ** 2:
            return
        scv = self.workers.claim_repair(s, int(target["x"]), int(target["y"]))
        if scv is not None:
            act.repair(scv, target)

    def _goliath_hud(self, s: State, act: Actions) -> None:
        act.draw_text_screen(
            10, 58,
            f"goliaths {s.count_completed(GOLIATH)}/{s.count(GOLIATH)}  "
            f"factories {s.count_completed(FACTORY)}  armory {s.count_completed(ARMORY)}  "
            f"attack={'yes' if getattr(self.policy, 'attacking', False) else 'no'}",
        )


BOT = Goliath

__all__ = ["Goliath", "GoliathPolicy", "BOT"]
