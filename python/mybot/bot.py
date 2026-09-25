"""MyBot: glue between the game and the manager stack.

Per decision:
    InformationManager -> perceive -> Policy -> Production -> Buildings
    -> scout / repairs -> Workers -> CombatCommander -> HUD

Strategy stays a Policy so it can later be learned. Managers own the messy loop.
"""
from __future__ import annotations

import logging
from typing import Optional

from bwbot import Actions, Bot, ClientConfig, GameInfo, Observation, Race

from .buildings import BuildingManager
from .combat import CombatCommander
from .information import InformationManager
from .logger import GameLogger
from .macro import Placer
from .opponent import OPENING_NAMES, OpponentModel
from .policy import Intent, Policy, ScriptedPolicy
from .production import ProductionManager
from .scout import ScoutManager
from .state import State, perceive
from .tactics import from_intents
from .workers import WorkerManager

log = logging.getLogger("mybot")


class MyBot(Bot):
    config = ClientConfig(frame_skip=4, local_speed=0, include_tiles=True)
    apm_budget: Optional[float] = 400.0

    def __init__(self, policy: Optional[Policy] = None) -> None:
        self.policy: Policy = policy or ScriptedPolicy()
        self.info = InformationManager()
        self.state: Optional[State] = None
        self.workers = WorkerManager()
        self.buildings = BuildingManager()
        self.production = ProductionManager()
        self.placer = Placer()
        self.scout = ScoutManager()
        self.combat = CombatCommander()
        self.opponent = OpponentModel()
        self.logger = GameLogger()
        self._stance = "none"

    def on_start(self, game: GameInfo) -> None:
        self.info.on_start(game)
        self.state = None
        self.workers.reset()
        self.buildings.reset()
        self.production.reset()
        self.placer.reset()
        self.scout.on_start(game, self.info)
        self.opponent.on_start(game)
        self.logger.on_start(game.map_name, type(self.policy).__name__)
        self._stance = "none"
        log.info("start: map=%s me=%s race=%s start=%s enemy_start=%s policy=%s", game.map_name,
                 game.self_player.name, game.self_race.name, game.self_player.start_location,
                 self.info.enemy_start, type(self.policy).__name__)

    def on_frame(self, obs: Observation, act: Actions) -> None:
        self.info.update(obs, self.game)
        self.opponent.update(obs, self.game, self.info)
        mem = self.info.as_memory()
        mem.opponent = self.opponent.snapshot()
        s = self.state = perceive(obs, self.game, mem)
        intents = self.policy.decide(s)
        self.logger.record(s, intents)
        army = self.production.ingest(intents, self.buildings)
        self.production.update(s, act, self.buildings)
        self.buildings.update(s, act, self.workers, self.placer)
        self.scout.update(s, act, self.workers, self.info)
        self.on_workers_ready(s, act)
        self.workers.update(s, act)
        self._stance = self.combat.update(s, act, army, self.info)
        self._hud(s, act, intents)

    def on_workers_ready(self, s: State, act: Actions) -> None:
        """Hook after construction/scout claims, before mineral/gas assignment."""

    def on_end(self, is_winner: bool) -> None:
        frame = self.state.frame if self.state else 0
        snap = self.opponent.snapshot()
        log.info("end: %s at frame %s opp=%s open=%s", "WIN" if is_winner else "LOSS", frame,
                 snap.race, OPENING_NAMES[snap.opening] if snap.opening < len(OPENING_NAMES) else snap.opening)
        self.logger.finish(is_winner, frame, {
            "opp_race": snap.race, "opp_opening": snap.opening, "opp_proxy": snap.proxy,
        })
        self.policy.on_game_end(self.state, is_winner)

    def _hud(self, s: State, act: Actions, intents: list[Intent]) -> None:
        jobs = ",".join(s.game.type_name(t.unit_type) + ":" + t.status for t in self.buildings.tasks) or "-"
        queued = ",".join(s.game.type_name(t) for t in self.production.queue) or "-"
        scout = f"#{self.scout.worker_id}" if self.scout.worker_id else ("done" if self.scout.done else "wait")
        act.draw_text_screen(10, 10, f"{type(self.policy).__name__}  f{s.frame}  min {s.minerals}  gas {s.gas}  "
                                     f"supply {s.supply_used}/{s.supply_total}")
        act.draw_text_screen(10, 22, f"workers {len(s.workers)}  army {len(s.army)}  enemies {len(s.enemies)}  "
                                     f"fog {len(s.enemy_buildings)}  queue [{queued}]")
        nat = f"{s.natural_tile}" if s.natural_tile else "-"
        act.draw_text_screen(10, 34, f"builds [{jobs}]  combat {self._stance}  scout {scout}  "
                                     f"natural {nat}  bases {len(s.game.bases)}  "
                                     f"intents: " + ", ".join(type(i).__name__ for i in intents))
        opp = s.opponent
        opening = OPENING_NAMES[opp.opening] if 0 <= opp.opening < len(OPENING_NAMES) else "?"
        try:
            rname = Race(int(opp.race)).name
        except ValueError:
            rname = str(opp.race)
        act.draw_text_screen(10, 46, f"opp race={rname}  open={opening}  air={opp.air_units}  "
                                     f"proxy={int(opp.proxy)}  tactic={from_intents(intents, s).name.lower()}")
        for task in self.buildings.tasks:
            if task.worker_id is not None:
                u = s.obs.unit(task.worker_id)
                if u is not None:
                    act.draw_circle(int(u["x"]), int(u["y"]), 12)
            if task.tile is not None:
                act.draw_tile_box(task.tile[0], task.tile[1],
                                  int(s.game.unit_types["tile_width"][task.unit_type]),
                                  int(s.game.unit_types["tile_height"][task.unit_type]))
        for _, x, y in s.enemy_buildings:
            act.draw_box(x - 16, y - 16, x + 16, y + 16)
        if s.natural_tile is not None:
            act.draw_box(s.natural_tile[0] * 32, s.natural_tile[1] * 32,
                         s.natural_tile[0] * 32 + 128, s.natural_tile[1] * 32 + 96)
        if s.main_choke is not None:
            act.draw_circle(s.main_choke[0], s.main_choke[1], 24)
