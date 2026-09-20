"""MyBot: the `bwbot.Bot` subclass the runner drives. Glue between game and policy.

Per decision (every `config.frame_skip` frames):
    1. state.perceive(obs)          -> State
    2. self.policy.decide(state)    -> [Intent, ...]
    3. execute each Intent          -> act.train / act.build / act.attack_move / ...
    4. standing orders that no policy should have to think about (idle workers mine)
    5. HUD

Execution is deliberately conservative: one build per structure type at a time, minerals reserved
for pending builds, no command spam (only idle units get new orders). Known gaps for other races:
no pylon-power check for Protoss placement, no creep check for Zerg.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from bwbot import Actions, Bot, ClientConfig, EventType, GameInfo, Observation, UnitType

from . import macro
from .policy import Attack, Build, Intent, Policy, Rally, ScriptedPolicy, Train
from .state import Memory, State, perceive

log = logging.getLogger("mybot")

BUILD_TIMEOUT_FRAMES = 24 * 15   # give up on a build order if construction hasn't started in 15 s


@dataclass
class _Budget:
    minerals: int
    gas: int
    supply: int                  # displayed supply left

    def can_afford(self, game: GameInfo, unit_type: int) -> bool:
        t = game.unit_types[int(unit_type)]
        return (self.minerals >= t["mineral_price"] and self.gas >= t["gas_price"]
                and self.supply >= t["supply_required"] // 2)

    def spend(self, game: GameInfo, unit_type: int) -> None:
        t = game.unit_types[int(unit_type)]
        self.minerals -= int(t["mineral_price"])
        self.gas -= int(t["gas_price"])
        self.supply -= int(t["supply_required"]) // 2


@dataclass
class _PendingBuild:
    frame: int                   # when the order was issued
    builder: int                 # worker unit id
    tile: tuple[int, int]        # top-left tile


class MyBot(Bot):
    # frame_skip=4: decide ~6x per game second; local_speed=0: run the game as fast as it goes
    # (use `python run.py --speed 42` to watch at human speed). Tiles (explored/visible map) are
    # needed for building placement.
    config = ClientConfig(frame_skip=4, local_speed=0, include_tiles=True)

    def __init__(self, policy: Optional[Policy] = None) -> None:
        self.policy: Policy = policy or ScriptedPolicy()
        self.mem = Memory()
        self.state: Optional[State] = None
        self.pending: dict[int, _PendingBuild] = {}     # structure type -> in-flight build order
        self.failed_tiles: set[tuple[int, int]] = set()  # placements that timed out this game

    # ------------------------------------------------------------------ lifecycle
    def on_start(self, game: GameInfo) -> None:
        self.mem = Memory()
        self.state = None
        self.pending = {}
        self.failed_tiles = set()
        me = game.self_player
        others = [tuple(s) for s in game.start_locations.tolist() if tuple(s) != tuple(me.start_location)]
        if others:
            self.mem.enemy_start = others[0]   # 2-player maps: the only other start; else a guess
        log.info("start: map=%s me=%s race=%s start=%s enemy_start=%s policy=%s", game.map_name, me.name,
                 game.self_race.name, me.start_location, self.mem.enemy_start, type(self.policy).__name__)

    def on_frame(self, obs: Observation, act: Actions) -> None:
        self._update_pending(obs)
        s = self.state = perceive(obs, self.game, self.mem)
        intents = self.policy.decide(s)

        budget = _Budget(s.minerals - self._reserved_minerals(), s.gas - self._reserved_gas(), s.supply_left)
        for intent in intents:
            self._execute(intent, s, act, budget)

        self._idle_workers_mine(s, act)
        self._hud(s, act, intents)

    def on_end(self, is_winner: bool) -> None:
        log.info("end: %s at frame %s", "WIN" if is_winner else "LOSS", self.state.frame if self.state else "?")
        self.policy.on_game_end(self.state, is_winner)

    # ------------------------------------------------------------------ intent execution
    def _execute(self, intent: Intent, s: State, act: Actions, budget: _Budget) -> None:
        if isinstance(intent, Train):
            self._train(intent.unit_type, s, act, budget)
        elif isinstance(intent, Build):
            self._build(intent.unit_type, s, act, budget)
        elif isinstance(intent, Attack):
            for u in s.obs.idle(s.army):
                act.attack_move(u, intent.x, intent.y)
        elif isinstance(intent, Rally):
            for u in s.obs.idle(s.army):
                if abs(int(u["x"]) - intent.x) + abs(int(u["y"]) - intent.y) > 160:
                    act.move(u, intent.x, intent.y)
        else:
            log.warning("unknown intent %r", intent)

    def _train(self, unit_type: int, s: State, act: Actions, budget: _Budget) -> None:
        if not budget.can_afford(self.game, unit_type):
            return
        producer_type = int(self.game.unit_types["what_builds"][int(unit_type)])
        producers = s.obs.my_completed(producer_type)
        if producer_type == UnitType.Zerg_Larva:
            if len(producers):
                act.morph(producers[0], unit_type)
                budget.spend(self.game, unit_type)
            return
        idle = producers[producers["train_queue_count"] == 0]
        if len(idle):
            act.train(idle[0], unit_type)
            budget.spend(self.game, unit_type)

    def _build(self, unit_type: int, s: State, act: Actions, budget: _Budget) -> None:
        if unit_type in self.pending or not budget.can_afford(self.game, unit_type) or len(s.workers) == 0:
            return
        tile = macro.find_build_tile(s.obs, unit_type, s.main_tile, skip=frozenset(self.failed_tiles))
        if tile is None:
            log.warning("no placement found for %s", self.game.type_name(unit_type))
            return
        builder = macro.pick_builder(s.obs, s.workers, tile)
        if builder is None:
            return
        act.build(builder, unit_type, tile[0], tile[1])
        budget.spend(self.game, unit_type)
        self.pending[int(unit_type)] = _PendingBuild(s.frame, int(builder["id"]), tile)
        log.info("f%d build %s at tile %s with worker #%d", s.frame, self.game.type_name(unit_type), tile,
                 int(builder["id"]))

    # ------------------------------------------------------------------ bookkeeping
    def _update_pending(self, obs: Observation) -> None:
        """A pending build is done once its structure appears (UnitCreate); on timeout, drop it and
        avoid that tile for the rest of the game."""
        for e in obs.iter_events(EventType.UnitCreate):
            u = obs.unit(e.unit)
            if u is not None and int(u["player"]) == obs.self_id:
                self.pending.pop(int(u["type"]), None)
        for t, pb in list(self.pending.items()):
            if obs.frame_count - pb.frame > BUILD_TIMEOUT_FRAMES:
                log.warning("f%d build %s at %s timed out; blacklisting tile", obs.frame_count,
                            self.game.type_name(t), pb.tile)
                self.failed_tiles.add(pb.tile)
                del self.pending[t]

    def _reserved_minerals(self) -> int:
        return sum(int(self.game.unit_types["mineral_price"][t]) for t in self.pending)

    def _reserved_gas(self) -> int:
        return sum(int(self.game.unit_types["gas_price"][t]) for t in self.pending)

    def _idle_workers_mine(self, s: State, act: Actions) -> None:
        builders = {pb.builder for pb in self.pending.values()}
        fields = s.obs.minerals_fields
        for w in s.obs.idle(s.workers):
            if int(w["id"]) in builders:
                continue
            patch = s.obs.nearest(fields, w["x"], w["y"])
            if patch is not None:
                act.gather(w, patch)

    def _hud(self, s: State, act: Actions, intents: list[Intent]) -> None:
        act.draw_text_screen(10, 10, f"{type(self.policy).__name__}  f{s.frame}  min {s.minerals}  gas {s.gas}  "
                                     f"supply {s.supply_used}/{s.supply_total}")
        act.draw_text_screen(10, 22, f"workers {len(s.workers)}  army {len(s.army)}  enemies {len(s.enemies)}  "
                                     f"pending {[self.game.type_name(t) for t in self.pending]}")
        act.draw_text_screen(10, 34, "intents: " + ", ".join(type(i).__name__ for i in intents))
        for t, pb in self.pending.items():
            u = s.obs.unit(pb.builder)
            if u is not None:
                act.draw_circle(int(u["x"]), int(u["y"]), 12)
            act.draw_tile_box(pb.tile[0], pb.tile[1], int(self.game.unit_types["tile_width"][t]),
                              int(self.game.unit_types["tile_height"][t]))
