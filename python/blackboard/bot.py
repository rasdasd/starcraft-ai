"""`BlackboardBot`: a bwbot `Bot` that runs a list of components on a blackboard."""
from __future__ import annotations

import logging
from typing import Callable, Optional

from bwbot import Actions, Bot, ClientConfig, GameInfo, Observation

from . import hud
from .board import Blackboard
from .component import Component
from .recorder import Recorder
from .scheduler import Scheduler

log = logging.getLogger("blackboard.bot")


class BlackboardBot(Bot):
    """Components are rebuilt for every game by `factory` so no state leaks between games."""

    config = ClientConfig(frame_skip=4, local_speed=0, include_tiles=True)

    def __init__(self, factory: Callable[[], list[Component]], name: str = "blackboard",
                 time_budget_ms: float = 40.0, show_hud: bool = True, board_config: Optional[dict] = None,
                 strict: bool = False, recorder: Optional[Recorder] = None) -> None:
        self.factory = factory
        self.name = name
        self.time_budget_ms = time_budget_ms
        self.show_hud = show_hud
        self.board_config = dict(board_config or {})
        self.strict = strict
        self.recorder = recorder if recorder is not None else Recorder()
        self.bb: Optional[Blackboard] = None
        self.sched: Optional[Scheduler] = None

    def on_start(self, game: GameInfo) -> None:
        cfg = dict(self.board_config)
        cfg["complete_map_information"] = bool(getattr(self.config, "complete_map_information", False))
        self.bb = Blackboard(game, config=cfg)
        self.bb.recorder = self.recorder
        self.sched = Scheduler(self.factory(), time_budget_ms=self.time_budget_ms, strict=self.strict)
        self.sched.start(self.bb)
        slots = {self.sched._key(c): c.describe() for c in self.sched.components}
        enemies = game.enemies
        self.recorder.start({
            "bot": self.name, "map": game.map_name, "map_hash": game.map_hash, "seed": game.random_seed,
            "self_race": int(game.self_player.race), "enemy_race": int(enemies[0].race) if enemies else -1,
            "enemy_name": enemies[0].name if enemies else "", "slots": slots, "config": cfg,
        })
        log.info("%s start: map=%s game=%s slots=%s", self.name, game.map_name, self.recorder.game_id,
                 ", ".join(f"{k}={v.get('impl')}" for k, v in slots.items()))

    def on_frame(self, obs: Observation, act: Actions) -> None:
        bb, sched = self.bb, self.sched
        assert bb is not None and sched is not None
        bb.begin_decision(obs, act)
        sched.run(bb)
        if self.show_hud:
            hud.draw(bb, sched, act, title=self.name)

    def on_end(self, is_winner: bool) -> None:
        if self.bb is None or self.sched is None:
            return
        bb, sched = self.bb, self.sched
        sched.end(bb, is_winner)
        fallbacks = {k: s.replaced_by for k, s in sched.stats.items() if s.replaced_by}
        timing = {k: [round(s.avg_ms, 3), round(s.max_ms, 2), s.failures, s.deferred] for k, s in sched.stats.items()}
        score = {}
        if bb.obs is not None:
            me = bb.obs.me
            score = {"unit": me.unit_score, "kill": me.kill_score, "building": me.building_score,
                     "razing": me.razing_score}
        self.recorder.finish(is_winner, bb.frame, stats=bb.stats, fallbacks=fallbacks, timing=timing, score=score,
                             max_decision_ms=round(sched.max_decision_ms, 2))
        slow = sorted(sched.stats.items(), key=lambda kv: -kv[1].max_ms)[:4]
        log.info("%s end: %s decisions=%d max=%.1fms slowest=%s fallbacks=%s", self.name,
                 "WIN" if is_winner else "LOSS", bb.decision, sched.max_decision_ms,
                 ", ".join(f"{k}:{s.max_ms:.1f}" for k, s in slow), fallbacks or "-")
