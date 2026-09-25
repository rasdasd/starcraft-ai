"""Adjutant: a StarCraft bot on the blackboard framework, with swappable scripted/learned slots.

    python run.py --bot adjutant                 # default profile
    python run.py --bot adjutant:Parity          # goliath-equivalent adapters
    BWBOT_PROFILE_FILE=overrides.json python run.py --bot adjutant

Profiles live in `adjutant.profiles`; each subclass below just names one.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from typing import Optional

from blackboard import BlackboardBot
from blackboard.profile import apply_env_override, build, resolve
from bwbot import ClientConfig, Race

from .profiles import BUILTINS

log = logging.getLogger("adjutant")


class Adjutant(BlackboardBot):
    profile: str = "adjutant"
    config = ClientConfig(frame_skip=4, local_speed=0, include_tiles=True)
    apm_budget: Optional[float] = 400.0

    def __init__(self, profile: Optional[str] = None, **overrides) -> None:
        name = profile or os.environ.get("BWBOT_PROFILE") or self.profile
        self._overrides = overrides
        spec = self.spec = self._resolve(name)
        cfg = spec.get("config", {})
        super().__init__(factory=lambda: build(self.spec), name=f"adjutant:{spec.get('name', name)}",
                         time_budget_ms=float(cfg.get("time_budget_ms", 40)), board_config=cfg,
                         show_hud=bool(cfg.get("hud", True)))
        if cfg.get("complete_map_information"):
            self.config = dataclasses.replace(self.config, complete_map_information=True)
        if "apm_budget" in cfg:
            self.apm_budget = cfg["apm_budget"] or None

    def _resolve(self, name: str) -> dict:
        spec = apply_env_override(resolve(name, BUILTINS))
        if self._overrides:
            from blackboard.profile import deep_merge
            spec = deep_merge(spec, self._overrides)
        return spec

    def on_start(self, game) -> None:
        """`race_profiles` ({race name: profile}) swaps in another profile for our race."""
        other = self.spec.get("race_profiles", {}).get(Race(int(game.self_race)).name)
        if other:
            log.info("race %s: profile %s", Race(int(game.self_race)).name, other)
            self.spec = self._resolve(other)
            self.name = f"adjutant:{self.spec.get('name', other)}"
        super().on_start(game)


class Parity(Adjutant):
    profile = "parity"


BOT = Adjutant

__all__ = ["Adjutant", "BOT", "Parity"]
