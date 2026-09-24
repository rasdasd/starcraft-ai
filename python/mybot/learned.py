"""LearnedPolicy: teacher opening + (model or opponent-aware) midgame.

The model only chooses a tactic and the next building. Production, construction,
workers, and the APM trim stay in the managers. With no `.npz` yet, opponent
reactions are the prior: bunker vs rush, goliaths vs air, hold until we have an army.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from bwbot import UnitType

from .model import LinearPolicy
from .opponent import OpeningGuess
from .policy import Build, Intent
from .state import State
from .tactics import Tactic, from_intents, replace_army

log = logging.getLogger("mybot.learned")

BUNKER = UnitType.Terran_Bunker
ARMORY = UnitType.Terran_Armory
FACTORY = UnitType.Terran_Factory
GOLIATH = UnitType.Terran_Goliath
MARINE = UnitType.Terran_Marine


def default_model_path() -> Optional[Path]:
    env = os.environ.get("BWBOT_MODEL")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path("bwapi-data") / "read" / "policy.npz",
        Path("bwapi-data") / "AI" / "policy.npz",
        Path("models") / "policy.npz",
        Path(__file__).resolve().parent.parent / "models" / "policy.npz",
    ])
    for p in candidates:
        if p.is_file():
            return p
    return None


class LearnedPolicy:
    """Wraps a scripted teacher. Swap the teacher for marine vs Goliath style."""

    def __init__(self, teacher=None, model_path: Optional[Path] = None) -> None:
        if teacher is None:
            from goliath.policy import GoliathPolicy
            teacher = GoliathPolicy()
        self.teacher = teacher
        self.model: Optional[LinearPolicy] = None
        path = Path(model_path) if model_path is not None else default_model_path()
        if path is not None:
            try:
                self.model = LinearPolicy.load(path)
                log.info("loaded policy %s", path)
            except Exception:
                log.exception("failed to load %s; using opponent prior", path)

    def reset(self) -> None:
        self.teacher.reset()

    def decide(self, s: State) -> list[Intent]:
        intents = self.teacher.decide(s)
        opening_done = getattr(self.teacher, "opening", None)
        if opening_done is not None and not opening_done.done:
            return intents
        if self.model is not None:
            return self._apply_model(s, intents)
        return self._apply_prior(s, intents)

    def _apply_model(self, s: State, intents: list[Intent]) -> list[Intent]:
        tactic, build = self.model.predict(s.as_features())
        out = replace_army(intents, tactic, s)
        if build and not any(isinstance(i, Build) and int(i.unit_type) == build for i in out):
            out.insert(0, Build(build))
        return out

    def _apply_prior(self, s: State, intents: list[Intent]) -> list[Intent]:
        """Hand prior used until there are enough logged games to fit a model."""
        opp = s.opponent
        opening = OpeningGuess(opp.opening) if opp is not None else OpeningGuess.UNKNOWN
        rush = opening in (OpeningGuess.RUSH, OpeningGuess.CHEESE) or (opp is not None and opp.proxy)
        air = opening is OpeningGuess.AIR or (opp is not None and opp.air_units > 0) or len(s.enemy_flyers) > 0

        out = list(intents)
        if rush and s.count(BUNKER) == 0:
            out.insert(0, Build(BUNKER))
        if air and s.count(ARMORY) == 0 and s.count_completed(FACTORY) > 0:
            out.insert(0, Build(ARMORY))

        tactic = from_intents(out, s)
        if rush and not s.under_attack:
            tactic = Tactic.HOLD
        elif s.under_attack:
            tactic = Tactic.DEFEND
        elif air and (s.count_completed(GOLIATH) >= 2 or s.count_completed(MARINE) >= 4):
            tactic = Tactic.HUNT_AIR
        elif tactic is Tactic.PUSH and len(s.army) < 6:
            tactic = Tactic.HOLD
        return replace_army(out, tactic, s)

    def on_game_end(self, s: Optional[State], won: bool) -> None:
        src = "model" if self.model is not None else "prior"
        log.info("game over: %s via %s", "WIN" if won else "LOSS", src)
        self.teacher.on_game_end(s, won)
