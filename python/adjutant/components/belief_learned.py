"""Learned belief and the truth logger that trains it.

`BeliefLog` (REPORT; may read `truth`) writes `belief/sample` rows in games with complete map
information. `LearnedBelief` is ScriptedBelief plus whichever of belief_{counts,tech,bases,opponent}.npz
load: predictions go to `belief.predicted`; counts the model is confident about raise the
estimate for unseen units (never below what was seen), and tech with p >= `tech_threshold` is
added to `belief.tech` / `belief.inferred`. With no model it is exactly ScriptedBelief.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from blackboard import Blackboard, Component, Phase
from blackboard.models import Model, model_search_dirs, try_load
from blackboard.profile import register

from ..learn.belief import BELIEF_SPEC, TECH, UNITS, belief_features, truth_targets
from .belief import ScriptedBelief

log = logging.getLogger("adjutant.belief")

HEADS = ("counts", "tech", "bases", "opponent")


@register("BeliefLog")
class BeliefLog(Component):
    phase = Phase.REPORT
    reads = ("world", "belief", "truth")
    period_frames = 24 * 15

    def tick(self, bb: Blackboard) -> None:
        if not bb.truth.enabled:
            return
        yc, yt, yb = truth_targets(bb)
        bb.record("belief", "sample", x=[round(float(v), 4) for v in belief_features(bb)], yc=yc, yt=yt, yb=yb)


@register("LearnedBelief")
class LearnedBelief(ScriptedBelief):
    def __init__(self, models: str = "", every_frames: int = 24, count_weight: float = 1.0,
                 tech_threshold: float = 0.7, temperature: float = 1.0) -> None:
        super().__init__(temperature=temperature)
        self.models_dir = models
        self.every = every_frames
        self.count_weight = count_weight
        self.tech_threshold = tech_threshold
        self.heads: dict[str, Model] = {}
        self.messages: dict[str, str] = {}

    def on_start(self, bb: Blackboard) -> None:
        super().on_start(bb)
        self.heads = {}
        search = ([self.models_dir] if self.models_dir else []) + [str(d) for d in model_search_dirs()]
        for h in HEADS:
            m, msg = try_load(f"belief_{h}.npz", BELIEF_SPEC, search=search)
            self.messages[h] = msg
            if m is not None:
                self.heads[h] = m
        log.info("LearnedBelief heads: %s", {h: self.messages[h] for h in HEADS})
        self._last = -10 ** 9
        self._pred: dict = {}

    def tick(self, bb: Blackboard) -> None:
        super().tick(bb)
        if not self.heads:
            return
        if bb.frame - self._last >= self.every:
            self._last = bb.frame
            self._pred = self.predict(belief_features(bb))
        self._apply(bb, self._pred)

    def predict(self, x: np.ndarray) -> dict:
        out: dict = {}
        if "counts" in self.heads:
            y = np.atleast_1d(self.heads["counts"].predict(x))
            out["counts"] = {t: float(max(0.0, np.expm1(v))) for t, v in zip(UNITS, y)}
        if "tech" in self.heads:
            p = np.atleast_1d(self.heads["tech"].predict(x))
            out["tech"] = {t: float(v) for t, v in zip(TECH, p)}
        if "bases" in self.heads:
            out["bases"] = float(np.atleast_1d(self.heads["bases"].predict(x))[0])
        if "opponent" in self.heads:
            m = self.heads["opponent"]
            p = np.atleast_1d(m.predict(x))
            out["opponent"] = {lab: float(v) for lab, v in zip(m.labels, p)}
        return out

    def _apply(self, bb: Blackboard, pred: dict) -> None:
        b = bb.belief
        b.predicted = dict(pred)
        for t, n in pred.get("counts", {}).items():
            seen = b.counts.get(t, 0.0)
            est = seen + self.count_weight * max(0.0, n - seen)
            if est >= 0.5 and est > seen:
                b.counts[t] = round(est, 2)
        added = {t for t, p in pred.get("tech", {}).items() if p >= self.tech_threshold and t not in b.tech}
        if added:
            b.tech = set(b.tech) | added
            b.inferred = set(b.inferred) | added

    def describe(self) -> dict:
        return {"impl": self.name, "heads": sorted(self.heads), "messages": self.messages}
