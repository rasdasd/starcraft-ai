"""Strategy slot: pick a template (or blend), publish Goal / Posture / opening progress.

`TemplateStrategy` does the bookkeeping; subclasses only implement `choose` (which template, or a
weighting over templates). The opening always comes from the top template; once it has finished,
the goal can be a blend.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

from blackboard import Blackboard, Component, Phase
from blackboard.profile import register
from mybot.opening import Opening

from .. import compat
from ..strategies import Template, blend_goals, get

log = logging.getLogger("adjutant.strategy")

Choice = Union[str, dict]          # template name, or {name: weight}


class TemplateStrategy(Component):
    phase = Phase.DECIDE
    reads = ("world", "meta", "belief", "threats")
    writes = ("strategy",)
    order = 10
    min_switch_frames = 24 * 30        # hysteresis: at most one switch per 30 s

    def __init__(self) -> None:
        self.current: Optional[Template] = None
        self.weights: dict[str, float] = {}
        self._opening: Optional[Opening] = None

    def on_start(self, bb: Blackboard) -> None:
        self.current = None
        self.weights = {}
        self._opening = None

    def choose(self, bb: Blackboard) -> Optional[Choice]:
        """Return a template name / weights to (re)decide, or None to keep the current choice."""
        raise NotImplementedError

    def tick(self, bb: Blackboard) -> None:
        choice = self.choose(bb)
        if choice is not None:
            weights = {choice: 1.0} if isinstance(choice, str) else {k: float(v) for k, v in choice.items() if v > 0}
            top = max(weights, key=weights.get)
            can_switch = self.current is None or bb.frame - bb.strategy.switched_frame >= self.min_switch_frames
            if self.current is None or (top != self.current.name and can_switch):
                self._switch(bb, top)
            if self.current is not None and (self.current.name == top or not can_switch):
                self.weights = weights
        self._apply(bb)

    def _switch(self, bb: Blackboard, name: str) -> None:
        prev = self.current.name if self.current else None
        self.current = get(name)
        st = bb.strategy
        st.template = name
        st.switched_frame = bb.frame
        st.history.append((bb.frame, name))
        if self._opening is None or not self._opening.done:
            self._opening = Opening(self.current.opening)
        bb.raise_event("strategy_changed")
        bb.say(f"strategy {prev} -> {name}")
        bb.record("strategy", "switch", frm=prev, to=name)
        log.info("f%d strategy %s -> %s", bb.frame, prev, name)

    def _apply(self, bb: Blackboard) -> None:
        t = self.current
        if t is None:
            return
        st = bb.strategy
        others = {k: w for k, w in self.weights.items() if k != t.name and w > 0}
        if others and self._opening is not None and self._opening.done:
            names = [t.name, *others]
            ws = [self.weights.get(t.name, 1.0), *others.values()]
            st.goal = blend_goals([get(n).goal(bb) for n in names], ws)
        else:
            st.goal = t.goal(bb)
        st.posture = t.posture(bb, st.posture)
        thr = bb.threats
        if thr.posture_override and bb.frame < thr.override_until:
            st.posture.stance = thr.posture_override
        st.opening = t.opening_steps()
        if self._opening is not None:
            st.opening_next = self._opening.next_build(compat.state(bb))
            st.opening_index = self._opening.i
            st.opening_done = self._opening.done
        st.values = dict(self.weights)

    def describe(self) -> dict:
        return {"impl": self.name}


@register("ScriptedStrategy")
class ScriptedStrategy(TemplateStrategy):
    """Always the same template."""

    def __init__(self, template: str = "goliath_1fact") -> None:
        super().__init__()
        self.template = template

    def choose(self, bb: Blackboard) -> Optional[Choice]:
        return self.template if self.current is None else None

    def describe(self) -> dict:
        return {"impl": self.name, "template": self.template}
