"""Strategy slot: pick a template (or blend), publish Goal / Posture / opening progress.

`TemplateStrategy` does the bookkeeping; subclasses only implement `choose` (which template, or a
weighting over templates). The opening always comes from the top template; once it has finished,
the goal can be a blend.

Every implementation logs a `strategy/ctx` row (context features + active template) every 30 s
and on each switch; `python -m adjutant.learn.train strategy` fits the win model from those rows.

  ScriptedStrategy  one fixed template
  RuleSelector      hand-written rules over meta / belief / threats
  Explore           random template at start, random switches at checkpoints (data collection)
  LearnedStrategy   per-template win probability; `select` the best or `blend` goals by it
"""
from __future__ import annotations

import logging
import os
import random
from typing import Optional, Sequence, Union

import numpy as np

from blackboard import Blackboard, Component, Phase
from blackboard.models import FeatureMismatch, Model, load_model, model_search_dirs
from blackboard.profile import register
from bwbot import Race
from ..strategies.opening import Opening

from ..strategies import TEMPLATES, Template, blend_goals, get
from ..units import AIR_TECH

log = logging.getLogger("adjutant.strategy")

Choice = Union[str, dict]          # template name, or {name: weight}
KNOWN_RACES = (int(Race.Zerg), int(Race.Terran), int(Race.Protoss))
REDECIDE_EVENTS = ("opening_changed", "enemy_race", "enemy_base_found", "under_attack", "enemy_seen")


def race_templates(race: int, names: Optional[Sequence[str]] = None) -> list[str]:
    pool = list(names) if names else sorted(TEMPLATES)
    return [n for n in pool if n in TEMPLATES and TEMPLATES[n].race == race]


def enemy_race(bb: Blackboard) -> int:
    r = bb.meta.enemy_race
    if r in KNOWN_RACES:
        return int(r)
    return int(bb.belief.enemy_race)


class TemplateStrategy(Component):
    phase = Phase.DECIDE
    reads = ("world", "meta", "belief", "threats", "macro")
    writes = ("strategy",)
    order = 10
    min_switch_frames = 24 * 30        # hysteresis: at most one switch per 30 s
    record_every = 24 * 30

    def __init__(self) -> None:
        self.current: Optional[Template] = None
        self.weights: dict[str, float] = {}
        self.values: Optional[dict[str, float]] = None
        self._opening: Optional[Opening] = None
        self._start: Optional[np.ndarray] = None     # own unit counts at the first decision

    def on_start(self, bb: Blackboard) -> None:
        self.current = None
        self.weights = {}
        self.values = None
        self._opening = None
        self._start = None

    def context(self, bb: Blackboard) -> np.ndarray:
        from ..learn.features import strategy_context
        return strategy_context(bb)

    def _record(self, bb: Blackboard, every: int, reason: str = "") -> None:
        if self.current is None or not bb.recorder.due("strategy", "ctx", bb.frame, every):
            return
        from ..learn.features import BUILD_SPEC, blend_features
        bf = blend_features(TEMPLATES, self.weights or {self.current.name: 1.0})
        bb.record("strategy", "ctx", x=[round(float(v), 4) for v in self.context(bb)],
                  b=[round(float(v), 4) for v in bf], bh=BUILD_SPEC.hash,
                  template=self.current.name, weights=self.weights, reason=reason)

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
        self._record(bb, self.record_every, "periodic")

    def _switch(self, bb: Blackboard, name: str) -> None:
        prev = self.current.name if self.current else None
        self.current = get(name)
        st = bb.strategy
        st.template = name
        st.switched_frame = bb.frame
        st.history.append((bb.frame, name))
        if self._start is None:
            self._start = bb.world.counts.copy()
        if self._opening is None or not self._opening.done:
            self._opening = Opening(self.current.opening, base=self._start)
        bb.raise_event("strategy_changed")
        bb.say(f"strategy {prev} -> {name}")
        bb.record("strategy", "switch", frm=prev, to=name)
        log.info("f%d strategy %s -> %s", bb.frame, prev, name)
        self.weights = {name: 1.0}
        self._record(bb, 0, "switch")

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
            st.opening_next = self._opening.next_build(bb.world)
            st.opening_index = self._opening.i
            st.opening_done = self._opening.done
        st.values = dict(self.values if self.values is not None else self.weights)

    def describe(self) -> dict:
        return {"impl": self.name}


@register("ScriptedStrategy")
class ScriptedStrategy(TemplateStrategy):
    """Always the same template."""

    def __init__(self, template: str = "goliath_1fact") -> None:
        super().__init__()
        self.template = template

    def choose(self, bb: Blackboard) -> Optional[Choice]:
        if self.current is not None:
            return None
        t = TEMPLATES.get(self.template)
        if t is None or t.race != int(bb.game.self_race):
            pick = RuleSelector().rule(bb)
            log.warning("ScriptedStrategy: %s is not a %s build; using %s", self.template,
                        race_label(int(bb.game.self_race)), pick)
            return pick
        return self.template

    def describe(self) -> dict:
        return {"impl": self.name, "template": self.template}


def race_label(race: int) -> str:
    return Race(race).name if race in KNOWN_RACES else "Unknown"


@register("RuleSelector")
class RuleSelector(TemplateStrategy):
    """Over the builds of our race (optionally only `templates`): a rush or proxy seen early -> a
    build tagged `rush_safe`; enemy air (or air tech) -> one tagged `anti_air`; otherwise the build
    whose `default_vs` names the enemy race ("Unknown" until a random opponent is seen), falling
    back to the "Unknown" default, then the first build by name. `defaults` ({"Zerg": name, ...})
    overrides the per-race choice."""

    def __init__(self, defaults: Optional[dict] = None, unknown: Optional[str] = None, rush_until_s: int = 360,
                 air_units: float = 3.0, templates: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.defaults = {(k if isinstance(k, str) else race_label(int(k))).capitalize(): v
                         for k, v in (defaults or {}).items()}
        if unknown:
            self.defaults["Unknown"] = unknown
        self.rush_until = rush_until_s * 24
        self.air_units = air_units
        self.only = list(templates) if templates else None

    def pool(self, bb: Blackboard) -> list[str]:
        return race_templates(int(bb.game.self_race), self.only)

    def rule(self, bb: Blackboard) -> Optional[str]:
        b, thr = bb.belief, bb.threats
        pool = self.pool(bb)
        if not pool:
            return None

        def tagged(tag: str) -> Optional[str]:
            return next((n for n in pool if tag in TEMPLATES[n].tags), None)

        rush = b.opening in ("rush", "cheese") or b.proxy or thr.has("rush") or thr.has("worker_rush") \
            or thr.has("proxy")
        if rush and bb.frame < self.rush_until and tagged("rush_safe"):
            return tagged("rush_safe")
        if (b.air >= self.air_units or any(t in AIR_TECH for t in b.tech)) and tagged("anti_air"):
            return tagged("anti_air")
        for race in (race_label(enemy_race(bb)), "Unknown"):
            if self.defaults.get(race) in pool:
                return self.defaults[race]
            pick = next((n for n in pool if race in TEMPLATES[n].default_vs), None)
            if pick:
                return pick
        return pool[0]

    def choose(self, bb: Blackboard) -> Optional[Choice]:
        return self.rule(bb)

    def describe(self) -> dict:
        return {"impl": self.name, "defaults": self.defaults, "templates": self.only}


def _rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        env = os.environ.get("BWBOT_GAME_ID")
        seed = hash(env) & 0xFFFFFFFF if env else int.from_bytes(os.urandom(4), "little")
    return random.Random(seed)


@register("Explore")
class Explore(TemplateStrategy):
    """Data collection: a uniformly random template at game start; after the opening, at each
    checkpoint, switch to another random template with probability `switch_prob`."""

    def __init__(self, templates: Optional[Sequence[str]] = None, switch_prob: float = 0.25,
                 checkpoint_s: int = 180, seed: Optional[int] = None) -> None:
        super().__init__()
        self.templates = list(templates) if templates else None
        self.switch_prob = switch_prob
        self.checkpoint = checkpoint_s * 24
        self.seed = seed

    def on_start(self, bb: Blackboard) -> None:
        super().on_start(bb)
        self.rng = _rng(self.seed)
        self.pool = race_templates(int(bb.game.self_race), self.templates)
        self._next = self.checkpoint

    def choose(self, bb: Blackboard) -> Optional[Choice]:
        if not self.pool:
            return None
        if self.current is None:
            return self.rng.choice(self.pool)
        if bb.frame >= self._next:
            self._next = bb.frame + self.checkpoint
            if bb.strategy.opening_done and len(self.pool) > 1 and self.rng.random() < self.switch_prob:
                return self.rng.choice([t for t in self.pool if t != self.current.name])
        return None

    def describe(self) -> dict:
        return {"impl": self.name, "templates": self.templates, "switch_prob": self.switch_prob}


def load_strategy_model(path: str) -> tuple[Optional[Model], list[str], str]:
    """(model, builds it was trained on, message). The model scores any build by its descriptor,
    so the list is informational; the feature spec must match this code's."""
    from ..learn.features import strategy_spec
    want = strategy_spec()
    cands = [path] + [os.path.join(str(d), os.path.basename(path)) for d in model_search_dirs()]
    for p in cands:
        if not os.path.isfile(p):
            continue
        try:
            m = load_model(p, want)
            return m, list(m.meta.get("templates", [])), f"loaded {p}"
        except (FeatureMismatch, KeyError, ValueError, OSError) as e:
            return None, [], f"rejected {p}: {e}"
    return None, [], f"not found: {path}"


@register("LearnedStrategy")
class LearnedStrategy(TemplateStrategy):
    """Win probability per build from `strategy.npz`, over every build of our race (or only
    `templates`), including builds the model never saw: it scores their descriptors. Re-decides at
    game start (meta-only context), every `period_s`, and on belief events. `select`: follow the
    best build unless it beats the current one by less than `margin`. `blend`: goal = builds
    weighted by softmax(p / temperature) (weights below `min_weight` dropped); the opening comes
    from the top. `epsilon` > 0 explores a random build at a decision. Without a usable model it
    runs the `RuleSelector` rules."""

    def __init__(self, model: str = "strategy.npz", mode: str = "select", period_s: int = 45, margin: float = 0.03,
                 temperature: float = 0.05, min_weight: float = 0.15, epsilon: float = 0.0,
                 templates: Optional[Sequence[str]] = None, seed: Optional[int] = None) -> None:
        super().__init__()
        if mode not in ("select", "blend"):
            raise ValueError(f"mode must be select|blend, got {mode!r}")
        self.model_path, self.mode = model, mode
        self.period = period_s * 24
        self.margin, self.temperature, self.min_weight = margin, temperature, min_weight
        self.epsilon, self.seed = epsilon, seed
        self.only = list(templates) if templates else None
        self.rules = RuleSelector()
        self.model: Optional[Model] = None
        self.model_templates: list[str] = []
        self.message = ""

    def on_start(self, bb: Blackboard) -> None:
        super().on_start(bb)
        self.rules.on_start(bb)
        self.rng = _rng(self.seed)
        self.model, self.model_templates, self.message = load_strategy_model(self.model_path)
        from ..learn.features import build_features
        usable = race_templates(int(bb.game.self_race), self.only)
        self.bfs = [build_features(TEMPLATES[t]) for t in usable]
        self.pool = usable
        if self.model is None or not usable:
            log.warning("LearnedStrategy: %s; using rules", self.message or "no usable templates")
            self.model = None
        else:
            log.info("LearnedStrategy: %s (%s, %d templates)", self.message, self.mode, len(usable))
        self._last = -10 ** 9

    def predict(self, bb: Blackboard) -> dict[str, float]:
        from ..learn.features import strategy_inputs
        X = strategy_inputs(self.context(bb), self.bfs)
        p = np.atleast_1d(self.model.predict(X))
        return {t: float(v) for t, v in zip(self.pool, p)}

    def _due(self, bb: Blackboard) -> bool:
        return self.current is None or bb.frame - self._last >= self.period or \
            any(e in bb.events for e in REDECIDE_EVENTS)

    def choose(self, bb: Blackboard) -> Optional[Choice]:
        if self.model is None:
            self.rules.only = self.only
            return self.rules.rule(bb)
        if not self._due(bb):
            return None
        self._last = bb.frame
        probs = self.predict(bb)
        self.values = probs
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            pick = self.rng.choice(self.pool)
            bb.record("strategy", "explore", pick=pick, probs=probs)
            return pick
        best = max(probs, key=probs.get)
        cur = self.current.name if self.current is not None else None
        if self.mode == "select":
            if cur in probs and best != cur and probs[best] < probs[cur] + self.margin:
                return cur
            return best
        z = np.array([probs[t] for t in self.pool]) / max(self.temperature, 1e-6)
        w = np.exp(z - z.max())
        w /= w.sum()
        weights = {t: float(v) for t, v in zip(self.pool, w) if v >= self.min_weight}
        if cur in weights and best != cur and probs[best] < probs[cur] + self.margin:
            weights[cur] = max(weights.values()) + 1e-3        # keep the current template on top
        return weights

    def describe(self) -> dict:
        return {"impl": self.name, "model": self.model_path, "mode": self.mode, "epsilon": self.epsilon,
                "loaded": self.model is not None, "message": self.message}
