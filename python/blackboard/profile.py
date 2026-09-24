"""Profiles: which implementation fills each slot, with parameters and fallbacks.

A profile is plain data:

    {"name": "adjutant", "base": "parity",            # optional inheritance
     "slots": {"strategy": {"impl": "LearnedStrategy", "model": "strategy.npz", "mode": "blend",
                             "fallback": {"impl": "RuleSelector"}},
               "tactics": null},                      # null removes a slot from the base
     "config": {"time_budget_ms": 40}}

Implementations are registered by name (`@register("RuleSelector")`). A JSON file (or the
`BWBOT_PROFILE_JSON` env var) can override any built-in profile without code changes; this keeps
the competition bot configurable after freezing. JSON (not TOML) for Python 3.10 compatibility.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from .component import Component

REGISTRY: dict[str, Callable[..., Component]] = {}


def register(name: Optional[str] = None):
    def deco(factory):
        REGISTRY[name or factory.__name__] = factory
        return factory
    return deco


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and "impl" not in v:
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(name_or_spec: Any, builtins: dict[str, dict]) -> dict:
    """Flatten `base` inheritance. Accepts a profile name, a dict, or a path to a JSON file."""
    if isinstance(name_or_spec, dict):
        spec = name_or_spec
    elif isinstance(name_or_spec, str) and name_or_spec in builtins:
        spec = builtins[name_or_spec]
    elif isinstance(name_or_spec, (str, Path)) and Path(name_or_spec).is_file():
        spec = json.loads(Path(name_or_spec).read_text(encoding="utf-8"))
    else:
        raise KeyError(f"unknown profile {name_or_spec!r}; built-ins: {sorted(builtins)}")
    spec = copy.deepcopy(spec)
    base = spec.pop("base", None)
    if base:
        parent = resolve(base, builtins)
        spec = deep_merge(parent, {**spec, "name": spec.get("name", parent.get("name"))})
    spec.setdefault("slots", {})
    spec.setdefault("config", {})
    return spec


def apply_env_override(spec: dict) -> dict:
    raw = os.environ.get("BWBOT_PROFILE_JSON") or os.environ.get("BWBOT_PROFILE_FILE")
    if not raw:
        return spec
    over = json.loads(Path(raw).read_text(encoding="utf-8")) if Path(raw).is_file() else json.loads(raw)
    return deep_merge(spec, over)


def build_component(slot: str, cfg: dict) -> Component:
    cfg = dict(cfg)
    impl = cfg.pop("impl")
    fb_cfg = cfg.pop("fallback", None)
    if impl not in REGISTRY:
        raise KeyError(f"slot {slot!r}: unknown implementation {impl!r}; registered: {sorted(REGISTRY)}")
    comp = REGISTRY[impl](**cfg)
    comp.slot = slot
    if fb_cfg:
        comp.fallback = build_component(slot, fb_cfg)
    return comp


def build(spec: dict) -> list[Component]:
    return [build_component(slot, cfg) for slot, cfg in spec["slots"].items() if cfg]
