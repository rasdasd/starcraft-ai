"""Builds as data: a JSON file per build, for any race.

    {
      "name": "mech_expand", "race": "Terran", "tags": ["mech", "expand", "macro"],
      "default_vs": ["Protoss", "Terran"],
      "description": "Factory expand into tanks/vultures, up to three bases.",
      "opening": [[9, "Supply_Depot"], [11, "Barracks"], [12, "Refinery"], [16, "Factory"], [20, "Command_Center"]],
      "attack_supply": 70, "retreat_supply": 30,
      "goal": {
        "workers": "min(60, workers_for())", "bases": "2 if count('SCV') < 40 else 3",
        "buildings": {"Factory": "min(6, 2 + 2 * max(0, bases - 1))", "Armory": "done('Factory') >= 2"},
        "units": {"Vulture": 8, "Siege_Tank_Tank_Mode": 16, "Goliath": "6 + 2 * enemy_air"},
        "addons": {"Machine_Shop": "min(3, done('Factory'))"},
        "upgrades": [["Terran_Vehicle_Weapons", 1], ["Ion_Thrusters", 1]],
        "techs": ["Tank_Siege_Mode"]
      },
      "phases": [{"when": "minute >= 12", "units": {"Goliath": 20}}]
    }

Type names may drop the race prefix ("Factory" for "Terran_Factory"). Opening steps are
`[at_supply, type]` and may be units (overlords, zerglings) as well as buildings. Every number in
`goal` and `phases` may instead be an expression over the board (see `Env`); booleans count as
0/1. `phases` apply in order when their `when` holds: dict entries are updated, scalars and lists
replaced. `attack_if` / `retreat_if` expressions replace the supply thresholds; `hold_if` vetoes
an attack. Unknown keys are errors, so typos do not silently drop parts of a build.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from blackboard import Blackboard
from blackboard.sections import Goal, Posture
from bwbot import Race, TechType, UnitType, UpgradeType
from mybot.opening import OpeningStep

from .base import Template, workers_for

log = logging.getLogger("adjutant.builds")

BUILTIN_DIR = Path(__file__).resolve().parents[1] / "builds"
RACES = ("Zerg", "Terran", "Protoss")
SPEC_KEYS = {"name", "race", "tags", "default_vs", "description", "opening", "attack_supply", "retreat_supply",
             "attack_if", "retreat_if", "hold_if", "goal", "phases", "source", "parent"}
GOAL_KEYS = {"workers", "bases", "units", "buildings", "addons", "upgrades", "techs"}


class SpecError(ValueError):
    pass


# ---------------------------------------------------------------------------- expressions
_NODES = (ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp, ast.Call, ast.Name, ast.Load,
          ast.Constant, ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
          ast.USub, ast.UAdd, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Tuple,
          ast.List, ast.keyword)


class Expr:
    """A whitelisted Python expression: arithmetic, comparisons, `and/or/not`, `x if c else y`,
    and calls of the functions `Env` provides. No attributes, subscripts or builtins."""

    def __init__(self, src: str) -> None:
        self.src = src
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as e:
            raise SpecError(f"bad expression {src!r}: {e.msg}") from None
        for node in ast.walk(tree):
            if not isinstance(node, _NODES):
                raise SpecError(f"{type(node).__name__} not allowed in {src!r}")
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise SpecError(f"only plain function calls allowed in {src!r}")
        self.names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.code = compile(tree, "<build>", "eval")

    def __call__(self, env: dict) -> Any:
        return eval(self.code, {"__builtins__": {}}, env)       # noqa: S307 (whitelisted AST)

    def __repr__(self) -> str:
        return f"Expr({self.src!r})"


def _value(v: Any, where: str) -> Any:
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        return Expr(v)
    raise SpecError(f"{where}: expected a number or expression, got {v!r}")


def _num(v: Any, env: dict) -> int:
    x = v(env) if isinstance(v, Expr) else v
    x = float(x)
    return 0 if math.isnan(x) else max(0, int(round(x)))


# ---------------------------------------------------------------------------- names
def _enum_lookup(enum, name: str, race: Optional[str]) -> int:
    for cand in ([name] if race is None else [name, f"{race}_{name}"]):
        v = getattr(enum, cand, None)
        if v is not None:
            return int(v)
    raise SpecError(f"unknown {enum.__name__} {name!r}" + (f" for {race}" if race else ""))


def unit_id(name: str, race: Optional[str]) -> int:
    return _enum_lookup(UnitType, name, race)


def any_unit_id(name: str) -> int:
    for r in (None, *RACES):
        try:
            return _enum_lookup(UnitType, name, r)
        except SpecError:
            pass
    raise SpecError(f"unknown UnitType {name!r}")


def race_name(race: int) -> str:
    try:
        return Race(int(race)).name
    except ValueError:
        return "Unknown"


# ---------------------------------------------------------------------------- board environment
_ENV_KEYS = ("frame", "minute", "supply", "supply_max", "minerals", "gas", "workers", "army", "bases",
             "income_m", "income_g", "under_attack", "enemy_race", "enemy_army", "enemy_air", "enemy_bases",
             "enemy_cloak", "enemy_opening", "enemy_proxy", "rush", "count", "done", "enemy", "workers_for",
             "min", "max", "abs", "round", "int", "float")


def board_env(bb: Blackboard, race: str) -> dict:
    """The names a build expression can use (`_ENV_KEYS`). `count`/`done` take our own type names,
    `enemy` any race's; `enemy_race` is "Zerg"/"Terran"/"Protoss"/"Unknown"."""
    w, b, thr = bb.world, bb.belief, bb.threats
    cache: dict[str, int] = {}

    def own(name: str) -> int:
        if name not in cache:
            cache[name] = unit_id(name, race)
        return cache[name]

    def enemy(name: str) -> int:
        return int(b.counts.get(any_unit_id(name), 0))

    rush = b.opening in ("rush", "cheese") or bool(b.proxy) or thr.has("rush") or thr.has("worker_rush") \
        or thr.has("proxy")
    return {
        "frame": bb.frame, "minute": bb.frame / (24 * 60), "supply": w.supply_used, "supply_max": w.supply_total,
        "minerals": w.minerals, "gas": w.gas, "workers": len(w.workers), "army": w.army_supply,
        "bases": max(1, len(w.depots)), "income_m": w.income_minerals, "income_g": w.income_gas,
        "under_attack": bool(w.under_attack), "enemy_race": race_name(_enemy_race(bb)),
        "enemy_army": float(b.army_supply), "enemy_air": float(b.air),
        "enemy_bases": sum(1 for eb in b.bases if eb.alive), "enemy_cloak": bool(b.cloak),
        "enemy_opening": str(b.opening), "enemy_proxy": bool(b.proxy), "rush": rush,
        "count": lambda n: w.count(own(n)), "done": lambda n: w.count_completed(own(n)), "enemy": enemy,
        "workers_for": lambda per_base=16, cap=60, per_gas=3: workers_for(bb, per_base, cap, per_gas),
        "min": min, "max": max, "abs": abs, "round": round, "int": int, "float": float,
    }


def _enemy_race(bb: Blackboard) -> int:
    r = int(bb.meta.enemy_race)
    if r in (int(Race.Zerg), int(Race.Terran), int(Race.Protoss)):
        return r
    return int(bb.belief.enemy_race)


# ---------------------------------------------------------------------------- build spec
class BuildSpec(Template):
    """A `Template` defined by a JSON document (see the module docstring)."""

    def __init__(self, doc: dict, source: str = "") -> None:
        extra = set(doc) - SPEC_KEYS
        if extra:
            raise SpecError(f"unknown keys {sorted(extra)}")
        try:
            self.name = str(doc["name"])
            rname = str(doc["race"]).capitalize()
        except KeyError as e:
            raise SpecError(f"missing {e.args[0]!r}") from None
        if rname not in RACES:
            raise SpecError(f"race must be one of {RACES}, got {doc['race']!r}")
        self.doc = doc
        self.source = source or str(doc.get("source", ""))
        self.race_name = rname
        self.race = int(getattr(Race, rname))
        self.tags = frozenset(str(t) for t in doc.get("tags", ()))
        self.default_vs = tuple(str(r).capitalize() for r in doc.get("default_vs", ()))
        self.description = str(doc.get("description", ""))
        self.parent = doc.get("parent")
        self.attack_supply = int(doc.get("attack_supply", 40))
        self.retreat_supply = int(doc.get("retreat_supply", 12))
        self.attack_if = Expr(doc["attack_if"]) if doc.get("attack_if") else None
        self.retreat_if = Expr(doc["retreat_if"]) if doc.get("retreat_if") else None
        self.hold_if = Expr(doc["hold_if"]) if doc.get("hold_if") else None
        steps = []
        for i, step in enumerate(doc.get("opening", ())):
            if not (isinstance(step, (list, tuple)) and len(step) == 2):
                raise SpecError(f"opening[{i}]: expected [supply, type], got {step!r}")
            steps.append(OpeningStep(int(step[0]), unit_id(str(step[1]), rname)))
        self.opening = tuple(steps)
        self.base = self._parse_goal(doc.get("goal", {}), "goal")
        self.phases: list[tuple[Optional[Expr], dict]] = []
        for i, ph in enumerate(doc.get("phases", ())):
            ph = dict(ph)
            when = ph.pop("when", None)
            self.phases.append((Expr(when) if when else None, self._parse_goal(ph, f"phases[{i}]")))
        self._check_names()

    # -------------------------------------------------------------- parsing
    def _parse_goal(self, g: dict, where: str) -> dict:
        extra = set(g) - GOAL_KEYS
        if extra:
            raise SpecError(f"{where}: unknown keys {sorted(extra)}")
        out: dict[str, Any] = {}
        for k in ("workers", "bases"):
            if k in g:
                out[k] = _value(g[k], f"{where}.{k}")
        for k in ("units", "buildings", "addons"):
            if k in g:
                out[k] = {unit_id(str(n), self.race_name): _value(v, f"{where}.{k}.{n}") for n, v in g[k].items()}
        if "upgrades" in g:
            ups = []
            for u in g["upgrades"]:
                name, lvl = (u, 1) if isinstance(u, str) else (u[0], u[1] if len(u) > 1 else 1)
                ups.append((_enum_lookup(UpgradeType, str(name), self.race_name), int(lvl)))
            out["upgrades"] = ups
        if "techs" in g:
            out["techs"] = [_enum_lookup(TechType, str(t), self.race_name) for t in g["techs"]]
        return out

    def _exprs(self) -> Iterable[Expr]:
        for e in (self.attack_if, self.retreat_if, self.hold_if):
            if e is not None:
                yield e
        for when, g in [(None, self.base), *self.phases]:
            if when is not None:
                yield when
            for k, v in g.items():
                vals = v.values() if isinstance(v, dict) else [v]
                yield from (x for x in vals if isinstance(x, Expr))

    def _check_names(self) -> None:
        known = set(_ENV_KEYS)
        for e in self._exprs():
            bad = e.names - known
            if bad:
                raise SpecError(f"unknown names {sorted(bad)} in {e.src!r}")

    # -------------------------------------------------------------- Template API
    def env(self, bb: Blackboard) -> dict:
        return board_env(bb, self.race_name)

    def goal(self, bb: Blackboard) -> Goal:
        env = self.env(bb)
        g: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.base.items()}
        for when, ph in self.phases:
            if when is not None and not when(env):
                continue
            for k, v in ph.items():
                if isinstance(v, dict):
                    g.setdefault(k, {}).update(v)
                else:
                    g[k] = v

        def counts(key: str) -> dict[int, int]:
            out = {}
            for t, v in g.get(key, {}).items():
                n = _num(v, env)
                if n > 0:
                    out[int(t)] = n
            return out

        return Goal(units=counts("units"), buildings=counts("buildings"), addons=counts("addons"),
                    upgrades=list(g.get("upgrades", [])), techs=list(g.get("techs", [])),
                    workers=_num(g["workers"], env) if "workers" in g else workers_for(bb),
                    bases=max(1, _num(g.get("bases", 1), env)))

    def posture(self, bb: Blackboard, prev: Posture) -> Posture:
        if self.attack_if is None and self.retreat_if is None and self.hold_if is None:
            return super().posture(bb, prev)
        env = self.env(bb)
        army = bb.world.army_supply
        attack = bool(self.attack_if(env)) if self.attack_if is not None else army >= self.attack_supply
        retreat = bool(self.retreat_if(env)) if self.retreat_if is not None else army < self.retreat_supply
        stance = prev.stance
        if attack:
            stance = "attack"
        elif retreat or stance not in ("attack", "contain"):
            stance = "defend" if bb.world.under_attack else "hold"
        if stance == "attack" and self.hold_if is not None and self.hold_if(env):
            stance = "hold"
        return Posture(stance=stance, attack_supply=self.attack_supply, retreat_supply=self.retreat_supply)

    def to_doc(self) -> dict:
        return dict(self.doc)

    def __repr__(self) -> str:
        return f"BuildSpec({self.name}, {self.race_name}, {self.source})"


# ---------------------------------------------------------------------------- loading
def build_dirs() -> list[Path]:
    """Built-ins, then `builds/` next to each model search dir (bwapi-data/read/builds in a
    tournament, python/models/builds for learned builds), then the repo's `builds/` (your own),
    then `BWBOT_BUILDS` (os.pathsep-separated). Later directories override earlier ones by name."""
    from blackboard.models import model_search_dirs
    dirs = [BUILTIN_DIR]
    dirs += [Path(d) / "builds" for d in reversed(model_search_dirs())]
    dirs.append(BUILTIN_DIR.parents[2] / "builds")
    dirs += [Path(p) for p in os.environ.get("BWBOT_BUILDS", "").split(os.pathsep) if p]
    seen, out = set(), []
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def load_file(path: Path) -> list[BuildSpec]:
    """A file holds one build or a list of builds."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = data if isinstance(data, list) else [data]
    out = []
    for i, doc in enumerate(docs):
        try:
            out.append(BuildSpec(doc, source=str(path)))
        except SpecError as e:
            raise SpecError(f"{path}{f'[{i}]' if len(docs) > 1 else ''}: {e}") from None
    return out


def load_dirs(dirs: Iterable[Path], on_error: Optional[Callable[[str], None]] = None) -> dict[str, BuildSpec]:
    """All `*.json` builds under `dirs` (recursively). A broken user build is reported and skipped,
    a broken built-in raises."""
    out: dict[str, BuildSpec] = {}
    for d in dirs:
        if not Path(d).is_dir():
            continue
        for p in sorted(Path(d).rglob("*.json")):
            try:
                specs = load_file(p)
            except (SpecError, json.JSONDecodeError, OSError) as e:
                if Path(d) == BUILTIN_DIR:
                    raise
                (on_error or log.warning)(f"skipping build {p}: {e}")
                continue
            for s in specs:
                out[s.name] = s
    return out
