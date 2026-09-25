"""Tactical Dispatcher: splits the army into squads and gives each one an order.

Squads (the `squads` section, executed and leased by the Micro slot):
- `defense`: enemies near our bases (or crisis threats). Takes the closest units until the engagement
  estimate says they win, or everything if they never do.
- `repair`: badly damaged mechanical units walk home for SCV repair and rejoin when fixed.
- `harass`: a few vultures raid the least defended enemy base when the posture asks for harassment.
- `main`: everything else, ordered from the strategy posture: hold at our front (main choke, or the
  natural's exit once we own it), contain outside the enemy natural, or attack the nearest known
  enemy building. Pushes start at the posture's attack supply, regroup when strung out, and retreat
  when the engagement estimate for the main cluster drops below `retreat_prob` (not when all-in).
- Enemy air near home while holding turns the main order into `hunt_air`.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from blackboard.sections import Squad, SquadOrder
from bwbot import EventType, Race, UnitType as U
from bwbot.observation import UnitFlag, UnitTypeFlag

from .. import engage as E
from ..learn.combat import OPTIONS, TACTICS_SPEC, tactics_features
from .engagement import _Enemy

FPS = 24
OPTION_OF = {"attack": "attack", "gather": "attack", "contain": "contain", "retreat": "retreat",
             "hold": "hold", "defend": "hold"}
NOT_SQUAD = {int(U.Terran_Vulture_Spider_Mine), int(U.Protoss_Interceptor), int(U.Protoss_Scarab),
             int(U.Zerg_Larva), int(U.Zerg_Egg), int(U.Zerg_Overlord)}
SUPPORT = {int(U.Terran_Medic), int(U.Terran_Science_Vessel), int(U.Protoss_Observer), int(U.Zerg_Defiler)}


def _center(rows) -> tuple[int, int]:
    return int(rows["x"].mean()), int(rows["y"].mean())


def _d2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


@register("Tactics")
class Tactics(Component):
    phase = Phase.DECIDE
    reads = ("world", "meta", "belief", "strategy", "engagements", "threats")
    writes = ("squads",)
    priority = Priority.COMBAT
    order = 20

    def __init__(self, defend_radius_tiles: int = 22, retreat_prob: float = 0.35, resume_prob: float = 0.6,
                 retreat_s: int = 12, regroup_frac: float = 0.6, harass_max: int = 4, repair_hp: float = 0.35,
                 repaired_hp: float = 0.85, min_push: int = 3, defend_win: float = 0.85,
                 executor: str = "micro", log_s: int = 5, choke_back_tiles: int = 4) -> None:
        self.log_frames = log_s * FPS
        self.choke_back_px = choke_back_tiles * 32
        self.defend_px = defend_radius_tiles * 32
        self.retreat_prob = retreat_prob
        self.resume_prob = resume_prob
        self.retreat_frames = retreat_s * FPS
        self.regroup_frac = regroup_frac
        self.harass_max = harass_max
        self.repair_hp = repair_hp
        self.repaired_hp = repaired_hp
        self.min_push = min_push
        self.defend_win = defend_win
        self.executor = executor

    def on_start(self, bb: Blackboard) -> None:
        self.pushing = False
        self.retreat_until = -1
        self.repairing: set[int] = set()
        self.nat_exit = self._natural_exit(bb.game)
        self.table = E.TypeTable(bb.game)
        self.types: dict[int, tuple[int, bool]] = {}      # id -> (type, ours)
        self.lost = [0.0, 0.0]                           # value lost: ours, theirs
        self.last_log = -10 ** 9
        self.option: Optional[str] = None
        self._enemy_cache: tuple[int, list] = (-1, [])

    # ------------------------------------------------------------------ tick
    def tick(self, bb: Blackboard) -> None:
        w, b, st = bb.world, bb.belief, bb.strategy
        frame = bb.frame
        self._track_losses(bb)
        units = self._available(bb)
        squads: dict[str, Squad] = {}
        self.option = None
        self.option_point: Optional[tuple[int, int]] = None
        if len(units) == 0:
            bb.squads.squads = squads
            bb.squads.army_order = None
            self.pushing = False
            return
        taken: set[int] = set()
        home = self._home(bb)

        rep = self._repair_squad(bb, units, home)
        if rep is not None:
            squads["repair"] = rep
            taken |= rep.units

        threat = self._home_threat(bb)
        if threat is not None:
            d = self._defense_squad(bb, units, taken, threat)
            if d is not None:
                squads["defense"] = d
                taken |= d.units

        if st.posture.harass and threat is None:
            h = self._harass_squad(bb, units, taken)
            if h is not None:
                squads["harass"] = h
                taken |= h.units

        rest = units[~np.isin(units["id"], list(taken))] if taken else units
        if len(rest):
            order = self._main_order(bb, rest, frame)
            squads["main"] = Squad("main", {int(u["id"]) for u in rest}, order, self.priority)
            if self.option is None:
                self.option = OPTION_OF.get(order.kind)
            if self.option is not None and frame - self.last_log >= self.log_frames:
                self.last_log = frame
                self._log_decision(bb, rest, order)
        bb.squads.squads = squads
        bb.squads.army_order = None

    def on_end(self, bb: Blackboard, won: bool) -> None:
        bb.record("tactics", "final", lost=[round(v) for v in self.lost])

    # ------------------------------------------------------------------ decision logging (tactics value model)
    def _track_losses(self, bb: Blackboard) -> None:
        obs = bb.obs
        for e in obs.iter_events(EventType.UnitDestroy):
            t = self.types.pop(int(e.unit), None)
            if t is not None:
                self.lost[0 if t[1] else 1] += self.table.get(t[0]).value
        mine, theirs = obs.my_units, obs.enemy_units
        if len(mine):
            self.types.update(zip(mine["id"].tolist(), zip(mine["type"].tolist(), [True] * len(mine))))
        if len(theirs):
            self.types.update(zip(theirs["id"].tolist(), zip(theirs["type"].tolist(), [False] * len(theirs))))

    def _known_enemies(self, bb: Blackboard) -> list:
        if self._enemy_cache[0] != bb.frame:
            ev = bb.services.get("engage")
            self._enemy_cache = (bb.frame, ev._enemies(bb.obs, bb.belief, bb.frame) if ev is not None else [])
        return self._enemy_cache[1]

    def _context(self, bb: Blackboard, rows) -> dict:
        w, b, g = bb.world, bb.belief, bb.game
        n = max(1, len(rows))
        tanks = sum(1 for u in rows if int(u["type"]) in (int(U.Terran_Siege_Tank_Tank_Mode),
                                                          int(U.Terran_Siege_Tank_Siege_Mode)))
        air = sum(1 for u in rows if _is_flyer(bb, u))
        stance = bb.strategy.posture.stance
        if bb.threats.posture_override and bb.frame <= bb.threats.override_until:
            stance = bb.threats.posture_override
        return {"frame": bb.frame, "own_army": float(w.army_supply), "b_army": float(b.army_supply),
                "global_p": float(bb.engagements.global_win_prob), "global_ratio": float(bb.engagements.global_ratio),
                "own_bases": len(w.depots), "enemy_bases": sum(1 for e in b.bases if e.alive),
                "workers": len(w.workers), "threat": float(bb.threats.level), "tank_frac": tanks / n,
                "air_frac": air / n, "stance": stance, "map_diag": math.hypot(g.map_width * 32, g.map_height * 32)}

    def _target_info(self, bb: Blackboard, rows, point) -> dict:
        ev = bb.services.get("engage")
        c = _center(rows)
        near = [e for e in self._known_enemies(bb) if _d2((e.x, e.y), point) <= (12 * 32) ** 2]
        p, value, static = 1.0, 0.0, 0.0
        if ev is not None and near:
            enemy = ev.enemy_side(near)
            p = ev.evaluate(ev.own_side(list(rows)), enemy).win_prob
            value = enemy.value
            static = sum(m.info.value for m in enemy.members if m.info.building)
        return {"dist": math.sqrt(_d2(c, point)), "p": float(p), "value": float(value), "static": float(static)}

    def _log_decision(self, bb: Blackboard, rows, order: SquadOrder) -> None:
        point = self.option_point
        if point is None:
            point = (order.x, order.y)
            if order.kind == "gather":
                point = self._attack_target(bb, _center(rows)) or point
        x = tactics_features(self._context(bb, rows), self.option, self._target_info(bb, rows, point))
        own_value = sum(self.table.get(int(u["type"])).value for u in rows)
        bb.record("tactics", "decision", opt=self.option, order=order.kind, x=[round(float(v), 4) for v in x],
                  fh=TACTICS_SPEC.hash, lost=[round(v) for v in self.lost], value=round(own_value),
                  learned=self.learned_choice)

    learned_choice = False

    # ------------------------------------------------------------------ squads
    def _available(self, bb: Blackboard):
        army = bb.world.army
        mine = bb.obs.my_units
        if len(mine):
            support = mine[np.isin(mine["type"], list(SUPPORT))]
            support = support[(support["flags"] & int(UnitFlag.Completed)) != 0]
            if len(support):
                army = np.concatenate([army, support]) if len(army) else support
        if len(army) == 0:
            return army
        keep = []
        for i, u in enumerate(army):
            uid, t = int(u["id"]), int(u["type"])
            if t in NOT_SQUAD:
                continue
            owner = bb.leases.owner(uid)
            if owner is not None and owner != self.executor:
                continue
            keep.append(i)
        return army[np.array(keep, dtype=np.intp)] if keep else army[:0]

    def _repair_squad(self, bb: Blackboard, units, home) -> Optional[Squad]:
        if bb.meta.self_race != int(Race.Terran) or len(bb.world.workers) == 0:
            self.repairing.clear()
            return None
        ut = bb.game.unit_types
        ids = set()
        for u in units:
            t = int(u["type"])
            if not ut["flags"][t] & UnitTypeFlag.Mechanical:
                continue
            frac = float(u["hit_points"]) / max(1, int(ut["max_hit_points"][t]))
            uid = int(u["id"])
            if uid in self.repairing and frac >= self.repaired_hp:
                self.repairing.discard(uid)
            elif frac < self.repair_hp:
                self.repairing.add(uid)
            if uid in self.repairing:
                ids.add(uid)
        self.repairing &= {int(u["id"]) for u in units}
        if not ids:
            return None
        return Squad("repair", ids, SquadOrder("retreat", home[0], home[1]), self.priority)

    def _home_threat(self, bb: Blackboard) -> Optional[tuple[tuple[int, int], list]]:
        """(center, enemy rows) of the biggest enemy group near one of our bases, if any."""
        for t in bb.threats.active:
            if t.severity >= 0.5:
                return (t.x, t.y), [u for u in bb.world.enemies if int(u["id"]) in set(t.units)]
        enemies = bb.world.enemies
        if len(enemies) == 0:
            return None
        ut = bb.game.unit_types
        homes = self._bases(bb)
        near = []
        for u in enemies:
            t = int(u["type"])
            flags = int(ut["flags"][t])
            if flags & UnitTypeFlag.Building and not (ut["ground_weapon"][t] != E.NONE_W):
                continue
            if t in (int(U.Zerg_Overlord), int(U.Protoss_Observer)):
                continue
            if any(_d2((int(u["x"]), int(u["y"])), h) <= self.defend_px ** 2 for h in homes):
                near.append(u)
        if not near:
            return None
        arr = np.array(near, dtype=enemies.dtype)
        groups = E.cluster(E.unit_positions(arr), 10 * 32)
        g = arr[groups[0]]
        return _center(g), list(g)

    def _defense_squad(self, bb: Blackboard, units, taken: set[int], threat) -> Optional[Squad]:
        (tx, ty), enemies = threat
        free = [u for u in units if int(u["id"]) not in taken]
        if not free:
            return None
        free.sort(key=lambda u: _d2((int(u["x"]), int(u["y"])), (tx, ty)))
        ev = bb.services.get("engage")
        chosen = []
        if ev is not None and enemies:
            enemy = ev.enemy_side([_as_enemy(u) for u in enemies])
            for u in free:
                chosen.append(u)
                if ev.evaluate(ev.own_side(chosen), enemy).win_prob >= self.defend_win:
                    break
        else:
            chosen = free
        kind = "hunt_air" if enemies and all(_is_flyer(bb, u) for u in enemies) else "defend"
        return Squad("defense", {int(u["id"]) for u in chosen}, SquadOrder(kind, tx, ty), Priority.COMBAT + 5)

    def _harass_squad(self, bb: Blackboard, units, taken: set[int]) -> Optional[Squad]:
        vult = [u for u in units if int(u["type"]) == int(U.Terran_Vulture) and int(u["id"]) not in taken]
        if len(vult) < 2:
            return None
        target = self._harass_target(bb)
        if target is None:
            return None
        ids = {int(u["id"]) for u in vult[: self.harass_max]}
        order = SquadOrder("harass", target[0], target[1])
        eng = self._engagement_for(bb, ids)
        if eng is not None and eng.contact and eng.win_prob < self.retreat_prob:
            home = self._hold_point(bb)
            order = SquadOrder("retreat", home[0], home[1])
        return Squad("harass", ids, order, self.priority)

    def _main_order(self, bb: Blackboard, rows, frame: int) -> SquadOrder:
        w, b, st = bb.world, bb.belief, bb.strategy
        posture = st.posture
        hold = self._hold_point(bb)
        stance = posture.stance
        if bb.threats.posture_override and frame <= bb.threats.override_until:
            stance = bb.threats.posture_override
        ids = {int(u["id"]) for u in rows}
        eng = self._engagement_for(bb, ids)
        all_in = stance == "all_in"

        if stance in ("attack", "all_in"):
            supply = w.army_supply
            if not self.pushing and (all_in or supply >= posture.attack_supply) and len(rows) >= self.min_push:
                self.pushing = True
            elif self.pushing and not all_in and supply < posture.retreat_supply:
                self.pushing = False
        else:
            self.pushing = False

        if self.pushing and not all_in and eng is not None and eng.contact and eng.win_prob < self.retreat_prob:
            self.retreat_until = frame + self.retreat_frames
        retreating = frame < self.retreat_until and not all_in
        if retreating and eng is not None and eng.win_prob >= self.resume_prob and \
                bb.engagements.global_win_prob >= 0.5:
            self.retreat_until = -1
            retreating = False
        if retreating:
            return SquadOrder("retreat", hold[0], hold[1])

        air = self._air_near(bb, hold)
        if self.pushing:
            groups = E.cluster(E.unit_positions(rows), 8 * 32)
            biggest = rows[groups[0]]
            if len(biggest) < self.regroup_frac * len(rows) and not (eng is not None and eng.contact):
                cx, cy = _center(biggest)
                return SquadOrder("gather", cx, cy)
            tgt = self._attack_target(bb, _center(rows))
            if tgt is not None:
                return SquadOrder("attack", tgt[0], tgt[1])
        if air is not None and _has_anti_air(bb, rows):
            return SquadOrder("hunt_air", air[0], air[1], air[2])
        if stance == "contain":
            c = self._contain_point(bb)
            if c is not None and (eng is None or not eng.contact or eng.win_prob >= self.retreat_prob):
                return SquadOrder("contain", c[0], c[1])
        return SquadOrder("defend" if stance == "defend" else "hold", hold[0], hold[1])

    # ------------------------------------------------------------------ positions
    def _home(self, bb: Blackboard) -> tuple[int, int]:
        mt = bb.world.main_tile
        return mt[0] * 32 + 64, mt[1] * 32 + 48

    def _bases(self, bb: Blackboard) -> list[tuple[int, int]]:
        return [self._home(bb)] + [tuple(d) for d in bb.world.depots]

    def _owns_natural(self, bb: Blackboard) -> bool:
        nt = bb.world.natural_tile
        if nt is None:
            return False
        c = (nt[0] * 32 + 64, nt[1] * 32 + 48)
        return any(_d2(d, c) <= (6 * 32) ** 2 for d in bb.world.depots)

    def _hold_point(self, bb: Blackboard) -> tuple[int, int]:
        if self.nat_exit is not None and self._owns_natural(bb):
            return self.nat_exit
        if bb.world.main_choke is not None:
            # inside the main, off the ramp itself: units parked in a narrow choke block our own SCVs
            cx, cy = bb.world.main_choke
            hx, hy = self._home(bb)
            d = math.hypot(hx - cx, hy - cy)
            k = min(0.5, self.choke_back_px / d) if d > 0 else 0.0
            return int(cx + (hx - cx) * k), int(cy + (hy - cy) * k)
        return self._home(bb)

    @staticmethod
    def _natural_exit(game) -> Optional[tuple[int, int]]:
        main = game.base(game.self_main_id) if game.self_main_id >= 0 else None
        nat = game.base(game.self_natural_id) if game.self_natural_id >= 0 else None
        if main is None or nat is None:
            return None
        others = [s for s in game.start_bases if tuple(s.tile) != tuple(game.self_player.start_location)]
        if others:
            ex = sum(s.tile[0] for s in others) / len(others) * 32
            ey = sum(s.tile[1] for s in others) / len(others) * 32
        else:
            ex, ey = game.map_width * 16, game.map_height * 16
        best, best_d = None, math.inf
        for c in game.chokes:
            if nat.area_id not in (c.area_a, c.area_b) or main.area_id in (c.area_a, c.area_b) or c.blocking:
                continue
            d = _d2(c.center, (ex, ey))
            if d < best_d:
                best, best_d = tuple(c.center), d
        return best

    def _contain_point(self, bb: Blackboard) -> Optional[tuple[int, int]]:
        g, b = bb.game, bb.belief
        if b.enemy_start is None:
            return None
        sb = next((s for s in g.start_bases if tuple(s.tile) == tuple(b.enemy_start)), None)
        nat = g.base(sb.natural_id) if sb is not None and sb.natural_id >= 0 else None
        if nat is None:
            return None
        hx, hy = self._home(bb)
        nx, ny = nat.center
        d = math.hypot(hx - nx, hy - ny) or 1.0
        k = min(1.0, 10 * 32 / d)
        return int(nx + (hx - nx) * k), int(ny + (hy - ny) * k)

    def _attack_target(self, bb: Blackboard, origin: tuple[int, int]) -> Optional[tuple[int, int]]:
        b = bb.belief
        if len(bb.world.enemies):
            ut = bb.game.unit_types
            ground = [u for u in bb.world.enemies if not ut["flags"][int(u["type"])] & UnitTypeFlag.Flyer]
            near = [u for u in ground if _d2((int(u["x"]), int(u["y"])), origin) <= (14 * 32) ** 2]
            if near:
                u = min(near, key=lambda u: _d2((int(u["x"]), int(u["y"])), origin))
                return int(u["x"]), int(u["y"])
        if b.buildings:
            t, x, y = min(b.buildings, key=lambda r: _d2((r[1], r[2]), origin))
            return x, y
        if b.enemy_start is not None:
            return b.enemy_start[0] * 32 + 64, b.enemy_start[1] * 32 + 48
        if b.start_candidates:
            s = max(b.start_candidates.items(), key=lambda kv: kv[1])[0]
            return s[0] * 32 + 64, s[1] * 32 + 48
        return None

    def _harass_target(self, bb: Blackboard) -> Optional[tuple[int, int]]:
        b = bb.belief
        alive = [e for e in b.bases if e.alive]
        if not alive:
            return None
        g = bb.game
        # the enemy base seen longest ago that is not their main: expansions are usually weakest
        main = tuple(b.enemy_start) if b.enemy_start is not None else None
        cands = [e for e in alive if main is None or tuple(e.tile) != main] or alive
        e = min(cands, key=lambda e: e.last_seen)
        base = g.base(e.base_id)
        return tuple(base.center) if base is not None else (e.tile[0] * 32 + 64, e.tile[1] * 32 + 48)

    def _air_near(self, bb: Blackboard, point) -> Optional[tuple[int, int, int]]:
        enemies = bb.world.enemies
        if len(enemies) == 0:
            return None
        ut = bb.game.unit_types
        best = None
        for u in enemies:
            t = int(u["type"])
            if not ut["flags"][t] & UnitTypeFlag.Flyer or t in (int(U.Zerg_Overlord), int(U.Protoss_Observer)):
                continue
            p = (int(u["x"]), int(u["y"]))
            if any(_d2(p, h) <= self.defend_px ** 2 for h in self._bases(bb) + [point]):
                if best is None or _d2(p, point) < _d2(best[:2], point):
                    best = (p[0], p[1], int(u["id"]))
        return best

    def _engagement_for(self, bb: Blackboard, ids: set[int]):
        best, n = None, 0
        for e in bb.engagements.by_squad.values():
            k = len(ids.intersection(e.own_ids))
            if k > n:
                best, n = e, k
        return best

    def describe(self) -> dict:
        return {"impl": self.name, "retreat_prob": self.retreat_prob, "resume_prob": self.resume_prob}


@register("LearnedTactics")
class LearnedTactics(Tactics):
    """Tactics whose main-squad order is picked by a value model (`tactics.npz`) over the options
    hold / contain / attack (nearest target) / attack_base (weakest known enemy base) / retreat.

    Options are re-scored every `eval_s`. The scripted order stays unless the best option scores
    `margin` above it; an override is kept for `hold_s` (dropped early when the engaged cluster is
    losing). Crisis overrides and anti-air hunts always use the scripted order. With probability
    `epsilon` per evaluation a random option is played for `explore_s` (logged, for training). With
    no model and epsilon 0 it is exactly Tactics.
    """

    def __init__(self, model: str = "tactics.npz", epsilon: float = 0.0, margin: float = 0.05,
                 hold_s: int = 15, explore_s: int = 20, eval_s: int = 2, seed: Optional[int] = None,
                 **kw) -> None:
        super().__init__(**kw)
        self.model_path = model
        self.eval_frames = eval_s * FPS
        self.epsilon = epsilon
        self.margin = margin
        self.hold_frames = hold_s * FPS
        self.explore_frames = explore_s * FPS
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.message = ""

    def on_start(self, bb: Blackboard) -> None:
        super().on_start(bb)
        from blackboard.models import model_search_dirs, try_load
        self.model, self.message = try_load(self.model_path, TACTICS_SPEC, search=model_search_dirs())
        self.choice: Optional[str] = None
        self.choice_until = -1
        self.next_eval = 0
        self.cached: dict = {}
        self.scores: dict[str, float] = {}

    def _main_order(self, bb: Blackboard, rows, frame: int) -> SquadOrder:
        scripted = super()._main_order(bb, rows, frame)
        self.learned_choice = False
        override = bb.threats.posture_override and frame <= bb.threats.override_until
        if scripted.kind == "hunt_air" or override or (self.model is None and self.epsilon <= 0):
            self.choice = None
            return scripted
        if self.choice in ("attack", "attack_base", "contain"):
            eng = self._engagement_for(bb, {int(u["id"]) for u in rows})
            if eng is not None and eng.contact and eng.win_prob < self.retreat_prob:
                self.choice, self.choice_until = None, -1
                self.retreat_until = frame + self.retreat_frames
                return SquadOrder("retreat", *self._hold_point(bb))
        fresh = frame >= self.next_eval
        if fresh:
            self.next_eval = frame + self.eval_frames
            self.cached = self._options(bb, rows)
        options = self.cached
        if self.choice is not None and frame < self.choice_until and self.choice in options:
            return self._take(bb, rows, self.choice, options)
        if not fresh:
            return scripted
        base = OPTION_OF.get(scripted.kind, "hold")
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            pick = str(self.rng.choice(sorted(options)))
            self.choice, self.choice_until = pick, frame + self.explore_frames
            return self._take(bb, rows, pick, options)
        if self.model is None:
            self.choice = None
            return scripted
        ctx = self._context(bb, rows)
        names = sorted(options)
        X = np.stack([tactics_features(ctx, o, options[o][1]) for o in names])
        vals = np.atleast_1d(self.model.predict(X)).reshape(len(names), -1)[:, 0]
        self.scores = {o: float(v) for o, v in zip(names, vals)}
        best = max(names, key=lambda o: self.scores[o])
        if base in self.scores and self.scores[best] - self.scores[base] < self.margin:
            best = base
        if best == base:
            self.choice = None
            return scripted
        self.choice, self.choice_until = best, frame + self.hold_frames
        return self._take(bb, rows, best, options)

    def _take(self, bb: Blackboard, rows, option: str, options: dict) -> SquadOrder:
        self.option = option
        self.learned_choice = True
        order = options[option][0]
        self.option_point = (order.x, order.y)
        if option in ("attack", "attack_base"):
            self.pushing = True
            groups = E.cluster(E.unit_positions(rows), 8 * 32)
            biggest = rows[groups[0]]
            eng = self._engagement_for(bb, {int(u["id"]) for u in rows})
            if len(biggest) < self.regroup_frac * len(rows) and not (eng is not None and eng.contact):
                cx, cy = _center(biggest)
                return SquadOrder("gather", cx, cy)
        else:
            self.pushing = False
        return order

    def _options(self, bb: Blackboard, rows) -> dict[str, tuple[SquadOrder, dict]]:
        hold = self._hold_point(bb)
        at_hold = self._target_info(bb, rows, hold)
        out = {"hold": (SquadOrder("hold", hold[0], hold[1]), at_hold),
               "retreat": (SquadOrder("retreat", hold[0], hold[1]), at_hold)}
        c = self._contain_point(bb)
        if c is not None:
            out["contain"] = (SquadOrder("contain", c[0], c[1]), self._target_info(bb, rows, c))
        tgt = self._attack_target(bb, _center(rows))
        if tgt is not None:
            out["attack"] = (SquadOrder("attack", tgt[0], tgt[1]), self._target_info(bb, rows, tgt))
        best = None
        for e in bb.belief.bases:
            if not e.alive:
                continue
            base = bb.game.base(e.base_id)
            p = tuple(base.center) if base is not None else (e.tile[0] * 32 + 64, e.tile[1] * 32 + 48)
            info = self._target_info(bb, rows, p)
            if best is None or (info["p"], -info["dist"]) > (best[1]["p"], -best[1]["dist"]):
                best = (p, info)
        if best is not None:
            out["attack_base"] = (SquadOrder("attack", best[0][0], best[0][1]), best[1])
        assert set(out) <= set(OPTIONS)
        return out

    def describe(self) -> dict:
        return {**super().describe(), "model": self.message, "epsilon": self.epsilon, "margin": self.margin}


def _as_enemy(u) -> _Enemy:
    return _Enemy(int(u["id"]), int(u["type"]), int(u["x"]), int(u["y"]),
                  float(u["hit_points"]) + float(u["shields"]), False)


def _is_flyer(bb: Blackboard, u) -> bool:
    return bool(bb.game.unit_types["flags"][int(u["type"])] & UnitTypeFlag.Flyer)


def _has_anti_air(bb: Blackboard, rows) -> bool:
    aw = bb.game.unit_types["air_weapon"]
    return any(int(aw[int(u["type"])]) != E.NONE_W for u in rows)
