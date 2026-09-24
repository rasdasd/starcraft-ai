"""Small scripted build-order bot, race-agnostic, for sparring in self-play.

Subclasses declare a build order, the army mix, upgrades and attack thresholds; placement comes
from the shim's `getBuildLocation` queries (creep and pylon power included). Deliberately simple
and deterministic-ish so Adjutant's results against it are comparable across runs.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

from bwbot import Actions, Bot, ClientConfig, EventType, GameInfo, Observation, UnitFlag, UnitType
from bwbot.observation import UnitTypeFlag

log = logging.getLogger("sparring")

LARVA = UnitType.Zerg_Larva


class ScriptBot(Bot):
    race = "Terran"
    config = ClientConfig(frame_skip=2, local_speed=0, include_tiles=True)

    worker: int = UnitType.Terran_SCV
    hall: int = UnitType.Terran_Command_Center
    supply: int = UnitType.Terran_Supply_Depot
    refinery: int = UnitType.Terran_Refinery
    build_order: list[tuple[int, int]] = []        # (supply trigger, building type)
    army: list[tuple[int, int, float]] = []        # (unit type, producer type, weight)
    upgrades: list[tuple[int, int, bool]] = []     # (building type, upgrade/tech id, is_tech)
    max_workers = 20
    gas_workers = 3
    attack_at = 12
    retreat_below = 4
    larva_based = False

    def on_start(self, game: GameInfo) -> None:
        self.rng = random.Random(game.map_name)
        me = game.self_player
        self.home = (int(me.start_location[0]) * 32 + 64, int(me.start_location[1]) * 32 + 48)
        self.home_tile = (int(me.start_location[0]), int(me.start_location[1]))
        starts = [tuple(int(v) for v in s) for s in game.start_locations.tolist()]
        self.targets = [(x * 32 + 64, y * 32 + 48) for x, y in starts if (x, y) != self.home_tile]
        self.rng.shuffle(self.targets)
        self.enemy_buildings: dict[int, tuple[int, int]] = {}
        self.queries: dict[int, int] = {}
        self.next_q = 1
        self.pending: Optional[tuple[int, int]] = None      # (type, frame issued)
        self._pending_target = 0
        self._last_supply = -10_000
        self.gas_ids: set[int] = set()
        self.ordered: dict[int, int] = {}                  # production orders in flight (latency)
        self.failures: dict[int, int] = {}
        self.bad_tiles: set[tuple[int, int, int]] = set()
        self.last_tile: Optional[tuple[int, int, int]] = None
        self.index = 0
        self.attacking = False

    # ------------------------------------------------------------------ helpers
    def cost(self, t: int) -> tuple[int, int]:
        ut = self.game.unit_types
        return int(ut["mineral_price"][t]), int(ut["gas_price"][t])

    def supply_cost(self, t: int) -> int:
        return int(self.game.unit_types["supply_required"][t]) // 2

    def is_building(self, t: int) -> bool:
        return bool(self.game.unit_types["flags"][t] & UnitTypeFlag.Building)

    # ------------------------------------------------------------------ frame
    def on_frame(self, obs: Observation, act: Actions) -> None:
        self.obs, self.act = obs, act
        mine = obs.my_units
        if len(mine) == 0:
            return
        self.minerals, self.gas = obs.minerals, obs.gas
        if self.pending is not None:
            t, f = self.pending
            m, g = self.cost(t)
            self.minerals -= m
            self.gas -= g
        self._remember_enemies(obs)
        self._placements(obs, act)
        self._workers(obs, act)
        self._build(obs, act)
        self._upgrades(obs, act)
        self._produce(obs, act)
        self._army(obs, act)
        if obs.frame_count % 1440 < 2:
            counts = {self.game.type_name(int(t)): int(n) for t, n in zip(*np.unique(mine["type"], return_counts=True))}
            log.info("f%d supply %d/%d min %d gas %d step %d/%d attacking=%s %s", obs.frame_count, obs.supply_used,
                     obs.supply_total, obs.minerals, obs.gas, self.index, len(self.build_order), self.attacking, counts)

    def _remember_enemies(self, obs: Observation) -> None:
        en = obs.enemy_units
        flags = self.game.unit_types["flags"][np.clip(en["type"], 0, len(self.game.unit_types) - 1)]
        for u in en[(flags & UnitTypeFlag.Building) != 0]:
            self.enemy_buildings[int(u["id"])] = (int(u["x"]), int(u["y"]))
        for e in obs.iter_events(EventType.UnitDestroy):
            self.enemy_buildings.pop(e.unit, None)
        if obs.tiles.ndim == 2 and self.enemy_buildings:
            seen = {int(i) for i in en["id"]}
            for uid, (x, y) in list(self.enemy_buildings.items()):
                ty, tx = min(y // 32, obs.tiles.shape[0] - 1), min(x // 32, obs.tiles.shape[1] - 1)
                if obs.visible[ty, tx] and uid not in seen:
                    del self.enemy_buildings[uid]

    def _placements(self, obs: Observation, act: Actions) -> None:
        for qid, ok, tx, ty in obs.placement_results:
            t = self.queries.pop(qid, None)
            if t is None or not ok:
                if t is not None:
                    log.info("f%d no location for %s", obs.frame_count, self.game.type_name(t))
                    self.pending = None
                    self._failed(t, obs.frame_count)
                continue
            if (t, tx, ty) in self.bad_tiles:
                self._request(act, t, jitter=True)
                continue
            builder = self._builder(obs, tx * 32, ty * 32)
            if builder is not None:
                act.build(builder, t, tx, ty)
                self.last_tile = (t, tx, ty)
            else:
                self.pending = None

    def _builder(self, obs: Observation, x: int, y: int):
        w = obs.completed(obs.my_units_of_type(self.worker))
        free = w[(w["carry_resource_type"] == 0) & ((w["flags"] & UnitFlag.Gathering) != 0)]
        return obs.nearest(free if len(free) else w, x, y)

    def _request(self, act: Actions, t: int, jitter: bool = False) -> None:
        qid = self.next_q
        self.next_q += 1
        self.queries[qid] = t
        near = self.home_tile
        pylons = self.obs.my_completed(UnitType.Protoss_Pylon)
        if len(pylons) and t != UnitType.Protoss_Pylon and t != self.refinery:
            p = pylons[self.rng.randrange(len(pylons))]
            near = (int(p["x"]) // 32, int(p["y"]) // 32)
        if (jitter or self.failures.get(t)) and t != self.refinery:
            near = (near[0] + self.rng.randint(-8, 8), near[1] + self.rng.randint(-8, 8))
        act.get_build_location(qid, t, near, 40)
        self.pending = (t, self.obs.frame_count)

    def _workers(self, obs: Observation, act: Actions) -> None:
        workers = obs.completed(obs.my_units_of_type(self.worker))
        fields = obs.minerals_fields
        refs = obs.completed(obs.my_units_of_type(self.refinery))
        self.gas_ids &= {int(i) for i in workers["id"]}
        want = self.gas_workers * len(refs)
        if len(self.gas_ids) < want:
            miners = workers[(workers["carry_resource_type"] == 0) & ((workers["flags"] & UnitFlag.Gathering) != 0)
                             & ~np.isin(workers["id"], list(self.gas_ids))]
            for w in miners[:want - len(self.gas_ids)]:
                act.gather(w, refs[0])
                self.gas_ids.add(int(w["id"]))
        for w in obs.idle(workers):
            if int(w["id"]) in self.gas_ids and len(refs):
                act.gather(w, refs[0])
                continue
            f = obs.nearest(fields, int(w["x"]), int(w["y"]))
            if f is not None:
                act.gather(w, f)

    def _required(self, t: int) -> int:
        return sum(1 for _, bt in self.build_order[: self.index + 1] if bt == t) + (1 if t == self.hall else 0)

    def _failed(self, t: int, f: int) -> None:
        self.failures[t] = self.failures.get(t, 0) + 1
        if self.last_tile is not None and self.last_tile[0] == t:
            self.bad_tiles.add(self.last_tile)
        if self.failures[t] >= 3 and self.index < len(self.build_order) and self.build_order[self.index][1] == t:
            log.info("f%d giving up on %s", f, self.game.type_name(t))
            self.index += 1
            self.failures[t] = 0

    def _build(self, obs: Observation, act: Actions) -> None:
        f = obs.frame_count
        supply_left = obs.supply_total - obs.supply_used
        need_supply = obs.supply_total < 200 and supply_left <= max(2, obs.supply_used // 8)
        if self.pending is not None:
            t, since = self.pending
            if obs.count(t) >= self._pending_target:
                self.pending = None
                self.failures[t] = 0
            elif f - since > 24 * 25:
                self.pending = None
                self._failed(t, f)
            elif need_supply and supply_left <= 0 and t != self.supply and not self.larva_based:
                self.pending = None
                self.queries.clear()
            else:
                return
        if need_supply:
            in_prod = obs.count(self.supply) - obs.count(self.supply, completed_only=True)
            hall_prod = obs.count(self.hall) - obs.count(self.hall, completed_only=True)
            if in_prod == 0 and hall_prod == 0 and self.minerals >= self.cost(self.supply)[0]:
                if self.larva_based:
                    larva = obs.my_units_of_type(LARVA)
                    if len(larva) and f - self._last_supply > int(self.game.unit_types["build_time"][self.supply]) + 24:
                        act.morph(larva[0], self.supply)
                        self.minerals -= self.cost(self.supply)[0]
                        self._last_supply = f
                else:
                    self._pending_target = obs.count(self.supply) + 1
                    self._request(act, self.supply)
                    return
        while self.index < len(self.build_order):
            trig, t = self.build_order[self.index]
            if obs.count(t) >= self._required(t):
                self.index += 1
                continue
            m, g = self.cost(t)
            if obs.supply_used >= trig and self.minerals >= m and self.gas >= g:
                self._pending_target = obs.count(t) + 1
                self._request(act, t)
            return

    def _upgrades(self, obs: Observation, act: Actions) -> None:
        me = obs.me
        for bt, uid, is_tech in self.upgrades:
            have = me.has_researched if is_tech else me.upgrade_level
            busy = me.is_researching if is_tech else me.is_upgrading
            if (have.size > uid and have[uid]) or (busy.size > uid and busy[uid]):
                continue
            b = obs.completed(obs.my_units_of_type(bt))
            b = obs.idle(b)
            if not len(b):
                continue
            table = self.game.tech_types if is_tech else self.game.upgrade_types
            info = next((d for d in table if d["id"] == uid), None)
            if info is None:
                continue
            m, g = int(info["mineral_price"]), int(info["gas_price"])
            if self.minerals >= m and self.gas >= g:
                (act.research if is_tech else act.upgrade)(b[0], uid)
                self.minerals -= m
                self.gas -= g

    def _produce(self, obs: Observation, act: Actions) -> None:
        supply_left = obs.supply_total - obs.supply_used
        workers = obs.count(self.worker)
        want_workers = workers < self.max_workers and (self.index >= len(self.build_order) or
                                                       obs.supply_used < self.build_order[self.index][0])
        if self.larva_based:
            makers = list(obs.my_units_of_type(LARVA))
        else:
            makers = []
            for t, prod, _ in self.army:
                for b in obs.completed(obs.my_units_of_type(prod)):
                    if b["train_queue_count"] == 0:
                        makers.append(b)
            for h in obs.completed(obs.my_units_of_type(self.hall)):
                if h["train_queue_count"] == 0:
                    makers.insert(0, h)
        f = obs.frame_count
        self.ordered = {u: t for u, t in self.ordered.items() if f - t < 16}
        for mk in makers:
            if supply_left <= 0:
                break
            if int(mk["id"]) in self.ordered:
                continue
            mtype = int(mk["type"])
            if want_workers and (self.larva_based or mtype == self.hall) and self.minerals >= 50:
                (act.morph if self.larva_based else act.train)(mk, self.worker)
                self.ordered[int(mk["id"])] = f
                self.minerals -= 50
                supply_left -= 1
                workers += 1
                want_workers = workers < self.max_workers and self.index >= len(self.build_order) - 1
                continue
            options = [(t, w) for t, prod, w in self.army
                       if (self.larva_based or prod == mtype) and self._tech_ok(obs, t)]
            if not options:
                continue
            t = self.rng.choices([o[0] for o in options], weights=[o[1] for o in options])[0]
            m, g = self.cost(t)
            if self.minerals >= m and self.gas >= g and supply_left >= self.supply_cost(t):
                (act.morph if self.larva_based else act.train)(mk, t)
                self.ordered[int(mk["id"])] = f
                self.minerals -= m
                self.gas -= g
                supply_left -= self.supply_cost(t)

    def _tech_ok(self, obs: Observation, t: int) -> bool:
        return all(obs.count(r, completed_only=True) > 0
                   for r in self.game.unit_type_required_units[t] if r != LARVA)

    def _army(self, obs: Observation, act: Actions) -> None:
        types = {t for t, _, _ in self.army}
        mine = obs.completed(obs.my_units)
        army = mine[np.isin(mine["type"], list(types))]
        n = len(army)
        if n >= self.attack_at:
            self.attacking = True
        elif n < self.retreat_below:
            self.attacking = False
        enemies = obs.enemy_units
        near_home = enemies[((enemies["x"] - self.home[0]) ** 2 + (enemies["y"] - self.home[1]) ** 2) < (24 * 32) ** 2]
        if obs.frame_count % 24 != 0 and not len(near_home):
            return
        if len(near_home) and not self.attacking:
            for u in army:
                if u["flags"] & UnitFlag.Idle or obs.frame_count % 48 == 0:
                    tgt = obs.nearest(near_home, int(u["x"]), int(u["y"]))
                    act.attack_move(u, int(tgt["x"]), int(tgt["y"]))
            return
        if self.attacking:
            target = self._target(army)
            for u in army:
                if u["flags"] & UnitFlag.Idle:
                    tgt = obs.nearest(enemies, int(u["x"]), int(u["y"])) if len(enemies) else None
                    if tgt is not None and (int(tgt["x"]) - int(u["x"])) ** 2 + (int(tgt["y"]) - int(u["y"])) ** 2 < 400 ** 2:
                        act.attack_move(u, int(tgt["x"]), int(tgt["y"]))
                    elif target is not None:
                        act.attack_move(u, *target)
        else:
            for u in obs.idle(army):
                if (int(u["x"]) - self.home[0]) ** 2 + (int(u["y"]) - self.home[1]) ** 2 > 320 ** 2:
                    act.move(u, self.home[0] + self.rng.randint(-96, 96), self.home[1] + self.rng.randint(-96, 96))

    def _target(self, army: np.ndarray) -> Optional[tuple[int, int]]:
        if not len(army):
            return None
        cx, cy = float(np.mean(army["x"])), float(np.mean(army["y"]))
        if self.enemy_buildings:
            return min(self.enemy_buildings.values(), key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        if not self.targets:
            return None
        tx, ty = self.targets[0]
        if (cx - tx) ** 2 + (cy - ty) ** 2 < 160 ** 2 and len(self.targets) > 1:
            self.targets.append(self.targets.pop(0))
        return self.targets[0]

    def on_end(self, is_winner: bool) -> None:
        log.info("%s game over: %s", type(self).__name__, "WIN" if is_winner else "LOSS")
