"""Engagement slot: Lanchester estimates per army cluster, the global army ratio, and fight logs.

`Engagement` clusters our army, pairs each cluster with the enemy combat units (visible, or seen in
the last `memory_s` seconds) and static defense within reach, and writes an `Engagement` per cluster
plus our whole army against the belief's estimated enemy army. It registers `services["engage"]`
(the evaluator) so tactics can score hypothetical fights.

Fight logging: when a cluster comes into contact, a snapshot (compositions, hit points, the estimate)
opens a fight; it closes after `quiet_s` without contact and is written as an `engagement/fight`
row with the losses on both sides. The learned engagement predictor trains on those rows.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from blackboard import Blackboard, Component, Phase
from blackboard.profile import register
from blackboard.sections import Engagement as EngagementRow
from bwbot import EventType
from bwbot.enums import UnitType as U
from bwbot.observation import UnitFlag

from .. import engage as E
from ..learn.combat import ENGAGE_SPEC, engage_features

log = logging.getLogger("adjutant.engagement")

FPS = 24


@dataclass
class _Enemy:
    id: int
    type: int
    x: int
    y: int
    hp: float
    cloaked: bool


@dataclass
class Fight:
    fid: int
    start: int
    last: int
    center: tuple[int, int]
    snap: dict
    own: dict[int, int] = field(default_factory=dict)      # id -> type
    enemy: dict[int, int] = field(default_factory=dict)
    own_lost: dict[int, int] = field(default_factory=dict)   # type -> count
    enemy_lost: dict[int, int] = field(default_factory=dict)


class FightLog:
    def __init__(self, quiet_frames: int = 5 * FPS, max_frames: int = 120 * FPS, join_px: int = 20 * 32) -> None:
        self.quiet = quiet_frames
        self.max = max_frames
        self.join2 = join_px * join_px
        self.active: list[Fight] = []
        self.closed = 0
        self._next = 0

    def observe(self, frame: int, center: tuple[int, int], own: dict[int, int], enemy: dict[int, int],
                snap_fn) -> None:
        for f in self.active:
            dx, dy = f.center[0] - center[0], f.center[1] - center[1]
            if dx * dx + dy * dy <= self.join2:
                f.own.update(own)
                f.enemy.update(enemy)
                f.last = frame
                f.center = center
                return
        self.active.append(Fight(self._next, frame, frame, center, snap_fn(), dict(own), dict(enemy)))
        self._next += 1

    def destroyed(self, uid: int) -> None:
        for f in self.active:
            if uid in f.own:
                t = f.own[uid]
                f.own_lost[t] = f.own_lost.get(t, 0) + 1
            elif uid in f.enemy:
                t = f.enemy[uid]
                f.enemy_lost[t] = f.enemy_lost.get(t, 0) + 1

    def close_due(self, frame: int, force: bool = False) -> list[Fight]:
        done = [f for f in self.active if force or frame - f.last > self.quiet or frame - f.start > self.max]
        self.active = [f for f in self.active if f not in done]
        self.closed += len(done)
        return done


def _lost_value(lost: dict[int, int], table: E.TypeTable) -> float:
    return sum(n * table.get(t).value for t, n in lost.items())


@register("Engagement")
class LanchesterEngagement(Component):
    phase = Phase.DECIDE
    reads = ("world", "belief")
    writes = ("engagements",)
    order = 5

    def __init__(self, cluster_tiles: int = 8, reach_tiles: int = 14, contact_tiles: int = 9,
                 memory_s: int = 20, sharpness: float = 2.0, log_fights: bool = True,
                 quiet_s: int = 5) -> None:
        self.cluster_px = cluster_tiles * 32
        self.reach_px = reach_tiles * 32
        self.contact_px = contact_tiles * 32
        self.memory = memory_s * FPS
        self.sharpness = sharpness
        self.log_fights = log_fights
        self.quiet_s = quiet_s

    def on_start(self, bb: Blackboard) -> None:
        self.own_table = E.TypeTable(bb.game)
        self.enemy_table = E.TypeTable(bb.game)
        self.fights = FightLog(quiet_frames=self.quiet_s * FPS)
        self.ctx: dict = {"frame": 0, "b_army": 0.0, "own_army": 0.0}
        bb.services["engage"] = self

    # ------------------------------------------------------------------ evaluator API (for tactics)
    def evaluate(self, own: E.Side, enemy: E.Side) -> E.Estimate:
        return E.evaluate(own, enemy, self.sharpness)

    def own_side(self, rows, detects: Optional[bool] = None) -> E.Side:
        side = E.side_from_rows(rows, self.own_table)
        if detects is not None:
            side.detects = side.detects or detects
        return side

    def enemy_side(self, enemies: list[_Enemy]) -> E.Side:
        side = E.Side()
        for e in enemies:
            info = self.enemy_table.get(e.type)
            if e.type in E.DETECTORS:
                side.detects = True
            if info.combat and not info.worker:
                side.members.append(E.Member(info, e.hp, e.cloaked))
        return side

    # ------------------------------------------------------------------ tick
    def tick(self, bb: Blackboard) -> None:
        obs, w, b = bb.obs, bb.world, bb.belief
        frame = bb.frame
        me = getattr(obs, "me", None)
        if me is not None and getattr(me.upgrade_level, "size", 0):
            self.own_table.upgrades = me.upgrade_level
            self.own_table._cache.clear()

        for e in obs.iter_events(EventType.UnitDestroy):
            self.fights.destroyed(int(e.unit))

        self.ctx = {"frame": frame, "b_army": float(b.army_supply), "own_army": float(w.army_supply)}
        enemies = self._enemies(obs, b, frame)
        comsat = any(int(u["type"]) == int(U.Terran_Comsat_Station) and int(u["energy"]) >= 50
                     for u in w.buildings) if len(w.buildings) else False
        own_static = [u for u in w.buildings if self.own_table.get(int(u["type"])).combat] if len(w.buildings) else []

        eng = bb.engagements
        eng.by_squad = {}
        army = w.army
        groups = E.cluster(E.unit_positions(army), self.cluster_px) if len(army) else []
        e_pos = np.array([(e.x, e.y) for e in enemies], np.float32).reshape(-1, 2)
        for ci, idx in enumerate(groups):
            rows = army[idx]
            cx, cy = int(rows["x"].mean()), int(rows["y"].mean())
            near = []
            if len(e_pos):
                d2 = ((e_pos - (cx, cy)) ** 2).sum(axis=1)
                near = [enemies[i] for i in np.nonzero(d2 <= self.reach_px ** 2)[0]]
            statics = [u for u in own_static
                       if (int(u["x"]) - cx) ** 2 + (int(u["y"]) - cy) ** 2 <= self.reach_px ** 2]
            own = self.own_side(list(rows) + statics, detects=comsat)
            enemy = self.enemy_side(near)
            est = self.evaluate(own, enemy)
            contact = False
            if near:
                own_xy = E.unit_positions(rows)
                near_xy = np.array([(e.x, e.y) for e in near], np.float32)
                dmin = ((own_xy[:, None, :] - near_xy[None, :, :]) ** 2).sum(axis=2).min()
                contact = bool(dmin <= self.contact_px ** 2) and len(enemy) > 0
            name = f"c{ci}"
            eng.by_squad[name] = EngagementRow(
                name, (cx, cy), own.hp * est.own_rate, enemy.hp * est.enemy_rate, est.win_prob,
                [e.id for e in near], [int(u["id"]) for u in rows], est.ratio if np.isfinite(est.ratio) else 1e6,
                est.own_left, contact)
            if contact and self.log_fights:
                self.fights.observe(frame, (cx, cy), {int(u["id"]): int(u["type"]) for u in rows},
                                    {e.id: e.type for e in near},
                                    lambda own=own, enemy=enemy, est=est: self._snapshot(bb, own, enemy, est))

        # whole army vs the belief's estimate of the enemy army (units only, no static defense)
        counts = {t: n for t, n in b.counts.items() if not self.enemy_table.get(t).building}
        g_own = self.own_side(list(army), detects=comsat)
        g_est = self.evaluate(g_own, E.side_from_counts(counts, self.enemy_table))
        eng.global_ratio = g_est.ratio if np.isfinite(g_est.ratio) else 1e6
        eng.global_win_prob = g_est.win_prob

        for f in self.fights.close_due(frame):
            self._write(bb, f, frame)
        eng.fights = self.fights.closed

    def on_end(self, bb: Blackboard, won: bool) -> None:
        for f in self.fights.close_due(bb.frame, force=True):
            self._write(bb, f, bb.frame, game_won=won)

    # ------------------------------------------------------------------ helpers
    def _enemies(self, obs, b, frame: int) -> list[_Enemy]:
        out: list[_Enemy] = []
        seen: set[int] = set()
        for u in obs.enemy_units:
            flags = int(u["flags"])
            info = self.enemy_table.get(int(u["type"]))
            if info.building and not flags & UnitFlag.Completed:
                continue
            cloaked = (bool(flags & (UnitFlag.Cloaked | UnitFlag.Burrowed)) and not flags & UnitFlag.Detected) \
                or int(u["type"]) in E.ALWAYS_CLOAKED
            out.append(_Enemy(int(u["id"]), int(u["type"]), int(u["x"]), int(u["y"]),
                              float(u["hit_points"]) + float(u["shields"]), cloaked))
            seen.add(int(u["id"]))
        for uid, (t, x, y, f) in (b.units or {}).items():
            if uid in seen:
                continue
            info = self.enemy_table.get(t)
            if not info.combat or (not info.building and frame - f > self.memory):
                continue
            out.append(_Enemy(uid, t, x, y, info.hp + info.shields, t in E.ALWAYS_CLOAKED))
        return out

    def _snapshot(self, bb: Blackboard, own: E.Side, enemy: E.Side, est: E.Estimate) -> dict:
        lanchester = E.evaluate(own, enemy, self.sharpness)
        return dict(
            x=[round(float(v), 4) for v in engage_features(own, enemy, lanchester, self.ctx)], fh=ENGAGE_SPEC.hash,
            own={str(k): v for k, v in own.composition().items()},
            enemy={str(k): v for k, v in enemy.composition().items()},
            own_hp=round(own.hp), enemy_hp=round(enemy.hp), own_value=own.value, enemy_value=enemy.value,
            own_detect=own.detects, enemy_cloaked=sum(1 for m in enemy.members if m.cloaked),
            own_upg=[int(x) for x in self.own_table.upgrades[:20]] if self.own_table.upgrades.size else [],
            est=est.as_dict(), frame=bb.frame,
        )

    def _write(self, bb: Blackboard, f: Fight, frame: int, game_won: Optional[bool] = None) -> None:
        own_lost = _lost_value(f.own_lost, self.own_table)
        enemy_lost = _lost_value(f.enemy_lost, self.enemy_table)
        row = dict(fid=f.fid, start=f.start, end=frame, center=list(f.center), snap=f.snap,
                   own_lost={str(k): v for k, v in f.own_lost.items()},
                   enemy_lost={str(k): v for k, v in f.enemy_lost.items()},
                   own_lost_value=own_lost, enemy_lost_value=enemy_lost,
                   won=enemy_lost > own_lost, n_own=len(f.own), n_enemy=len(f.enemy))
        if game_won is not None:
            row["game_won"] = game_won
        bb.record("engagement", "fight", **row)

    def describe(self) -> dict:
        return {"impl": self.name, "sharpness": self.sharpness, "cluster_px": self.cluster_px,
                "reach_px": self.reach_px}


@register("LearnedEngagement")
class LearnedEngagement(LanchesterEngagement):
    """Lanchester sides and estimate, with the win probability from `engage.npz` (trained on fight
    rows). `weight` blends model and Lanchester; with no model it is exactly LanchesterEngagement."""

    def __init__(self, model: str = "engage.npz", weight: float = 1.0, **kw) -> None:
        super().__init__(**kw)
        self.model_path = model
        self.weight = weight
        self.model = None
        self.message = ""

    def on_start(self, bb: Blackboard) -> None:
        super().on_start(bb)
        from blackboard.models import model_search_dirs, try_load
        self.model, self.message = try_load(self.model_path, ENGAGE_SPEC, search=model_search_dirs())
        log.info("LearnedEngagement: %s", self.message)

    def evaluate(self, own: E.Side, enemy: E.Side) -> E.Estimate:
        est = E.evaluate(own, enemy, self.sharpness)
        if self.model is None or len(own) == 0 or len(enemy) == 0:
            return est
        p = float(self.model.predict(engage_features(own, enemy, est, self.ctx)))
        return dataclasses.replace(est, win_prob=self.weight * p + (1 - self.weight) * est.win_prob)

    def describe(self) -> dict:
        return {**super().describe(), "model": self.message, "weight": self.weight}
