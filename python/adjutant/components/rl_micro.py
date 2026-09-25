"""RLMicro (experimental): Combat Micro whose fight behaviour is a learned Q policy.

Everything except fighting (retreats, walking, tanks, spellcasters) stays scripted. A combat unit
with enemies in reach picks one of `learn.micro.ACTIONS` every `decide_frames`:
    focus    scripted focus fire (threat per hit point, no overkill)
    nearest  attack the closest enemy it can hit
    kite     step 3 tiles away from the closest threat
    back     walk back toward the squad point (or away from the enemies if already there)
    stay     no new command
Without a model the choice is the scripted one (kite when the scripted kite rule fires, else focus),
so an exploring RLMicro with no model plays like Micro plus `epsilon` random actions. With a
model, the greedy action replaces the scripted one only when its Q value is `margin` higher
(offline Q estimates for rarely tried actions are noisy). Transitions
for a `log_frac` sample of units go to `micro/step` rows (see learn/micro.py).
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from blackboard import Blackboard
from blackboard.profile import register
from bwbot import EventType

from ..learn.micro import ACTIONS, MICRO_SPEC, ORDERS, all_actions
from .micro import Micro, _range

ORDER_GROUP = {"attack": "attack", "attack_base": "attack", "harass": "attack", "hunt_air": "attack",
               "defend": "defend", "hold": "hold", "contain": "hold"}
A = {a: i for i, a in enumerate(ACTIONS)}


@register("RLMicro")
class RLMicro(Micro):
    def __init__(self, model: str = "micro.npz", epsilon: float = 0.1, margin: float = 0.05, decide_frames: int = 12,
                 log_frac: float = 0.3, radius_tiles: int = 8, left_frames: int = 48, seed: Optional[int] = None,
                 **kw) -> None:
        super().__init__(**kw)
        self.model_path = model
        self.epsilon = epsilon
        self.margin = margin
        self.decide_frames = decide_frames
        self.log_frac = log_frac
        self.radius_px = radius_tiles * 32
        self.left_frames = left_frames
        self.seed = seed

    def on_start(self, bb: Blackboard) -> None:
        super().on_start(bb)
        from blackboard.models import model_search_dirs, try_load
        self.model, self.message = try_load(self.model_path, MICRO_SPEC, search=model_search_dirs())
        self.rng = random.Random(self.seed)
        self.choice: dict[int, tuple[int, int]] = {}          # uid -> (action, until frame)
        self.pending: dict[int, dict] = {}                     # uid -> open transition
        self.dead: set[int] = set()
        self.counts = [0] * len(ACTIONS)
        self.logged = 0

    # ------------------------------------------------------------------ tick
    def tick(self, bb: Blackboard) -> None:
        obs = bb.obs
        for e in obs.iter_events(EventType.UnitDestroy):
            self.dead.add(int(e.unit))
        self._hp_now = self._hp_map(obs.my_units, self.table)
        self._hp_now.update(self._hp_map(self._targets(bb.world.enemies), self.etable))
        mine = obs.my_units
        self._own_xy = np.array([(int(u["x"]), int(u["y"])) for u in mine], np.float32).reshape(-1, 2)
        self._own_ids = [int(u["id"]) for u in mine]
        super().tick(bb)
        for uid in [k for k, p in self.pending.items() if uid_gone(k, p, self.dead, bb.frame, self.left_frames)]:
            self._close(bb, uid, None)

    def on_end(self, bb: Blackboard, won: bool) -> None:
        for uid in list(self.pending):
            self._close(bb, uid, None)

    # ------------------------------------------------------------------ fighting
    def _fight(self, bb, u, info, sq, enemies, e_xy, near, d2, assigned, frame):
        uid = int(u["id"])
        held = self.choice.get(uid)
        if held is not None and frame < held[1]:
            a = held[0]
        else:
            x = self._features(bb, u, info, sq, enemies, e_xy, near, d2)
            a = self._choose(bb, u, info, enemies, near, d2, x, frame)
            self.choice[uid] = (a, frame + self.decide_frames)
            self.counts[a] += 1
            if uid in self.pending:
                self._close(bb, uid, x)
            if self._sampled(uid):
                self.pending[uid] = {"f": frame, "last": frame, "x": x, "a": a,
                                     "snap": self._snapshot(u, enemies, e_xy, d2)}
        p = self.pending.get(uid)
        if p is not None:
            p["last"] = frame
        return self._do(bb, ACTIONS[a], u, info, sq, enemies, near, d2, assigned, frame)

    def _choose(self, bb, u, info, enemies, near, d2, x, frame) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(ACTIONS))
        scripted = A["kite"] if self._kite(u, info, enemies, near, d2, bb.act, frame) is not None else A["focus"]
        if self.model is None:
            return scripted
        q = np.asarray(self.model.predict(all_actions(x)), np.float32).reshape(len(ACTIONS), -1)[:, 0]
        best = int(np.argmax(q))
        return best if q[best] - q[scripted] > self.margin else scripted

    def _do(self, bb, action, u, info, sq, enemies, near, d2, assigned, frame):
        act = bb.act
        if action == "focus":
            return self._attack(bb, u, self._pick(u, info, enemies, near, d2, assigned, harass=sq.order.kind == "harass"),
                                frame)
        if action == "nearest":
            return self._attack(bb, u, enemies[min(near, key=lambda i: d2[i])], frame)
        ux, uy = int(u["x"]), int(u["y"])
        if action == "kite":
            e = enemies[min(near, key=lambda i: d2[i])]
            tx, ty = _away(ux, uy, int(e["x"]), int(e["y"]), 96)
            return self._cmd(u, 4.5, ("kite", tx // 32, ty // 32), frame, lambda: act.move(u, tx, ty), repeat=6)
        if action == "back":
            dx, dy = sq.order.x, sq.order.y
            if (ux - dx) ** 2 + (uy - dy) ** 2 <= (4 * 32) ** 2:
                cx = float(np.mean([int(enemies[i]["x"]) for i in near]))
                cy = float(np.mean([int(enemies[i]["y"]) for i in near]))
                dx, dy = _away(ux, uy, int(cx), int(cy), 160)
            return self._cmd(u, 4.5, ("back", dx // 64, dy // 64), frame, lambda: act.move(u, dx, dy), repeat=12)
        return None

    # ------------------------------------------------------------------ state and reward
    def _features(self, bb, u, info, sq, enemies, e_xy, near, d2) -> np.ndarray:
        ux, uy = int(u["x"]), int(u["y"])
        w = info.air if info.flyer and info.ground is None else (info.ground or info.air)
        rng_px = max(_range(info.ground), _range(info.air))
        cd = int(u["ground_weapon_cooldown"]) if info.ground is not None else int(u["air_weapon_cooldown"])
        r2 = self.radius_px ** 2
        local = np.nonzero(d2 <= r2)[0] if len(d2) else np.zeros(0, int)
        threats = melee = 0
        e_hp = 0.0
        for i in local:
            e = enemies[i]
            ei = self.etable.get(int(e["type"]))
            e_hp += float(e["hit_points"]) + float(e["shields"])
            ew = ei.air if info.flyer else ei.ground
            if ew is not None and not ei.worker and math.sqrt(d2[i]) <= ew.max_range + 48:
                threats += 1
                melee += ew.max_range <= 32
        own_d2 = ((self._own_xy - (ux, uy)) ** 2).sum(axis=1) if len(self._own_xy) else np.zeros(0)
        own_local = [self._own_ids[i] for i in np.nonzero(own_d2 <= r2)[0]]
        o_hp = sum(self._hp_now.get(i, (0.0, 0.0))[0] for i in own_local)
        n_i = min(near, key=lambda i: d2[i])
        ne = enemies[n_i]
        nei = self.etable.get(int(ne["type"]))
        new = nei.air if info.flyer else nei.ground
        tgt = self._pick(u, info, enemies, near, d2, {}, harass=sq.order.kind == "harass")
        t_d = math.hypot(int(tgt["x"]) - ux, int(tgt["y"]) - uy) if tgt is not None else 0.0
        t_hp = _hp_frac(tgt, self.etable.get(int(tgt["type"]))) if tgt is not None else 0.0
        group = ORDER_GROUP.get(sq.order.kind, "other")
        x = [
            float(u["hit_points"]) / max(1.0, info.hp), float(u["shields"]) / max(1.0, info.shields),
            cd / max(1, w.cooldown) if w is not None else 0.0, float(rng_px >= 96), rng_px / 256, float(info.flyer),
            math.log1p(info.value) / 6, melee / 5, threats / 5, len(local) / 10, len(own_local) / 10,
            math.sqrt(d2[n_i]) / 320, (new.max_range if new is not None else 0) / 256,
            float(new is not None and new.max_range <= 32), _hp_frac(ne, nei),
            math.log1p(o_hp) / 8, math.log1p(e_hp) / 8, max(-1.0, min(1.0, math.log((o_hp + 1) / (e_hp + 1)) / 5)),
            t_d / 320, t_hp, min(2.0, math.hypot(sq.order.x - ux, sq.order.y - uy) / 1024),
        ] + [float(group == o) for o in ORDERS]
        return np.asarray(x, np.float32)

    def _snapshot(self, u, enemies, e_xy, d2) -> dict:
        r2 = self.radius_px ** 2
        ids = [int(enemies[i]["id"]) for i in np.nonzero(d2 <= r2)[0]] if len(d2) else []
        ux, uy = int(u["x"]), int(u["y"])
        if len(self._own_xy):
            own_d2 = ((self._own_xy - (ux, uy)) ** 2).sum(axis=1)
            ids += [self._own_ids[i] for i in np.nonzero(own_d2 <= r2)[0]]
        return {i: self._hp_now[i] for i in ids if i in self._hp_now}

    def _reward(self, snap: dict) -> float:
        r = 0.0
        for i, (hp0, per_hp, mine) in snap.items():
            if i in self.dead:
                lost = hp0
            elif i in self._hp_now:
                lost = max(0.0, hp0 - self._hp_now[i][0])
            else:
                lost = 0.0                        # out of sight: unknown, count nothing
            r += (-lost if mine else lost) * per_hp
        return r / 100.0

    def _close(self, bb: Blackboard, uid: int, x2: Optional[np.ndarray]) -> None:
        p = self.pending.pop(uid)
        if x2 is None:
            self.choice.pop(uid, None)
        bb.record(self.slot, "step", uid=uid, a=p["a"], r=round(self._reward(p["snap"]), 4),
                  x=[round(float(v), 4) for v in p["x"]],
                  x2=None if x2 is None else [round(float(v), 4) for v in x2], fh=MICRO_SPEC.hash,
                  dt=bb.frame - p["f"])
        self.logged += 1

    def _hp_map(self, units, table) -> dict:
        out = {}
        if units is None or len(units) == 0:
            return out
        for u in units:
            info = table.get(int(u["type"]))
            hp = float(u["hit_points"]) + float(u["shields"])
            out[int(u["id"])] = (hp, info.value / max(1.0, info.hp + info.shields), table is self.table)
        return out

    def _sampled(self, uid: int) -> bool:
        return self.log_frac > 0 and (uid * 2654435761) % 1000 < self.log_frac * 1000

    def describe(self) -> dict:
        d = super().describe()
        d.update(model=self.model_path, loaded=self.message, epsilon=self.epsilon, decide_frames=self.decide_frames,
                 log_frac=self.log_frac)
        return d

    def summary(self) -> dict:
        return {"actions": dict(zip(ACTIONS, self.counts)), "logged": self.logged}


def uid_gone(uid: int, p: dict, dead: set, frame: int, left_frames: int) -> bool:
    return uid in dead or frame - p["last"] > left_frames


def _away(ux: int, uy: int, ex: int, ey: int, dist: float) -> tuple[int, int]:
    dx, dy = ux - ex, uy - ey
    n = math.hypot(dx, dy) or 1.0
    return int(ux + dx / n * dist), int(uy + dy / n * dist)


def _hp_frac(e, ei) -> float:
    return (float(e["hit_points"]) + float(e["shields"])) / max(1.0, ei.hp + ei.shields)
