"""Engagement Evaluator v1: Lanchester square-law fight estimates from GameInfo weapon data.

Per unit type we precompute hit points, armor, size and per-frame damage of its ground/air weapon
against each unit size (explosive/concussive modifiers, weapon upgrades, multi-hit factor). A side's
fighting strength is (damage per frame against the other side's hit-point mix) x (its own hit points);
the square law says the side with the larger product wins and keeps sqrt(1 - weaker/stronger) of
itself. Cloaked units the other side cannot detect take no damage. Bunkers count as four marines,
carriers as their interceptors and reavers as scarabs.

Pure numpy/python: the Engagement component and the learned predictor both build on `Side`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from bwbot.observation import UnitFlag, UnitTypeFlag
from bwbot.enums import Race, UnitType as U, UpgradeType as G, WeaponType as W

NONE_W = int(W.None_)
# damage type -> multiplier per unit size (index: 0 independent, 1 small, 2 medium, 3 large)
_EXPLOSIVE, _CONCUSSIVE = 1, 2
_MULT = {_EXPLOSIVE: (1.0, 0.5, 0.75, 1.0), _CONCUSSIVE: (1.0, 1.0, 0.5, 0.25)}

# stand-ins for units whose damage is dealt by other units
PROXY_WEAPON = {
    int(U.Terran_Bunker): (int(U.Terran_Marine), 4, 32),          # (unit, count, extra range px)
    int(U.Protoss_Carrier): (int(U.Protoss_Interceptor), 8, 0),
    int(U.Protoss_Reaver): (int(U.Protoss_Scarab), 1, 0),
}
ALWAYS_CLOAKED = {int(U.Protoss_Dark_Templar), int(U.Protoss_Observer), int(U.Zerg_Lurker),
                  int(U.Terran_Vulture_Spider_Mine)}
DETECTORS = {int(U.Terran_Science_Vessel), int(U.Terran_Missile_Turret), int(U.Zerg_Overlord),
             int(U.Zerg_Spore_Colony), int(U.Protoss_Observer), int(U.Protoss_Photon_Cannon)}
SPLASH_CROWD = 6          # enemy ground units at which splash weapons count 1.5x


def armor_upgrade(race: int, flyer: bool) -> int:
    if race == Race.Terran:
        return int(G.Terran_Ship_Plating if flyer else G.Terran_Vehicle_Plating)
    if race == Race.Zerg:
        return int(G.Zerg_Flyer_Carapace if flyer else G.Zerg_Carapace)
    if race == Race.Protoss:
        return int(G.Protoss_Air_Armor if flyer else G.Protoss_Ground_Armor)
    return -1


@dataclass
class Weapon:
    per_hit: float            # raw damage incl. upgrades, before armor/size
    factor: int
    cooldown: int
    dtype: int
    max_range: int            # pixels
    splash: bool
    count: int = 1            # proxied units (bunker marines, interceptors)

    def per_frame(self, armor: float, size: int) -> float:
        mult = _MULT.get(self.dtype, (1.0, 1.0, 1.0, 1.0))[min(max(size, 0), 3)]
        hit = max(0.5, (self.per_hit - armor) * mult)
        return hit * self.factor * self.count / max(1, self.cooldown)


@dataclass
class TypeInfo:
    type_id: int
    hp: float
    shields: float
    armor: float
    size: int
    flyer: bool
    building: bool
    worker: bool
    ground: Optional[Weapon]
    air: Optional[Weapon]
    value: float              # minerals + gas
    supply: float

    @property
    def combat(self) -> bool:
        return self.ground is not None or self.air is not None


class TypeTable:
    """Per-type fight stats for one player's upgrade levels."""

    def __init__(self, game, upgrades: Optional[np.ndarray] = None) -> None:
        self.game = game
        self.upgrades = upgrades if upgrades is not None else np.zeros(0, np.int32)
        self._weapons = {int(w["id"]): w for w in game.weapon_types}
        self._cache: dict[int, TypeInfo] = {}

    def level(self, upgrade: int) -> int:
        u = int(upgrade)
        return int(self.upgrades[u]) if 0 <= u < self.upgrades.size else 0

    def _weapon(self, wid: int, count: int = 1, extra_range: int = 0) -> Optional[Weapon]:
        w = self._weapons.get(int(wid))
        if w is None or int(wid) == NONE_W or int(w["damage_amount"]) <= 0:
            return None
        per_hit = float(w["damage_amount"]) + float(w["damage_bonus"]) * self.level(int(w["upgrade_type"]))
        return Weapon(per_hit, max(1, int(w["damage_factor"])), max(1, int(w["damage_cooldown"])),
                      int(w["damage_type"]), int(w["max_range"]) + extra_range,
                      int(w["outer_splash_radius"]) > 0, count)

    def get(self, type_id: int) -> TypeInfo:
        t = int(type_id)
        info = self._cache.get(t)
        if info is not None:
            return info
        ut = self.game.unit_types
        r = ut[min(max(t, 0), len(ut) - 1)]
        flags = int(r["flags"])
        flyer = bool(flags & UnitTypeFlag.Flyer)
        g_w = self._weapon(int(r["ground_weapon"]))
        a_w = self._weapon(int(r["air_weapon"]))
        if t in PROXY_WEAPON:
            src, n, extra = PROXY_WEAPON[t]
            s = ut[src]
            g_w = self._weapon(int(s["ground_weapon"]), n, extra) or g_w
            a_w = self._weapon(int(s["air_weapon"]), n, extra) or a_w
        armor = float(r["armor"]) + self.level(armor_upgrade(int(r["race"]), flyer))
        info = TypeInfo(t, float(r["max_hit_points"]), float(r["max_shields"]), armor, int(r["size"]), flyer,
                        bool(flags & UnitTypeFlag.Building), bool(flags & UnitTypeFlag.Worker), g_w, a_w,
                        float(r["mineral_price"]) + float(r["gas_price"]), float(r["supply_required"]) / 2)
        self._cache[t] = info
        return info


@dataclass
class Member:
    info: TypeInfo
    hp: float                 # current hit points + shields
    cloaked: bool = False     # cannot be hit unless the other side detects


@dataclass
class Side:
    members: list[Member] = field(default_factory=list)
    detects: bool = False

    @property
    def hp(self) -> float:
        return sum(m.hp for m in self.members)

    @property
    def value(self) -> float:
        return sum(m.info.value for m in self.members)

    def composition(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for m in self.members:
            out[m.info.type_id] = out.get(m.info.type_id, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self.members)


def side_from_rows(rows, table: TypeTable, include_workers: bool = False,
                   include_buildings: bool = True) -> Side:
    """Side from UNIT_DTYPE rows (own or visible enemy units). Non-combat units are left out."""
    side = Side()
    for u in rows:
        info = table.get(int(u["type"]))
        if not info.combat or (info.worker and not include_workers) or (info.building and not include_buildings):
            if int(u["type"]) in DETECTORS:
                side.detects = True
            continue
        flags = int(u["flags"])
        if info.building and not flags & UnitFlag.Completed:
            continue
        cloaked = bool(flags & (UnitFlag.Cloaked | UnitFlag.Burrowed)) and not flags & UnitFlag.Detected
        cloaked = cloaked or int(u["type"]) in ALWAYS_CLOAKED
        side.members.append(Member(info, float(u["hit_points"]) + float(u["shields"]), cloaked))
        if int(u["type"]) in DETECTORS:
            side.detects = True
    return side


def side_from_counts(counts: dict[int, float], table: TypeTable, include_workers: bool = False) -> Side:
    """Side at full health from estimated counts per type (belief), fractional counts rounded."""
    side = Side()
    for t, n in counts.items():
        info = table.get(int(t))
        if int(t) in DETECTORS:
            side.detects = True
        if not info.combat or (info.worker and not include_workers):
            continue
        for _ in range(int(round(n))):
            side.members.append(Member(info, info.hp + info.shields, int(t) in ALWAYS_CLOAKED))
    return side


def damage_rate(attackers: Side, targets: Side) -> float:
    """Damage per frame `attackers` deal against the hit-point mix of `targets`."""
    total = targets.hp
    if total <= 0 or not attackers.members:
        return 0.0
    ground_n = sum(1 for m in targets.members if not m.info.flyer)
    hittable: dict[int, list] = {}                    # type -> [info, hp fraction]
    for m in targets.members:
        if m.cloaked and not attackers.detects:
            continue
        e = hittable.setdefault(m.info.type_id, [m.info, 0.0])
        e[1] += m.hp / total
    if not hittable:
        return 0.0
    shooters: dict[int, list] = {}
    for a in attackers.members:
        e = shooters.setdefault(a.info.type_id, [a.info, 0])
        e[1] += 1
    rate = 0.0
    for a_info, n in shooters.values():
        for t_info, w_frac in hittable.values():
            w = a_info.air if t_info.flyer else a_info.ground
            if w is None:
                continue
            sh_frac = min(1.0, t_info.shields / max(1.0, t_info.hp + t_info.shields))
            # shields ignore the size modifier; hull damage takes armor and size into account
            pf = sh_frac * w.per_frame(0.0, 0) + (1 - sh_frac) * w.per_frame(t_info.armor, t_info.size)
            if w.splash and not t_info.flyer and ground_n >= SPLASH_CROWD:
                pf *= 1.5
            rate += n * w_frac * pf
    return rate


@dataclass
class Estimate:
    own_rate: float           # damage per frame we deal
    enemy_rate: float
    own_hp: float
    enemy_hp: float
    ratio: float              # own strength / enemy strength (inf if the enemy cannot hurt us)
    win_prob: float
    own_left: float           # expected surviving fraction of our side (0 if we lose)
    enemy_left: float
    frames: float             # expected fight length

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if math.isfinite(v) else 1e6) for k, v in self.__dict__.items()}


def evaluate(own: Side, enemy: Side, sharpness: float = 2.0) -> Estimate:
    ra, rb = damage_rate(own, enemy), damage_rate(enemy, own)
    ha, hb = own.hp, enemy.hp
    ea, eb = ra * ha, rb * hb
    if ha <= 0 < hb:
        return Estimate(ra, rb, ha, hb, 0.0, 0.0, 0.0, 1.0, 0.0)
    if hb <= 0 < ha:
        return Estimate(ra, rb, ha, hb, math.inf, 1.0, 1.0, 0.0, 0.0)
    if ea <= 0 and eb <= 0:
        return Estimate(ra, rb, ha, hb, 1.0, 0.5, 1.0 if ha else 0.0, 1.0 if hb else 0.0, 0.0)
    if eb <= 0:
        return Estimate(ra, rb, ha, hb, math.inf, 1.0, 1.0, 0.0, hb / max(ra, 1e-9))
    if ea <= 0:
        return Estimate(ra, rb, ha, hb, 0.0, 0.0, 0.0, 1.0, ha / max(rb, 1e-9))
    ratio = ea / eb
    p = 1.0 / (1.0 + ratio ** -sharpness)
    if ratio >= 1:
        own_left, enemy_left = math.sqrt(1 - 1 / ratio), 0.0
        frames = hb / ra * (1 + own_left) / 2
    else:
        own_left, enemy_left = 0.0, math.sqrt(1 - ratio)
        frames = ha / rb * (1 + enemy_left) / 2
    return Estimate(ra, rb, ha, hb, ratio, p, own_left, enemy_left, frames)


def cluster(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Single-link clusters of (n, 2) pixel positions; returns index arrays, biggest first."""
    n = len(points)
    if n == 0:
        return []
    labels = -np.ones(n, np.int32)
    r2 = radius * radius
    cur = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = cur
        stack = [i]
        while stack:
            j = stack.pop()
            d = ((points - points[j]) ** 2).sum(axis=1)
            for k in np.nonzero((d <= r2) & (labels < 0))[0]:
                labels[k] = cur
                stack.append(int(k))
        cur += 1
    groups = [np.nonzero(labels == c)[0] for c in range(cur)]
    groups.sort(key=len, reverse=True)
    return groups


def unit_positions(rows) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros((0, 2), np.float32)
    return np.stack([rows["x"].astype(np.float32), rows["y"].astype(np.float32)], axis=1)


def members_of(rows: Iterable, ids: set[int]):
    return [u for u in rows if int(u["id"]) in ids]
