"""Read-side wrappers over the shim's FlatBuffers messages.

`GameInfo` (from GameStart) holds static per-match data; `Observation` (from Frame) exposes the
unit list as a zero-copy numpy structured array plus convenience filters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
from typing import Iterator, Optional

import numpy as np

from .enums import EventType, Race, UnitType
from .generated import bw

# ---------------------------------------------------------------------------
# UnitState struct layout. MUST match proto/bw.fbs::UnitState field order (all 4-byte fields).
# ---------------------------------------------------------------------------
UNIT_FIELDS = [
    ("id", "<i4"), ("type", "<i4"), ("player", "<i4"), ("x", "<i4"), ("y", "<i4"),
    ("hit_points", "<i4"), ("shields", "<i4"), ("energy", "<i4"), ("resources", "<i4"), ("resource_group", "<i4"),
    ("flags", "<u4"), ("visible_mask", "<u4"),
    ("order", "<i4"), ("order_target", "<i4"), ("order_target_x", "<i4"), ("order_target_y", "<i4"),
    ("secondary_order", "<i4"),
    ("target", "<i4"), ("target_x", "<i4"), ("target_y", "<i4"),
    ("build_type", "<i4"), ("build_unit", "<i4"), ("remaining_build_time", "<i4"), ("remaining_train_time", "<i4"),
    ("train_queue_count", "<i4"), ("train_queue_0", "<i4"), ("train_queue_1", "<i4"), ("train_queue_2", "<i4"),
    ("train_queue_3", "<i4"), ("train_queue_4", "<i4"),
    ("tech", "<i4"), ("upgrade", "<i4"), ("remaining_research_time", "<i4"), ("remaining_upgrade_time", "<i4"),
    ("ground_weapon_cooldown", "<i4"), ("air_weapon_cooldown", "<i4"), ("spell_cooldown", "<i4"),
    ("angle", "<f4"), ("velocity_x", "<f4"), ("velocity_y", "<f4"),
    ("kill_count", "<i4"), ("acid_spore_count", "<i4"), ("interceptor_count", "<i4"), ("scarab_count", "<i4"),
    ("spider_mine_count", "<i4"),
    ("defense_matrix_points", "<i4"), ("defense_matrix_timer", "<i4"), ("ensnare_timer", "<i4"),
    ("irradiate_timer", "<i4"), ("lockdown_timer", "<i4"), ("maelstrom_timer", "<i4"), ("order_timer", "<i4"),
    ("plague_timer", "<i4"), ("remove_timer", "<i4"), ("stasis_timer", "<i4"), ("stim_timer", "<i4"),
    ("addon", "<i4"), ("transport", "<i4"), ("carrier", "<i4"), ("hatchery", "<i4"), ("nydus_exit", "<i4"),
    ("power_up", "<i4"),
    ("rally_x", "<i4"), ("rally_y", "<i4"), ("rally_unit", "<i4"),
    ("carry_resource_type", "<i4"), ("last_attacker_player", "<i4"), ("last_hit_points", "<i4"), ("replay_id", "<i4"),
]
UNIT_DTYPE = np.dtype(UNIT_FIELDS)
assert UNIT_DTYPE.itemsize == 276, UNIT_DTYPE.itemsize  # flatc-reported struct size

BULLET_DTYPE = np.dtype([
    ("id", "<i4"), ("type", "<i4"), ("player", "<i4"), ("source", "<i4"), ("target", "<i4"),
    ("x", "<i4"), ("y", "<i4"), ("target_x", "<i4"), ("target_y", "<i4"), ("remove_timer", "<i4"),
    ("angle", "<f4"), ("velocity_x", "<f4"), ("velocity_y", "<f4"), ("exists_visible", "<i4"),
])
assert BULLET_DTYPE.itemsize == 56

POS_DTYPE = np.dtype([("x", "<i4"), ("y", "<i4")])


class UnitFlag(IntFlag):
    """Bits of UnitState.flags (mirror of bw.fbs UnitFlag)."""

    Exists = 1 << 0
    Completed = 1 << 1
    Idle = 1 << 2
    Moving = 1 << 3
    Attacking = 1 << 4
    AttackFrame = 1 << 5
    StartingAttack = 1 << 6
    Gathering = 1 << 7
    BeingGathered = 1 << 8
    Constructing = 1 << 9
    Training = 1 << 10
    Morphing = 1 << 11
    Burrowed = 1 << 12
    Cloaked = 1 << 13
    Detected = 1 << 14
    Lifted = 1 << 15
    Sieged = 1 << 16
    Accelerating = 1 << 17
    Braking = 1 << 18
    Blind = 1 << 19
    Hallucination = 1 << 20
    Interruptible = 1 << 21
    Invincible = 1 << 22
    Parasited = 1 << 23
    Selected = 1 << 24
    Stuck = 1 << 25
    UnderStorm = 1 << 26
    UnderDarkSwarm = 1 << 27
    UnderDWeb = 1 << 28
    Powered = 1 << 29
    HasNuke = 1 << 30
    RecentlyAttacked = 1 << 31


class TileFlag(IntFlag):
    Visible = 1
    Explored = 2
    Creep = 4


# ---------------------------------------------------------------------------
# zero-copy helpers
# ---------------------------------------------------------------------------

def _struct_vector(tab, voffset: int, dtype: np.dtype) -> np.ndarray:
    """View a FlatBuffers vector of structs at vtable offset `voffset` as a numpy array."""
    o = tab.Offset(voffset)
    if o == 0:
        return np.empty(0, dtype=dtype)
    start = tab.Vector(o)
    n = tab.VectorLen(o)
    return np.frombuffer(tab.Bytes, dtype=dtype, count=n, offset=start)


def _scalar_vector(tab, voffset: int, dtype: np.dtype) -> np.ndarray:
    o = tab.Offset(voffset)
    if o == 0:
        return np.empty(0, dtype=dtype)
    start = tab.Vector(o)
    n = tab.VectorLen(o)
    return np.frombuffer(tab.Bytes, dtype=dtype, count=n, offset=start)


def _s(b: Optional[bytes]) -> str:
    return b.decode("utf-8", "replace") if b else ""


# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

UNIT_TYPE_DTYPE = np.dtype([
    ("id", "<i4"), ("race", "<i4"), ("mineral_price", "<i4"), ("gas_price", "<i4"), ("build_time", "<i4"),
    ("supply_required", "<i4"), ("supply_provided", "<i4"), ("max_hit_points", "<i4"), ("max_shields", "<i4"),
    ("max_energy", "<i4"), ("armor", "<i4"), ("ground_weapon", "<i4"), ("air_weapon", "<i4"),
    ("max_ground_hits", "<i4"), ("max_air_hits", "<i4"), ("sight_range", "<i4"), ("seek_range", "<i4"),
    ("top_speed", "<f4"), ("acceleration", "<i4"), ("tile_width", "<i4"), ("tile_height", "<i4"),
    ("dimension_left", "<i4"), ("dimension_up", "<i4"), ("dimension_right", "<i4"), ("dimension_down", "<i4"),
    ("size", "<i4"), ("what_builds", "<i4"), ("what_builds_count", "<i4"), ("required_tech", "<i4"),
    ("space_required", "<i4"), ("space_provided", "<i4"), ("flags", "<u4"),
])


class UnitTypeFlag(IntFlag):
    Building = 1 << 0
    Worker = 1 << 1
    Flyer = 1 << 2
    Organic = 1 << 3
    Mechanical = 1 << 4
    Robotic = 1 << 5
    Detector = 1 << 6
    ResourceContainer = 1 << 7
    ResourceDepot = 1 << 8
    Refinery = 1 << 9
    Addon = 1 << 10
    FlyingBuilding = 1 << 11
    Spell = 1 << 12
    Invincible = 1 << 13
    Burrowable = 1 << 14
    Cloakable = 1 << 15
    Hero = 1 << 16
    Powerup = 1 << 17
    Beacon = 1 << 18
    Critter = 1 << 19
    Neutral = 1 << 20
    CanProduce = 1 << 21
    CanAttack = 1 << 22
    CanMove = 1 << 23
    RegeneratesHP = 1 << 24
    IsMineralField = 1 << 25
    IsSpecialBuilding = 1 << 26
    ProducesLarva = 1 << 27
    RequiresPsi = 1 << 28
    RequiresCreep = 1 << 29
    TwoUnitsInOneEgg = 1 << 30


@dataclass
class PlayerInfo:
    id: int
    name: str
    race: int
    type: int
    force: int
    is_self: bool
    is_enemy: bool
    is_ally: bool
    is_neutral: bool
    is_observer: bool
    start_location: tuple[int, int]
    color: int

    @classmethod
    def from_fb(cls, p: bw.PlayerInfo) -> "PlayerInfo":
        sl = p.StartLocation()
        return cls(
            p.Id(), _s(p.Name()), p.Race(), p.Type(), p.Force(), p.IsSelf(), p.IsEnemy(), p.IsAlly(),
            p.IsNeutral(), p.IsObserver(), (sl.X(), sl.Y()) if sl else (-1, -1), p.Color(),
        )


@dataclass
class GameInfo:
    """Static per-match data from GameStart. Tile arrays are indexed [y, x]."""

    map_name: str
    map_file_name: str
    map_path_name: str
    map_hash: str
    map_width: int
    map_height: int
    ground_height: np.ndarray      # (h, w) uint8
    buildable: np.ndarray          # (h, w) bool
    walkable: np.ndarray           # (4h, 4w) bool  (walk tiles, 8px)
    region_id: np.ndarray          # (h, w) uint16
    start_locations: np.ndarray    # (n,) POS_DTYPE, tile coords
    players: list[PlayerInfo]
    self_id: int
    enemy_id: int
    neutral_id: int
    game_type: int
    latency_frames: int
    random_seed: int
    is_replay: bool
    frame_skip: int
    unit_types: np.ndarray         # (UnitType.MAX,) UNIT_TYPE_DTYPE indexed by type id
    unit_type_names: list[str]
    unit_type_required_units: list[list[int]]
    weapon_types: list[dict]
    upgrade_types: list[dict]
    tech_types: list[dict]
    _by_id: dict[int, PlayerInfo] = field(default_factory=dict, repr=False)

    @property
    def self_player(self) -> PlayerInfo:
        return self._by_id[self.self_id]

    @property
    def self_race(self) -> Race:
        return Race(self.self_player.race)

    @property
    def enemies(self) -> list[PlayerInfo]:
        return [p for p in self.players if p.is_enemy]

    def player(self, pid: int) -> Optional[PlayerInfo]:
        return self._by_id.get(pid)

    def type_name(self, type_id: int) -> str:
        return self.unit_type_names[type_id] if 0 <= type_id < len(self.unit_type_names) else f"type{type_id}"

    def type_flags(self, type_id: int) -> UnitTypeFlag:
        return UnitTypeFlag(int(self.unit_types["flags"][type_id]))

    @classmethod
    def from_fb(cls, g: bw.GameStart) -> "GameInfo":
        w, h = g.MapWidth(), g.MapHeight()
        tab = g._tab
        ground = _scalar_vector(tab, 16, np.uint8).reshape(h, w)
        buildable = _scalar_vector(tab, 18, np.uint8).reshape(h, w).astype(bool)
        walkable = _scalar_vector(tab, 20, np.uint8).reshape(h * 4, w * 4).astype(bool)
        region = _scalar_vector(tab, 22, np.uint16).reshape(h, w)
        starts = _struct_vector(tab, 24, POS_DTYPE).copy()
        players = [PlayerInfo.from_fb(g.Players(i)) for i in range(g.PlayersLength())]

        n = g.UnitTypesLength()
        ut = np.zeros(max(n, int(UnitType.MAX)), dtype=UNIT_TYPE_DTYPE)
        names: list[str] = [""] * len(ut)
        req: list[list[int]] = [[] for _ in range(len(ut))]
        for i in range(n):
            t = g.UnitTypes(i)
            tid = t.Id()
            if not (0 <= tid < len(ut)):
                continue
            ut[tid] = (
                tid, t.Race(), t.MineralPrice(), t.GasPrice(), t.BuildTime(), t.SupplyRequired(), t.SupplyProvided(),
                t.MaxHitPoints(), t.MaxShields(), t.MaxEnergy(), t.Armor(), t.GroundWeapon(), t.AirWeapon(),
                t.MaxGroundHits(), t.MaxAirHits(), t.SightRange(), t.SeekRange(), t.TopSpeed(), t.Acceleration(),
                t.TileWidth(), t.TileHeight(), t.DimensionLeft(), t.DimensionUp(), t.DimensionRight(),
                t.DimensionDown(), t.Size(), t.WhatBuilds(), t.WhatBuildsCount(), t.RequiredTech(),
                t.SpaceRequired(), t.SpaceProvided(), t.Flags(),
            )
            names[tid] = _s(t.Name())
            req[tid] = [int(t.RequiredUnits(j)) for j in range(t.RequiredUnitsLength())]

        def weapon(x: bw.WeaponTypeInfo) -> dict:
            return dict(
                id=x.Id(), name=_s(x.Name()), tech=x.Tech(), what_uses=x.WhatUses(), damage_amount=x.DamageAmount(),
                damage_bonus=x.DamageBonus(), damage_cooldown=x.DamageCooldown(), damage_factor=x.DamageFactor(),
                upgrade_type=x.UpgradeType(), damage_type=x.DamageType(), explosion_type=x.ExplosionType(),
                min_range=x.MinRange(), max_range=x.MaxRange(), inner_splash_radius=x.InnerSplashRadius(),
                median_splash_radius=x.MedianSplashRadius(), outer_splash_radius=x.OuterSplashRadius(),
                targets_air=x.TargetsAir(), targets_ground=x.TargetsGround(),
            )

        def upgrade(x: bw.UpgradeTypeInfo) -> dict:
            return dict(
                id=x.Id(), name=_s(x.Name()), race=x.Race(), mineral_price=x.MineralPrice(),
                mineral_price_factor=x.MineralPriceFactor(), gas_price=x.GasPrice(), gas_price_factor=x.GasPriceFactor(),
                upgrade_time=x.UpgradeTime(), upgrade_time_factor=x.UpgradeTimeFactor(), max_repeats=x.MaxRepeats(),
                what_upgrades=x.WhatUpgrades(),
            )

        def tech(x: bw.TechTypeInfo) -> dict:
            return dict(
                id=x.Id(), name=_s(x.Name()), race=x.Race(), mineral_price=x.MineralPrice(), gas_price=x.GasPrice(),
                research_time=x.ResearchTime(), energy_cost=x.EnergyCost(), what_researches=x.WhatResearches(),
                weapon=x.Weapon(), targets_unit=x.TargetsUnit(), targets_position=x.TargetsPosition(),
            )

        info = cls(
            map_name=_s(g.MapName()), map_file_name=_s(g.MapFileName()), map_path_name=_s(g.MapPathName()),
            map_hash=_s(g.MapHash()), map_width=w, map_height=h,
            ground_height=ground.copy(), buildable=buildable, walkable=walkable, region_id=region.copy(),
            start_locations=starts, players=players, self_id=g.SelfId(), enemy_id=g.EnemyId(),
            neutral_id=g.NeutralId(), game_type=g.GameType(), latency_frames=g.LatencyFrames(),
            random_seed=g.RandomSeed(), is_replay=g.IsReplay(), frame_skip=g.FrameSkip(),
            unit_types=ut, unit_type_names=names, unit_type_required_units=req,
            weapon_types=[weapon(g.WeaponTypes(i)) for i in range(g.WeaponTypesLength())],
            upgrade_types=[upgrade(g.UpgradeTypes(i)) for i in range(g.UpgradeTypesLength())],
            tech_types=[tech(g.TechTypes(i)) for i in range(g.TechTypesLength())],
        )
        info._by_id = {p.id: p for p in players}
        return info


# ---------------------------------------------------------------------------
# Per-frame data
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    id: int
    minerals: int
    gas: int
    gathered_minerals: int
    gathered_gas: int
    supply_used: int          # BWAPI units: 2 per displayed supply
    supply_total: int
    is_victorious: bool
    is_defeated: bool
    left_game: bool
    all_unit_count: np.ndarray        # indexed by UnitType (self only; empty otherwise)
    completed_unit_count: np.ndarray
    dead_unit_count: np.ndarray
    killed_unit_count: np.ndarray
    upgrade_level: np.ndarray         # indexed by UpgradeType
    has_researched: np.ndarray        # indexed by TechType (uint8)
    is_researching: np.ndarray
    is_upgrading: np.ndarray
    unit_score: int
    kill_score: int
    building_score: int
    razing_score: int
    custom_score: int

    @property
    def supply_used_display(self) -> int:
        return self.supply_used // 2

    @property
    def supply_total_display(self) -> int:
        return self.supply_total // 2

    @classmethod
    def from_fb(cls, p: bw.PlayerState) -> "PlayerState":
        tab = p._tab
        return cls(
            p.Id(), p.Minerals(), p.Gas(), p.GatheredMinerals(), p.GatheredGas(), p.SupplyUsed(), p.SupplyTotal(),
            p.IsVictorious(), p.IsDefeated(), p.LeftGame(),
            _scalar_vector(tab, 28, np.int32), _scalar_vector(tab, 30, np.int32), _scalar_vector(tab, 32, np.int32),
            _scalar_vector(tab, 34, np.int32), _scalar_vector(tab, 36, np.int32), _scalar_vector(tab, 38, np.uint8),
            _scalar_vector(tab, 40, np.uint8), _scalar_vector(tab, 42, np.uint8),
            p.UnitScore(), p.KillScore(), p.BuildingScore(), p.RazingScore(), p.CustomScore(),
        )


@dataclass
class Event:
    type: EventType | int
    unit: int
    player: int
    is_winner: bool
    text: str
    position: tuple[int, int]

    @classmethod
    def from_fb(cls, e: bw.Event) -> "Event":
        pos = e.Position()
        return cls(EventType.get(e.Type()), e.Unit(), e.Player(), e.IsWinner(), _s(e.Text()),
                   (pos.X(), pos.Y()) if pos else (0, 0))


class Observation:
    """One game frame. `units` is a numpy structured array (UNIT_DTYPE) viewing the receive buffer;
    it is only valid until the next frame is received - copy it if you keep it."""

    __slots__ = (
        "game", "frame", "frame_count", "elapsed_time", "fps", "average_fps", "latency_frames",
        "remaining_latency_frames", "is_paused", "self_id", "units", "bullets", "_players", "_events", "_tiles",
        "nuke_dots", "serialize_us", "last_roundtrip_us", "_by_id",
    )

    def __init__(self, game: GameInfo, frame: bw.Frame):
        self.game = game
        self.frame = frame
        self.frame_count = frame.FrameCount()
        self.elapsed_time = frame.ElapsedTime()
        self.fps = frame.Fps()
        self.average_fps = frame.AverageFps()
        self.latency_frames = frame.LatencyFrames()
        self.remaining_latency_frames = frame.RemainingLatencyFrames()
        self.is_paused = frame.IsPaused()
        self.self_id = frame.SelfId()
        tab = frame._tab
        self.units: np.ndarray = _struct_vector(tab, 24, UNIT_DTYPE)
        self.bullets: np.ndarray = _struct_vector(tab, 26, BULLET_DTYPE)
        self.nuke_dots: np.ndarray = _struct_vector(tab, 32, POS_DTYPE)
        self.serialize_us = frame.SerializeUs()
        self.last_roundtrip_us = frame.LastRoundtripUs()
        self._players: Optional[dict[int, PlayerState]] = None
        self._events: Optional[list[Event]] = None
        self._tiles: Optional[np.ndarray] = None
        self._by_id: Optional[dict[int, int]] = None

    # --- players ---------------------------------------------------------
    @property
    def players(self) -> dict[int, PlayerState]:
        if self._players is None:
            self._players = {}
            for i in range(self.frame.PlayersLength()):
                ps = PlayerState.from_fb(self.frame.Players(i))
                self._players[ps.id] = ps
        return self._players

    @property
    def me(self) -> PlayerState:
        return self.players[self.self_id]

    @property
    def minerals(self) -> int:
        return self.me.minerals

    @property
    def gas(self) -> int:
        return self.me.gas

    @property
    def supply_used(self) -> int:
        """Displayed supply (BWAPI value / 2)."""
        return self.me.supply_used // 2

    @property
    def supply_total(self) -> int:
        return self.me.supply_total // 2

    # --- events / tiles --------------------------------------------------
    @property
    def events(self) -> list[Event]:
        if self._events is None:
            self._events = [Event.from_fb(self.frame.Events(i)) for i in range(self.frame.EventsLength())]
        return self._events

    @property
    def tiles(self) -> np.ndarray:
        """(h, w) uint8 TileFlag bitmap; empty if include_tiles is off."""
        if self._tiles is None:
            t = _scalar_vector(self.frame._tab, 30, np.uint8)
            g = self.game
            self._tiles = t.reshape(g.map_height, g.map_width) if t.size == g.map_width * g.map_height else t
        return self._tiles

    @property
    def visible(self) -> np.ndarray:
        return (self.tiles & TileFlag.Visible) != 0

    @property
    def explored(self) -> np.ndarray:
        return (self.tiles & TileFlag.Explored) != 0

    # --- unit filters (all return views/subsets of self.units) ---------
    @property
    def my_units(self) -> np.ndarray:
        return self.units[self.units["player"] == self.self_id]

    @property
    def enemy_units(self) -> np.ndarray:
        enemy_ids = [p.id for p in self.game.enemies]
        return self.units[np.isin(self.units["player"], enemy_ids)]

    @property
    def neutral_units(self) -> np.ndarray:
        return self.units[self.units["player"] == self.game.neutral_id]

    @property
    def minerals_fields(self) -> np.ndarray:
        flags = self.game.unit_types["flags"][np.clip(self.units["type"], 0, len(self.game.unit_types) - 1)]
        return self.units[(flags & UnitTypeFlag.IsMineralField) != 0]

    @property
    def geysers(self) -> np.ndarray:
        return self.units[self.units["type"] == UnitType.Resource_Vespene_Geyser]

    def units_of_type(self, type_id: int, player: Optional[int] = None) -> np.ndarray:
        m = self.units["type"] == int(type_id)
        if player is not None:
            m &= self.units["player"] == player
        return self.units[m]

    def my_units_of_type(self, type_id: int, completed_only: bool = False) -> np.ndarray:
        u = self.units_of_type(type_id, self.self_id)
        if completed_only:
            u = u[(u["flags"] & UnitFlag.Completed) != 0]
        return u

    def my_completed(self, type_id: int) -> np.ndarray:
        return self.my_units_of_type(type_id, completed_only=True)

    def count(self, type_id: int, completed_only: bool = False) -> int:
        """Own units of a type (uses the per-type counters in PlayerState, includes in-production)."""
        me = self.me
        arr = me.completed_unit_count if completed_only else me.all_unit_count
        return int(arr[int(type_id)]) if arr.size > int(type_id) else len(self.my_units_of_type(type_id, completed_only))

    def unit(self, unit_id: int) -> Optional[np.void]:
        if self._by_id is None:
            self._by_id = {int(i): k for k, i in enumerate(self.units["id"])}
        k = self._by_id.get(int(unit_id))
        return self.units[k] if k is not None else None

    def has_flag(self, u: np.void, flag: UnitFlag) -> bool:
        return bool(int(u["flags"]) & flag)

    def idle(self, units: np.ndarray) -> np.ndarray:
        return units[(units["flags"] & UnitFlag.Idle) != 0]

    def completed(self, units: np.ndarray) -> np.ndarray:
        return units[(units["flags"] & UnitFlag.Completed) != 0]

    @staticmethod
    def positions(units: np.ndarray) -> np.ndarray:
        """(n, 2) float array of pixel positions."""
        return np.stack([units["x"], units["y"]], axis=1).astype(np.float32)

    def nearest(self, units: np.ndarray, x: float, y: float) -> Optional[np.void]:
        if len(units) == 0:
            return None
        d = (units["x"] - x) ** 2 + (units["y"] - y) ** 2
        return units[int(np.argmin(d))]

    def iter_events(self, *types: EventType) -> Iterator[Event]:
        for e in self.events:
            if not types or e.type in types:
                yield e

    def __repr__(self) -> str:
        return (f"Observation(frame={self.frame_count}, units={len(self.units)}, "
                f"minerals={self.minerals if self.frame.PlayersLength() else '?'})")
