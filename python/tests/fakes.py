"""Synthetic GameInfo / Observation for unit tests (no shim, no StarCraft).

Unit data is hand-filled for the types the bots use; values follow BWAPI (supply in BWAPI units,
build times in frames). Map: 128x128 tiles, two start bases with naturals, one extra base each side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from bwbot.enums import Race, TechType, UnitType as U, UpgradeType, WeaponType as W
from bwbot.observation import (
    UNIT_DTYPE, UNIT_TYPE_DTYPE, Event, GameInfo, MapArea, MapBase, MapChoke, Observation, PlayerInfo, PlayerState,
    StartBase, UnitFlag, UnitTypeFlag as F,
)

N_TYPES = 234
B, WK, FL, ORG, MECH = F.Building, F.Worker, F.Flyer, F.Organic, F.Mechanical
ATK, MOV, PROD, DEPOT, REF, ADDON, DET = F.CanAttack, F.CanMove, F.CanProduce, F.ResourceDepot, F.Refinery, F.Addon, F.Detector

# type: (race, m, g, time, supply_req, supply_prov, hp, shields, armor, gw, aw, range_tiles, size, builder, bcount,
#        tile_w, tile_h, flags, required)
_T = Race.Terran
_Z = Race.Zerg
_P = Race.Protoss
NONE_W = int(W.None_)
UNITS: dict[int, tuple] = {
    U.Terran_SCV: (_T, 50, 0, 300, 2, 0, 60, 0, 0, W.Fusion_Cutter, NONE_W, 1, 1, U.Terran_Command_Center, 1, 1, 1, WK | ORG | MECH | ATK | MOV, []),
    U.Terran_Marine: (_T, 50, 0, 360, 2, 0, 40, 0, 0, W.Gauss_Rifle, W.Gauss_Rifle, 1, 1, U.Terran_Barracks, 1, 1, 1, ORG | ATK | MOV, []),
    U.Terran_Medic: (_T, 50, 25, 450, 2, 0, 60, 0, 1, NONE_W, NONE_W, 1, 1, U.Terran_Barracks, 1, 1, 1, ORG | MOV, [U.Terran_Academy]),
    U.Terran_Firebat: (_T, 50, 25, 360, 2, 0, 50, 0, 1, W.Flame_Thrower, NONE_W, 1, 1, U.Terran_Barracks, 1, 1, 1, ORG | ATK | MOV, [U.Terran_Academy]),
    U.Terran_Vulture: (_T, 75, 0, 450, 4, 0, 80, 0, 0, W.Fragmentation_Grenade, NONE_W, 2, 2, U.Terran_Factory, 1, 1, 1, MECH | ATK | MOV, []),
    U.Terran_Goliath: (_T, 100, 50, 600, 4, 0, 125, 0, 1, W.Twin_Autocannons, W.Hellfire_Missile_Pack, 2, 3, U.Terran_Factory, 1, 1, 1, MECH | ATK | MOV, [U.Terran_Armory]),
    U.Terran_Siege_Tank_Tank_Mode: (_T, 150, 100, 750, 4, 0, 150, 0, 1, W.Arclite_Cannon, NONE_W, 2, 3, U.Terran_Factory, 1, 1, 1, MECH | ATK | MOV, [U.Terran_Machine_Shop]),
    U.Terran_Command_Center: (_T, 400, 0, 1800, 0, 20, 1500, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 4, 3, B | MECH | PROD | DEPOT, []),
    U.Terran_Supply_Depot: (_T, 100, 0, 600, 0, 16, 500, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 3, 2, B | MECH, []),
    U.Terran_Refinery: (_T, 100, 0, 600, 0, 0, 750, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 4, 2, B | MECH | REF, []),
    U.Terran_Barracks: (_T, 150, 0, 1200, 0, 0, 1000, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 4, 3, B | MECH | PROD, [U.Terran_Command_Center]),
    U.Terran_Academy: (_T, 150, 0, 1200, 0, 0, 600, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 3, 2, B | MECH, [U.Terran_Barracks]),
    U.Terran_Factory: (_T, 200, 100, 1200, 0, 0, 1250, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 4, 3, B | MECH | PROD, [U.Terran_Barracks]),
    U.Terran_Starport: (_T, 150, 100, 1050, 0, 0, 1300, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 4, 3, B | MECH | PROD, [U.Terran_Factory]),
    U.Terran_Machine_Shop: (_T, 50, 50, 600, 0, 0, 750, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_Factory, 1, 2, 2, B | MECH | ADDON, []),
    U.Terran_Comsat_Station: (_T, 50, 50, 600, 0, 0, 500, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_Command_Center, 1, 2, 2, B | MECH | ADDON, [U.Terran_Academy]),
    U.Terran_Engineering_Bay: (_T, 125, 0, 900, 0, 0, 850, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 4, 3, B | MECH, [U.Terran_Command_Center]),
    U.Terran_Armory: (_T, 100, 50, 1200, 0, 0, 750, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 3, 2, B | MECH, [U.Terran_Factory]),
    U.Terran_Missile_Turret: (_T, 75, 0, 450, 0, 0, 200, 0, 0, NONE_W, W.Longbolt_Missile, 7, 3, U.Terran_SCV, 1, 2, 2, B | MECH | ATK | DET, [U.Terran_Engineering_Bay]),
    U.Terran_Bunker: (_T, 100, 0, 450, 0, 0, 350, 0, 1, NONE_W, NONE_W, 0, 3, U.Terran_SCV, 1, 3, 2, B | MECH, [U.Terran_Barracks]),
    # zerg
    U.Zerg_Larva: (_Z, 0, 0, 0, 0, 0, 10, 0, 10, NONE_W, NONE_W, 0, 1, U.Zerg_Hatchery, 1, 1, 1, ORG, []),
    U.Zerg_Drone: (_Z, 50, 0, 300, 2, 0, 40, 0, 0, W.Spines, NONE_W, 1, 1, U.Zerg_Larva, 1, 1, 1, WK | ORG | ATK | MOV, []),
    U.Zerg_Overlord: (_Z, 100, 0, 600, 0, 16, 200, 0, 0, NONE_W, NONE_W, 0, 3, U.Zerg_Larva, 1, 2, 2, FL | ORG | MOV | DET, []),
    U.Zerg_Zergling: (_Z, 50, 0, 420, 1, 0, 35, 0, 0, W.Claws, NONE_W, 0, 1, U.Zerg_Larva, 1, 1, 1, ORG | ATK | MOV, [U.Zerg_Spawning_Pool]),
    U.Zerg_Hydralisk: (_Z, 75, 25, 420, 2, 0, 80, 0, 0, W.Needle_Spines, W.Needle_Spines, 4, 2, U.Zerg_Larva, 1, 1, 1, ORG | ATK | MOV, [U.Zerg_Hydralisk_Den]),
    U.Zerg_Mutalisk: (_Z, 100, 100, 600, 4, 0, 120, 0, 0, W.Glave_Wurm, W.Glave_Wurm, 3, 1, U.Zerg_Larva, 1, 2, 2, FL | ORG | ATK | MOV, [U.Zerg_Spire]),
    U.Zerg_Hatchery: (_Z, 300, 0, 1800, 0, 2, 1250, 0, 1, NONE_W, NONE_W, 0, 3, U.Zerg_Drone, 1, 4, 3, B | ORG | PROD | DEPOT, []),
    U.Zerg_Lair: (_Z, 150, 100, 1500, 0, 2, 1800, 0, 1, NONE_W, NONE_W, 0, 3, U.Zerg_Hatchery, 1, 4, 3, B | ORG | PROD | DEPOT, [U.Zerg_Spawning_Pool]),
    U.Zerg_Spawning_Pool: (_Z, 200, 0, 1200, 0, 0, 750, 0, 1, NONE_W, NONE_W, 0, 3, U.Zerg_Drone, 1, 3, 2, B | ORG, [U.Zerg_Hatchery]),
    U.Zerg_Hydralisk_Den: (_Z, 100, 50, 600, 0, 0, 850, 0, 1, NONE_W, NONE_W, 0, 3, U.Zerg_Drone, 1, 3, 2, B | ORG, [U.Zerg_Spawning_Pool]),
    U.Zerg_Spire: (_Z, 200, 150, 1800, 0, 0, 600, 0, 1, NONE_W, NONE_W, 0, 3, U.Zerg_Drone, 1, 2, 2, B | ORG, [U.Zerg_Lair]),
    U.Zerg_Extractor: (_Z, 50, 0, 600, 0, 0, 750, 0, 1, NONE_W, NONE_W, 0, 3, U.Zerg_Drone, 1, 4, 2, B | ORG | REF, []),
    U.Zerg_Creep_Colony: (_Z, 75, 0, 300, 0, 0, 400, 0, 0, NONE_W, NONE_W, 0, 3, U.Zerg_Drone, 1, 2, 2, B | ORG, []),
    U.Zerg_Sunken_Colony: (_Z, 50, 0, 300, 0, 0, 300, 0, 2, W.Subterranean_Tentacle, NONE_W, 7, 3, U.Zerg_Creep_Colony, 1, 2, 2, B | ORG | ATK, [U.Zerg_Spawning_Pool]),
    # protoss
    U.Protoss_Probe: (_P, 50, 0, 300, 2, 0, 20, 20, 0, W.Particle_Beam, NONE_W, 1, 1, U.Protoss_Nexus, 1, 1, 1, WK | MECH | ATK | MOV, []),
    U.Protoss_Zealot: (_P, 100, 0, 600, 4, 0, 100, 60, 1, W.Psi_Blades, NONE_W, 0, 1, U.Protoss_Gateway, 1, 1, 1, ORG | ATK | MOV, []),
    U.Protoss_Dragoon: (_P, 125, 50, 750, 4, 0, 100, 80, 1, W.Phase_Disruptor, W.Phase_Disruptor, 4, 3, U.Protoss_Gateway, 1, 1, 1, MECH | ATK | MOV, [U.Protoss_Cybernetics_Core]),
    U.Protoss_Nexus: (_P, 400, 0, 1800, 0, 18, 750, 750, 1, NONE_W, NONE_W, 0, 3, U.Protoss_Probe, 1, 4, 3, B | MECH | PROD | DEPOT, []),
    U.Protoss_Pylon: (_P, 100, 0, 450, 0, 16, 300, 300, 0, NONE_W, NONE_W, 0, 3, U.Protoss_Probe, 1, 2, 2, B | MECH, []),
    U.Protoss_Assimilator: (_P, 100, 0, 600, 0, 0, 450, 450, 1, NONE_W, NONE_W, 0, 3, U.Protoss_Probe, 1, 4, 2, B | MECH | REF, []),
    U.Protoss_Gateway: (_P, 150, 0, 900, 0, 0, 500, 500, 1, NONE_W, NONE_W, 0, 3, U.Protoss_Probe, 1, 4, 3, B | MECH | PROD, [U.Protoss_Nexus]),
    U.Protoss_Cybernetics_Core: (_P, 200, 0, 900, 0, 0, 500, 500, 1, NONE_W, NONE_W, 0, 3, U.Protoss_Probe, 1, 3, 2, B | MECH, [U.Protoss_Gateway]),
    U.Protoss_Forge: (_P, 150, 0, 600, 0, 0, 550, 550, 1, NONE_W, NONE_W, 0, 3, U.Protoss_Probe, 1, 3, 2, B | MECH, [U.Protoss_Nexus]),
    U.Protoss_Photon_Cannon: (_P, 150, 0, 750, 0, 0, 100, 100, 0, W.STS_Photon_Cannon, W.STA_Photon_Cannon, 7, 3, U.Protoss_Probe, 1, 2, 2, B | MECH | ATK | DET, [U.Protoss_Forge]),
    # neutral
    U.Resource_Mineral_Field: (Race.None_, 0, 0, 0, 0, 0, 100000, 0, 0, NONE_W, NONE_W, 0, 0, NONE_W, 0, 2, 1, F.ResourceContainer | F.IsMineralField | F.Neutral, []),
    U.Resource_Vespene_Geyser: (Race.None_, 0, 0, 0, 0, 0, 100000, 0, 0, NONE_W, NONE_W, 0, 0, NONE_W, 0, 4, 2, F.ResourceContainer | F.Neutral, []),
}

# id: (damage, bonus, cooldown, factor, damage_type, min_range, max_range px, splash, air, ground, upgrade)
# damage types: 1 explosive, 2 concussive, 3 normal
WEAPONS: dict[int, tuple] = {
    W.Gauss_Rifle: (6, 1, 15, 1, 3, 0, 128, 0, True, True, UpgradeType.Terran_Infantry_Weapons),
    W.Flame_Thrower: (8, 1, 22, 1, 2, 0, 32, 15, False, True, UpgradeType.Terran_Infantry_Weapons),
    W.Fragmentation_Grenade: (20, 2, 30, 1, 2, 0, 160, 0, False, True, UpgradeType.Terran_Vehicle_Weapons),
    W.Twin_Autocannons: (12, 1, 22, 1, 3, 0, 192, 0, False, True, UpgradeType.Terran_Vehicle_Weapons),
    W.Hellfire_Missile_Pack: (10, 2, 22, 2, 1, 0, 160, 0, True, False, UpgradeType.Terran_Vehicle_Weapons),
    W.Arclite_Cannon: (30, 3, 37, 1, 1, 0, 224, 0, False, True, UpgradeType.Terran_Vehicle_Weapons),
    W.Fusion_Cutter: (5, 1, 15, 1, 3, 0, 10, 0, False, True, UpgradeType.None_),
    W.Longbolt_Missile: (20, 0, 15, 1, 1, 0, 224, 0, True, False, UpgradeType.None_),
    W.Spines: (5, 0, 22, 1, 3, 0, 32, 0, False, True, UpgradeType.None_),
    W.Claws: (5, 1, 8, 1, 3, 0, 15, 0, False, True, UpgradeType.Zerg_Melee_Attacks),
    W.Needle_Spines: (10, 1, 15, 1, 1, 0, 128, 0, True, True, UpgradeType.Zerg_Missile_Attacks),
    W.Glave_Wurm: (9, 1, 30, 1, 3, 0, 96, 0, True, True, UpgradeType.Zerg_Flyer_Attacks),
    W.Subterranean_Tentacle: (40, 0, 32, 1, 1, 0, 224, 0, False, True, UpgradeType.None_),
    W.Particle_Beam: (5, 0, 22, 1, 3, 0, 32, 0, False, True, UpgradeType.None_),
    W.Psi_Blades: (8, 1, 22, 2, 3, 0, 15, 0, False, True, UpgradeType.Protoss_Ground_Weapons),
    W.Phase_Disruptor: (20, 2, 30, 1, 1, 0, 128, 0, True, True, UpgradeType.Protoss_Ground_Weapons),
    W.STS_Photon_Cannon: (20, 0, 22, 1, 3, 0, 224, 0, False, True, UpgradeType.None_),
    W.STA_Photon_Cannon: (20, 0, 22, 1, 3, 0, 224, 0, True, False, UpgradeType.None_),
}

UPGRADES = {
    UpgradeType.Terran_Infantry_Weapons: (100, 75, 100, 75, 4000, 480, 3, U.Terran_Engineering_Bay),
    UpgradeType.Terran_Infantry_Armor: (100, 75, 100, 75, 4000, 480, 3, U.Terran_Engineering_Bay),
    UpgradeType.Terran_Vehicle_Weapons: (100, 100, 75, 75, 4000, 480, 3, U.Terran_Armory),
    UpgradeType.Terran_Vehicle_Plating: (100, 100, 75, 75, 4000, 480, 3, U.Terran_Armory),
    UpgradeType.U_238_Shells: (150, 150, 0, 0, 1500, 0, 1, U.Terran_Academy),
    UpgradeType.Ion_Thrusters: (100, 100, 0, 0, 1500, 0, 1, U.Terran_Machine_Shop),
    UpgradeType.Charon_Boosters: (100, 100, 0, 0, 2000, 0, 1, U.Terran_Machine_Shop),
}

TECHS = {
    TechType.Stim_Packs: (100, 100, 1200, 0, U.Terran_Academy),
    TechType.Spider_Mines: (100, 100, 1200, 0, U.Terran_Machine_Shop),
    TechType.Tank_Siege_Mode: (150, 150, 1200, 0, U.Terran_Machine_Shop),
    TechType.Scanner_Sweep: (0, 0, 0, 50, U.Terran_Comsat_Station),
}


def make_game(self_race: Race = Race.Terran, enemy_race: Race = Race.Zerg, map_name: str = "Fake Map",
              n_starts: int = 2) -> GameInfo:
    ut = np.zeros(N_TYPES, dtype=UNIT_TYPE_DTYPE)
    ut["ground_weapon"] = NONE_W
    ut["air_weapon"] = NONE_W
    ut["what_builds"] = -1
    names = [f"type{i}" for i in range(N_TYPES)]
    req: list[list[int]] = [[] for _ in range(N_TYPES)]
    for tid, (race, m, g, bt, sr, sp, hp, sh, arm, gw, aw, rng, size, builder, bc, tw, th, fl, rq) in UNITS.items():
        r = ut[int(tid)]
        r["id"], r["race"], r["mineral_price"], r["gas_price"], r["build_time"] = int(tid), int(race), m, g, bt
        r["supply_required"], r["supply_provided"], r["max_hit_points"], r["max_shields"] = sr, sp, hp, sh
        r["armor"], r["ground_weapon"], r["air_weapon"] = arm, int(gw), int(aw)
        r["max_ground_hits"] = 1 if int(gw) != NONE_W else 0
        r["max_air_hits"] = 1 if int(aw) != NONE_W else 0
        r["sight_range"] = 7 * 32
        r["seek_range"] = rng * 32
        r["top_speed"] = 0.0 if fl & B else 4.0
        r["tile_width"], r["tile_height"], r["size"] = tw, th, size
        r["dimension_left"] = r["dimension_right"] = tw * 16 - 1
        r["dimension_up"] = r["dimension_down"] = th * 16 - 1
        r["what_builds"], r["what_builds_count"] = int(builder), bc
        r["required_tech"] = int(TechType.None_)
        r["flags"] = int(fl)
        ut[int(tid)] = r
        names[int(tid)] = U(int(tid)).name
        req[int(tid)] = [int(x) for x in rq] + ([int(builder)] if int(builder) >= 0 and int(builder) != NONE_W else [])

    weapons = []
    for wid, (dmg, bonus, cd, factor, dtype, rmin, rmax, splash, air, ground, upg) in WEAPONS.items():
        weapons.append(dict(id=int(wid), name=W(int(wid)).name, tech=int(TechType.None_), what_uses=-1,
                            damage_amount=dmg, damage_bonus=bonus, damage_cooldown=cd, damage_factor=factor,
                            upgrade_type=int(upg), damage_type=dtype, explosion_type=1, min_range=rmin,
                            max_range=rmax, inner_splash_radius=splash, median_splash_radius=splash,
                            outer_splash_radius=splash, targets_air=air, targets_ground=ground))
    upgrades = [dict(id=int(k), name=UpgradeType(int(k)).name, race=1, mineral_price=v[0], gas_price=v[1],
                     mineral_price_factor=v[2], gas_price_factor=v[3], upgrade_time=v[4], upgrade_time_factor=v[5],
                     max_repeats=v[6], what_upgrades=int(v[7])) for k, v in UPGRADES.items()]
    techs = [dict(id=int(k), name=TechType(int(k)).name, race=1, mineral_price=v[0], gas_price=v[1],
                  research_time=v[2], energy_cost=v[3], what_researches=int(v[4]), weapon=NONE_W,
                  targets_unit=False, targets_position=k == TechType.Scanner_Sweep) for k, v in TECHS.items()]

    w = h = 128
    starts_all = [(8, 8), (116, 116), (116, 8), (8, 116)][:n_starts]
    starts = np.zeros(len(starts_all), dtype=[("x", "<i4"), ("y", "<i4")])
    for i, s in enumerate(starts_all):
        starts[i] = s
    players = [
        PlayerInfo(0, "me", int(self_race), 0, 0, True, False, False, False, False, starts_all[0], 0),
        PlayerInfo(1, "enemy", int(enemy_race), 0, 1, False, True, False, False, False, starts_all[1], 1),
        PlayerInfo(11, "neutral", int(Race.None_), 0, 0, False, False, False, True, False, (-1, -1), 0),
    ]
    areas, bases, chokes, start_bases = [], [], [], []
    naturals = {(8, 8): (24, 10), (116, 116): (100, 114), (116, 8): (100, 10), (8, 116): (24, 114)}
    for i, s in enumerate(starts_all):
        n = naturals[s]
        a_main, a_nat = 2 * i + 1, 2 * i + 2
        areas += [MapArea(a_main, (s[0] * 4, s[1] * 4), 0, 0, 0, 0), MapArea(a_nat, (n[0] * 4, n[1] * 4), 0, 0, 0, 0)]
        bid = len(bases)
        bases.append(MapBase(bid, a_main, s, (s[0] * 32 + 64, s[1] * 32 + 48), 8, 1, True))
        bases.append(MapBase(bid + 1, a_nat, n, (n[0] * 32 + 64, n[1] * 32 + 48), 7, 1, False))
        cx, cy = (s[0] + n[0]) * 16 + 64, (s[1] + n[1]) * 16 + 48
        chokes.append(MapChoke(i, a_main, a_nat, (cx, cy), 96, False))
        start_bases.append(StartBase(s, bid, bid + 1))
    bases.append(MapBase(len(bases), 99, (60, 60), (60 * 32 + 64, 60 * 32 + 48), 8, 1, False))

    info = GameInfo(
        map_name=map_name, map_file_name=f"{map_name}.scx", map_path_name=f"maps/{map_name}.scx",
        map_hash="deadbeef" * 5, map_width=w, map_height=h,
        ground_height=np.zeros((h, w), np.uint8), buildable=np.ones((h, w), bool),
        walkable=np.ones((h * 4, w * 4), bool), region_id=np.zeros((h, w), np.uint16), start_locations=starts,
        players=players, self_id=0, enemy_id=1, neutral_id=11, game_type=2, latency_frames=2, random_seed=1,
        is_replay=False, frame_skip=0, unit_types=ut, unit_type_names=names, unit_type_required_units=req,
        weapon_types=weapons, upgrade_types=upgrades, tech_types=techs, areas=areas, bases=bases, chokes=chokes,
        start_bases=start_bases, self_main_id=0, self_natural_id=1,
    )
    info._by_id = {p.id: p for p in players}
    info._bases_by_id = {b.id: b for b in bases}
    return info


# ---------------------------------------------------------------------------- units / observations
@dataclass
class FakeWorld:
    """Mutable game state from which observations are generated."""

    game: GameInfo
    frame: int = 0
    minerals: int = 50
    gas: int = 0
    units: list[dict] = field(default_factory=list)
    upgrades: dict[int, int] = field(default_factory=dict)
    techs: set[int] = field(default_factory=set)
    events: list[Event] = field(default_factory=list)
    next_id: int = 1
    tiles: Optional[np.ndarray] = None       # per-tile TileFlag bytes; None = everything visible

    def add(self, unit_type: int, x: int, y: int, player: int = 0, completed: bool = True, idle: bool = True,
            **kw) -> int:
        uid = self.next_id
        self.next_id += 1
        t = self.game.unit_types[int(unit_type)]
        flags = UnitFlag.Exists | (UnitFlag.Completed if completed else 0) | (UnitFlag.Idle if idle else 0)
        d = dict(id=uid, type=int(unit_type), player=player, x=x, y=y, hit_points=int(t["max_hit_points"]),
                 shields=int(t["max_shields"]), flags=int(flags), visible_mask=0b11, order=-1, target=-1,
                 order_target=-1, build_type=-1, build_unit=-1, addon=-1, transport=-1, carrier=-1,
                 hatchery=-1, nydus_exit=-1, power_up=-1, rally_unit=-1, tech=int(TechType.None_),
                 upgrade=int(UpgradeType.None_), last_attacker_player=-1, last_hit_points=int(t["max_hit_points"]),
                 resources=0, resource_group=0)
        d.update(kw)
        self.units.append(d)
        return uid

    def remove(self, uid: int) -> None:
        self.units = [u for u in self.units if u["id"] != uid]

    def get(self, uid: int) -> Optional[dict]:
        return next((u for u in self.units if u["id"] == uid), None)

    def standard_start(self, workers: int = 4) -> None:
        """CC + workers + a mineral line at our main, enemy main building at theirs."""
        g = self.game
        sx, sy = g.self_player.start_location
        race = Race(g.self_player.race)
        depot = {Race.Terran: U.Terran_Command_Center, Race.Zerg: U.Zerg_Hatchery}.get(race, U.Protoss_Nexus)
        worker = {Race.Terran: U.Terran_SCV, Race.Zerg: U.Zerg_Drone}.get(race, U.Protoss_Probe)
        self.add(depot, sx * 32 + 64, sy * 32 + 48)
        for i in range(8):
            self.add(U.Resource_Mineral_Field, sx * 32 - 96, sy * 32 + i * 32, player=g.neutral_id, resources=1500,
                     resource_group=1)
        self.add(U.Resource_Vespene_Geyser, sx * 32 + 256, sy * 32, player=g.neutral_id, resources=5000)
        for i in range(workers):
            self.add(worker, sx * 32 + i * 10, sy * 32 + 100)
        if race == Race.Zerg:
            for _ in range(3):
                self.add(U.Zerg_Larva, sx * 32 + 64, sy * 32 + 90)
            self.add(U.Zerg_Overlord, sx * 32 + 64, sy * 32)

    def observe(self, frame: Optional[int] = None, visible_enemy: bool = True) -> Observation:
        if frame is not None:
            self.frame = frame
        g = self.game
        arr = np.zeros(len(self.units), dtype=UNIT_DTYPE)
        for i, d in enumerate(self.units):
            for k, v in d.items():
                arr[i][k] = v
        if not visible_enemy:
            arr = arr[arr["player"] != g.enemy_id]
        all_c = np.zeros(N_TYPES, np.int32)
        done_c = np.zeros(N_TYPES, np.int32)
        supply_used = supply_total = 0
        for d in self.units:
            if d["player"] != g.self_id:
                continue
            t = g.unit_types[d["type"]]
            all_c[d["type"]] += 1
            supply_used += int(t["supply_required"])
            if d["flags"] & UnitFlag.Completed:
                done_c[d["type"]] += 1
                supply_total += int(t["supply_provided"])
        up = np.zeros(int(UpgradeType.MAX), np.int32)
        for k, v in self.upgrades.items():
            up[int(k)] = v
        res = np.zeros(int(TechType.MAX), np.uint8)
        for k in self.techs:
            res[int(k)] = 1
        me = PlayerState(g.self_id, self.minerals, self.gas, 0, 0, supply_used, min(400, supply_total), False, False,
                         False, all_c, done_c, np.zeros(N_TYPES, np.int32), np.zeros(N_TYPES, np.int32), up, res,
                         np.zeros(int(TechType.MAX), np.uint8), np.zeros(int(UpgradeType.MAX), np.uint8),
                         0, 0, 0, 0, 0)
        empty = np.zeros(0, np.int32)
        enemy = PlayerState(g.enemy_id, 0, 0, 0, 0, 0, 0, False, False, False, empty, empty, empty, empty, empty,
                            np.zeros(0, np.uint8), np.zeros(0, np.uint8), np.zeros(0, np.uint8), 0, 0, 0, 0, 0)
        obs = Observation.__new__(Observation)
        obs.game = g
        obs.frame = None
        obs.frame_count = self.frame
        obs.elapsed_time = self.frame // 24
        obs.fps = obs.average_fps = 24
        obs.latency_frames = obs.remaining_latency_frames = 0
        obs.is_paused = False
        obs.self_id = g.self_id
        obs.units = arr
        obs.bullets = np.zeros(0)
        obs.nuke_dots = np.zeros(0)
        obs.serialize_us = obs.last_roundtrip_us = obs.game_apm = 0
        obs._players = {g.self_id: me, g.enemy_id: enemy}
        obs._events = list(self.events)
        self.events.clear()
        obs._tiles = self.tiles.copy() if self.tiles is not None else np.full((g.map_height, g.map_width), 3, np.uint8)
        obs._by_id = None
        obs._placement_results = []
        return obs


def ids(units: Iterable) -> list[int]:
    return [int(u["id"]) for u in units]


# ---------------------------------------------------------------------------- tiny economy simulator
MINERAL_RATE = 0.045      # per worker per frame (~65/min)
GAS_RATE = 0.07


class Sim:
    """Applies a bot's commands to a FakeWorld with costs, build times and income. No combat,
    no movement: enough to walk build orders and check that planners/executors make progress."""

    def __init__(self, world: FakeWorld) -> None:
        from bwbot.enums import Order, UnitCommandType as C
        self.w = world
        self.C = C
        self.Order = Order
        self.pending: list[tuple[int, str, int, int]] = []    # (finish frame, kind, uid or type, extra)
        self.jobs: dict[int, str] = {}                       # worker id -> "min" | "gas"
        self.fm = 0.0
        self.fg = 0.0
        self.log: list[str] = []

    def _cost(self, t: int) -> tuple[int, int, int]:
        ut = self.w.game.unit_types[t]
        return int(ut["mineral_price"]), int(ut["gas_price"]), int(ut["supply_required"])

    def _afford(self, m: int, g: int, s: int = 0) -> bool:
        obs = self.w.observe()
        free = obs.me.supply_total - obs.me.supply_used
        return self.w.minerals >= m and self.w.gas >= g and free >= s

    def apply(self, act) -> None:
        C, w = self.C, self.w
        for c in act.unit_cmds:
            u = w.get(c.unit)
            if u is None:
                continue
            if c.type == C.Train:
                m, g, s = self._cost(c.extra)
                if u.get("train_queue_count", 0) == 0 and u["flags"] & UnitFlag.Completed and self._afford(m, g, s):
                    w.minerals -= m
                    w.gas -= g
                    u["train_queue_count"] = 1
                    u["flags"] &= ~int(UnitFlag.Idle)
                    uid = w.add(c.extra, u["x"], u["y"] + 40, completed=False)
                    bt = int(w.game.unit_types["build_time"][c.extra])
                    self.pending.append((w.frame + bt, "train", uid, u["id"]))
            elif c.type == C.Build:
                m, g, _ = self._cost(c.extra)
                if u.get("build_type", -1) == c.extra or not self._afford(m, g):
                    continue
                w.minerals -= m
                w.gas -= g
                ut = w.game.unit_types[c.extra]
                px = c.x * 32 + int(ut["tile_width"]) * 16
                py = c.y * 32 + int(ut["tile_height"]) * 16
                bid = w.add(c.extra, px, py, completed=False, idle=False)
                u["build_type"] = c.extra
                u["flags"] = (u["flags"] | int(UnitFlag.Constructing)) & ~int(UnitFlag.Idle)
                u["order"] = int(self.Order.ConstructingBuilding)
                self.jobs.pop(u["id"], None)
                self.pending.append((w.frame + int(ut["build_time"]), "build", bid, u["id"]))
                self.log.append(f"f{w.frame} build {w.game.type_name(c.extra)}")
            elif c.type == C.Build_Addon:
                m, g, _ = self._cost(c.extra)
                if u.get("addon", -1) >= 0 or not self._afford(m, g):
                    continue
                w.minerals -= m
                w.gas -= g
                aid = w.add(c.extra, u["x"] + 64, u["y"], completed=False, idle=False)
                u["addon"] = aid
                self.pending.append((w.frame + int(w.game.unit_types["build_time"][c.extra]), "build", aid, -1))
            elif c.type in (C.Upgrade, C.Research):
                infos = w.game.upgrade_types if c.type == C.Upgrade else w.game.tech_types
                info = next((i for i in infos if i["id"] == c.extra), None)
                if info is None or u.get("remaining_upgrade_time", 0) or u.get("remaining_research_time", 0):
                    continue
                if not self._afford(info["mineral_price"], info["gas_price"]):
                    continue
                w.minerals -= info["mineral_price"]
                w.gas -= info["gas_price"]
                key = "remaining_upgrade_time" if c.type == C.Upgrade else "remaining_research_time"
                dur = info.get("upgrade_time", info.get("research_time", 0))
                u[key] = dur
                kind = "upgrade" if c.type == C.Upgrade else "research"
                self.pending.append((w.frame + dur, kind, c.extra, u["id"]))
            elif c.type == C.Gather:
                tgt = w.get(c.target)
                if tgt is None:
                    continue
                gas = tgt["type"] in (int(U.Terran_Refinery), int(U.Zerg_Extractor), int(U.Protoss_Assimilator))
                self.jobs[u["id"]] = "gas" if gas else "min"
                u["order"] = int(self.Order.HarvestGas if gas else self.Order.MiningMinerals)
                u["order_target"] = c.target
                u["flags"] &= ~int(UnitFlag.Idle)

    def step(self, frames: int) -> None:
        w = self.w
        for _ in range(frames):
            w.frame += 1
            n_min = sum(1 for j in self.jobs.values() if j == "min")
            n_gas = sum(1 for j in self.jobs.values() if j == "gas")
            self.fm += n_min * MINERAL_RATE
            self.fg += n_gas * GAS_RATE
            if self.fm >= 1:
                w.minerals += int(self.fm)
                self.fm -= int(self.fm)
            if self.fg >= 1:
                w.gas += int(self.fg)
                self.fg -= int(self.fg)
        done = [p for p in self.pending if p[0] <= w.frame]
        self.pending = [p for p in self.pending if p[0] > w.frame]
        for _, kind, a, b in done:
            if kind in ("train", "build"):
                u = w.get(a)
                if u is not None:
                    u["flags"] |= int(UnitFlag.Completed) | int(UnitFlag.Idle)
                src = w.get(b) if b >= 0 else None
                if src is not None:
                    src["train_queue_count"] = 0
                    src["build_type"] = -1
                    src["flags"] = (src["flags"] | int(UnitFlag.Idle)) & ~int(UnitFlag.Constructing)
                    src["order"] = -1
            elif kind == "upgrade":
                w.upgrades[a] = w.upgrades.get(a, 0) + 1
                src = w.get(b)
                if src is not None:
                    src["remaining_upgrade_time"] = 0
            elif kind == "research":
                w.techs.add(a)
                src = w.get(b)
                if src is not None:
                    src["remaining_research_time"] = 0
        for wid in list(self.jobs):
            if w.get(wid) is None:
                del self.jobs[wid]

    def run(self, bot, frames: int, skip: int = 8, until=None) -> None:
        from bwbot import Actions
        for _ in range(0, frames, skip):
            act = Actions()
            act.frame_count = self.w.frame
            obs = self.w.observe()
            bot.on_frame(obs, act)
            self.apply(act)
            self.step(skip)
            if until is not None and until(self.w):
                return
