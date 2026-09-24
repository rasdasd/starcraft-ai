"""Unit-type groups shared by adjutant components (raw BWAPI ids)."""
from __future__ import annotations

from bwbot import UnitType as U

WORKERS = frozenset({int(U.Terran_SCV), int(U.Zerg_Drone), int(U.Protoss_Probe)})
DEPOTS = frozenset({int(U.Terran_Command_Center), int(U.Zerg_Hatchery), int(U.Zerg_Lair), int(U.Zerg_Hive),
                    int(U.Protoss_Nexus)})
SUPPLY = frozenset({int(U.Terran_Supply_Depot), int(U.Zerg_Overlord), int(U.Protoss_Pylon)})

AIR_COMBAT = frozenset({
    int(U.Terran_Wraith), int(U.Terran_Valkyrie), int(U.Terran_Battlecruiser),
    int(U.Zerg_Mutalisk), int(U.Zerg_Scourge), int(U.Zerg_Guardian), int(U.Zerg_Devourer), int(U.Zerg_Queen),
    int(U.Protoss_Scout), int(U.Protoss_Carrier), int(U.Protoss_Corsair), int(U.Protoss_Arbiter),
})
AIR_TRANSPORT = frozenset({int(U.Terran_Dropship), int(U.Protoss_Shuttle), int(U.Zerg_Overlord)})
AIR_TECH = frozenset({int(U.Terran_Starport), int(U.Terran_Control_Tower), int(U.Zerg_Spire),
                      int(U.Zerg_Greater_Spire), int(U.Protoss_Stargate), int(U.Protoss_Fleet_Beacon)})

CLOAKERS = frozenset({int(U.Terran_Wraith), int(U.Terran_Ghost), int(U.Zerg_Lurker), int(U.Protoss_Dark_Templar),
                      int(U.Protoss_Arbiter), int(U.Protoss_Observer)})
CLOAK_TECH = frozenset({int(U.Protoss_Templar_Archives), int(U.Protoss_Citadel_of_Adun),
                        int(U.Zerg_Lurker_Egg), int(U.Terran_Covert_Ops), int(U.Protoss_Arbiter_Tribunal)})
DETECTORS = frozenset({int(U.Terran_Missile_Turret), int(U.Terran_Comsat_Station), int(U.Terran_Science_Vessel)})

STATIC_DEFENSE = frozenset({int(U.Terran_Bunker), int(U.Terran_Missile_Turret), int(U.Zerg_Sunken_Colony),
                            int(U.Zerg_Spore_Colony), int(U.Protoss_Photon_Cannon)})
PRODUCTION = frozenset({int(U.Terran_Barracks), int(U.Terran_Factory), int(U.Terran_Starport),
                        int(U.Zerg_Hatchery), int(U.Zerg_Lair), int(U.Zerg_Hive), int(U.Protoss_Gateway),
                        int(U.Protoss_Robotics_Facility), int(U.Protoss_Stargate)})
REFINERIES = frozenset({int(U.Terran_Refinery), int(U.Zerg_Extractor), int(U.Protoss_Assimilator)})
