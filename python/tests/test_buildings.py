import numpy as np

from bwbot import Actions, UnitFlag, UnitType as U
from bwbot.enums import UnitCommandType as C

from fakes import FakeWorld, Sim, make_game
from mybot.buildings import CONSTRUCTING, BuildingManager
from mybot.macro import Placer
from mybot.production import ProductionManager
from mybot.state import Memory, perceive
from mybot.workers import WorkerManager


def _tick(w, sim, pm, bm, wm, placer):
    act = Actions()
    s = perceive(w.observe(), w.game, Memory())
    pm.update(s, act, bm)
    bm.update(s, act, wm, placer)
    sim.apply(act)
    sim.step(8)
    return act


def test_protoss_buildings_go_in_pylon_power_and_zerg_on_creep():
    from bwbot import Race
    from bwbot.observation import TileFlag, UnitTypeFlag
    from mybot.macro import find_build_tile

    g = make_game(self_race=Race.Protoss)
    g.unit_types["flags"][int(U.Protoss_Gateway)] |= int(UnitTypeFlag.RequiresPsi)
    w = FakeWorld(g, minerals=500)
    w.standard_start(4)
    sx, sy = g.self_player.start_location
    near = (sx + 2, sy + 6)
    assert find_build_tile(w.observe(), U.Protoss_Gateway, near) is None          # no pylon yet
    w.add(U.Protoss_Pylon, (sx + 8) * 32 + 32, (sy + 8) * 32 + 32)
    tx, ty = find_build_tile(w.observe(), U.Protoss_Gateway, near)
    assert ((tx + 2 - (sx + 9)) / 7.5) ** 2 + ((ty + 1.5 - (sy + 9)) / 4.5) ** 2 <= 1

    gz = make_game(self_race=Race.Zerg)
    gz.unit_types["flags"][int(U.Zerg_Spawning_Pool)] |= int(UnitTypeFlag.RequiresCreep)
    wz = FakeWorld(gz, minerals=500)
    wz.standard_start(4)
    tiles = np.full((gz.map_height, gz.map_width), 3, np.uint8)
    tiles[sy + 8:sy + 14, sx + 4:sx + 12] |= np.uint8(TileFlag.Creep)
    wz.tiles = tiles
    tx, ty = find_build_tile(wz.observe(), U.Zerg_Spawning_Pool, (sx + 2, sy + 10))
    assert sy + 8 <= ty and ty + 2 <= sy + 14 and sx + 4 <= tx and tx + 3 <= sx + 12


def test_terran_builder_pulled_off_construction_is_sent_back():
    g = make_game()
    w = FakeWorld(g, minerals=500)
    w.standard_start(4)
    sim, pm, bm, wm, placer = Sim(w), ProductionManager(), BuildingManager(), WorkerManager(), Placer()
    pm.ensure_build(U.Terran_Supply_Depot, bm)
    for _ in range(40):
        _tick(w, sim, pm, bm, wm, placer)
        if bm.tasks and bm.tasks[0].status == CONSTRUCTING:
            break
    task = bm.tasks[0]
    assert task.status == CONSTRUCTING
    scv = w.get(task.worker_id)
    scv["flags"] &= ~int(UnitFlag.Constructing)       # pulled away mid-construction
    scv["order"] = -1
    back = []
    for _ in range(3):
        act = _tick(w, sim, pm, bm, wm, placer)
        back += [c for c in act.unit_cmds if c.type == C.Right_Click_Unit and c.target == task.building_id]
    assert back and back[0].unit == task.worker_id
