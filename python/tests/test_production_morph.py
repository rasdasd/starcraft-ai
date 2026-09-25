from bwbot import Actions, Race, UnitType as U

from fakes import FakeWorld, Sim, make_game
from mybot.buildings import BuildingManager
from mybot.production import ProductionManager
from mybot.state import Memory, perceive


def test_building_made_from_a_building_is_morphed_not_placed():
    g = make_game(self_race=Race.Zerg, enemy_race=Race.Terran)
    w = FakeWorld(g, minerals=400)
    w.standard_start(4)
    sx, sy = g.self_player.start_location
    w.add(U.Zerg_Spawning_Pool, sx * 32 + 200, sy * 32 + 100)
    colony = w.add(U.Zerg_Creep_Colony, sx * 32 + 260, sy * 32 + 100)
    pm, bm = ProductionManager(), BuildingManager()
    pm.ensure_build(U.Zerg_Sunken_Colony, bm)
    act = Actions()
    pm.update(perceive(w.observe(), g, Memory()), act, bm)
    Sim(w).apply(act)
    assert not pm.queue and not bm.tasks
    assert w.get(colony)["type"] == int(U.Zerg_Sunken_Colony)
