from bwbot import Actions, Race, UnitType as U

from fakes import FakeWorld, Sim, make_game
from mybot.buildings import BuildingManager
from mybot.production import ProductionManager
from mybot.state import Memory, perceive


def test_eggs_count_as_what_they_morph_into():
    from bwbot.observation import UnitTypeFlag
    g = make_game(self_race=Race.Zerg, enemy_race=Race.Terran)
    g.unit_types["flags"][int(U.Zerg_Zergling)] |= int(UnitTypeFlag.TwoUnitsInOneEgg)
    w = FakeWorld(g, minerals=400)
    w.standard_start(4)
    sx, sy = g.self_player.start_location
    w.add(U.Zerg_Egg, sx * 32 + 64, sy * 32 + 100, completed=False, build_type=int(U.Zerg_Overlord))
    w.add(U.Zerg_Egg, sx * 32 + 96, sy * 32 + 100, completed=False, build_type=int(U.Zerg_Zergling))
    s = perceive(w.observe(), g, Memory())
    assert s.count(U.Zerg_Overlord) == 2 and s.count(U.Zerg_Zergling) == 2


def test_zerg_supply_is_trained_from_larva():
    from adjutant import Adjutant
    from blackboard.recorder import Recorder
    g = make_game(self_race=Race.Zerg, enemy_race=Race.Terran)
    bot = Adjutant("planned")
    bot.recorder = Recorder(enabled=False)
    w = FakeWorld(g, minerals=50)
    w.standard_start(9)
    bot.game = g
    bot.on_start(g)
    Sim(w).run(bot, 24 * 60 * 3, skip=8)
    kinds = {(it.kind, it.type_id) for it in bot.bb.plan.items}
    assert ("build", int(U.Zerg_Overlord)) not in kinds
    assert not any(t.unit_type == int(U.Zerg_Overlord) for t in bot.bb.services["buildings"].tasks)
    assert w.observe().count(U.Zerg_Overlord) >= 2


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
