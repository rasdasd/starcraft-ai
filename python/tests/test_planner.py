import os

from adjutant.components.planner import GreedyPlanner
from blackboard.recorder import Recorder
from bwbot import Race, TechType as T, UnitType as U

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _bot(template):
    from adjutant import Adjutant
    bot = Adjutant("scripted", slots={"strategy": {"impl": "ScriptedStrategy", "template": template}})
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    return bot


def _play(template, minutes, workers=6):
    g = make_game()
    bot = _bot(template)
    w = FakeWorld(g, minerals=50)
    w.standard_start(workers)
    bot.game = g
    bot.on_start(g)
    sim = Sim(w)
    blocked = 0
    worst = 0
    for _ in range(minutes * 2):
        sim.run(bot, 24 * 30, skip=8)
        s = bot.bb.world
        blocked = blocked + 1 if s.supply_left <= 0 and s.supply_total < 200 else 0
        worst = max(worst, blocked)
    assert all(st.failures == 0 for st in bot.sched.stats.values()), bot.sched.stats
    return bot, w, worst


def _n(w, t, completed=False):
    from bwbot import UnitFlag
    return sum(1 for u in w.units if u["type"] == int(t) and u["player"] == 0
               and (not completed or u["flags"] & UnitFlag.Completed))


def test_mech_expand_gets_factories_shop_tanks_and_expands():
    bot, w, worst = _play("mech_expand", 14)
    assert worst <= 2, "supply blocked for more than a minute"
    assert _n(w, U.Terran_Factory) >= 2
    assert _n(w, U.Terran_Machine_Shop) >= 1
    assert _n(w, U.Terran_Command_Center) >= 2
    assert _n(w, U.Terran_Siege_Tank_Tank_Mode) + _n(w, U.Terran_Vulture) > 0
    assert _n(w, U.Terran_Refinery) <= 2        # one geyser per owned base on the fake map


def test_machine_shop_before_minute_eight_while_army_is_short():
    bot, w, _ = _play("mech_expand", 8)
    assert _n(w, U.Terran_Machine_Shop) >= 1
    assert _n(w, U.Terran_Siege_Tank_Tank_Mode) >= 1


def test_idle_barracks_spends_surplus_on_marines():
    g = make_game()
    bot = _bot("mech_expand")
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    sx, sy = g.self_player.start_location
    w.add(U.Terran_Barracks, sx * 32 + 300, sy * 32 + 200)
    for _ in range(3):
        w.add(U.Terran_Supply_Depot, sx * 32 + 400, sy * 32 + 300)
    bot.game = g
    bot.on_start(g)
    w.minerals = 800
    Sim(w).run(bot, 8, skip=8)
    fill = [it for it in bot.bb.plan.items if it.reason == "filler"]
    assert fill and all(it.type_id == int(U.Terran_Marine) for it in fill)


def test_bio_builds_rax_academy_and_stim():
    bot, w, worst = _play("bio_2rax", 12)
    assert worst <= 2
    assert _n(w, U.Terran_Barracks) >= 3
    assert _n(w, U.Terran_Academy) == 1
    assert _n(w, U.Terran_Marine) >= 10
    assert int(T.Stim_Packs) in w.techs or any(u.get("remaining_research_time", 0) for u in w.units)
    items = bot.bb.plan.items
    assert [it.priority for it in items] == sorted((it.priority for it in items), reverse=True)


def test_dispatched_jobs_are_counted_so_nothing_is_built_twice():
    g = make_game()
    bot = _bot("mech_expand")
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    bot.game = g
    bot.on_start(g)
    sim = Sim(w)
    planner = next(c for c in bot.sched.components if isinstance(c, GreedyPlanner))
    for _ in range(12):
        sim.run(bot, 24 * 10, skip=8)
        for t in bot.bb.macro.pending:
            assert planner.have(bot.bb, t) >= bot.bb.world.count(t) + 1
    assert _n(w, U.Terran_Barracks) == 1 and _n(w, U.Terran_Refinery) == 1


def test_expansion_is_an_expand_item_toward_the_natural():
    g = make_game()
    bot = _bot("mech_expand")
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    bot.game = g
    bot.on_start(g)
    Sim(w).run(bot, 24 * 10, skip=8)
    assert bot.bb.macro.next_base == g.self_natural_id
    assert not any(it.kind == "build" and it.type_id == int(U.Terran_Command_Center) for it in bot.bb.plan.items)


def test_saturated_bases_take_another_past_the_build_goal():
    from types import SimpleNamespace as NS
    from blackboard.sections import MacroState, OwnBase
    m = MacroState(bases=[OwnBase(0, (0, 0), (0, 0), True, patches=9, refineries=1),
                          OwnBase(1, (0, 0), (0, 0), True, patches=8)])
    m.worker_target = 2 * 17 + 3
    w = NS(workers=[0] * 36, minerals=300, under_attack=False)
    bb, goal, hall = NS(macro=m, world=w), NS(bases=2, workers=60), int(U.Terran_Command_Center)
    p = GreedyPlanner()
    assert p.want_bases(bb, goal, hall) == 3
    w.minerals = 100
    assert p.want_bases(bb, goal, hall) == 2          # not floating
    w.minerals, w.workers = 300, [0] * 20
    assert p.want_bases(bb, goal, hall) == 2          # not saturated
    w.workers = [0] * 44
    w.minerals = 100
    assert p.want_bases(bb, goal, hall) == 3          # heavily oversaturated: expand without a float
    w.workers = [0] * 36
    m.expanding = 5
    assert p.want_bases(bb, goal, hall) == 2          # one in flight already counts


def test_floating_gas_buys_upgrades_for_the_army_and_their_building():
    import numpy as np
    from types import SimpleNamespace as NS
    from adjutant.techtree import TechTree
    from blackboard.sections import MacroState
    from bwbot import UpgradeType as G
    forge, gw = int(U.Protoss_Forge), int(G.Protoss_Ground_Weapons)
    units = {int(U.Protoss_Nexus): 1}
    w = NS(count=lambda t: units.get(t, 0), count_completed=lambda t: units.get(t, 0))
    bb = NS(world=w, macro=MacroState())
    me = NS(upgrade_level=np.zeros(64, int), is_upgrading=np.zeros(64, bool))
    goal = NS(units={int(U.Protoss_Dragoon): 20})
    p = GreedyPlanner()
    p.race = int(Race.Protoss)
    tree = TechTree(make_game())

    def run():
        out, emitted = [], set()
        p._gas_sink(bb, tree, goal, lambda kind, t, *a, **k: out.append((kind, t)), me, emitted)
        return out

    assert ("build", forge) in run()
    units[forge] = 1
    assert ("upgrade", gw) in run()
    me.is_upgrading[gw] = True
    assert ("upgrade", gw) not in run()
    goal.units = {int(U.Protoss_Corsair): 5}
    assert run() == []


def test_workers_are_made_one_base_ahead():
    from types import SimpleNamespace as NS
    from adjutant.strategies.base import workers_for
    from blackboard.sections import MacroState, OwnBase
    ref = int(U.Terran_Refinery)
    w = NS(count=lambda t: 1 if t == ref else 0, count_completed=lambda t: 0)
    main = OwnBase(0, (0, 0), (0, 0), True, patches=8)
    bb = NS(macro=MacroState(bases=[main]), game=NS(self_race=int(Race.Terran)), world=w)
    assert workers_for(bb) == 16 * 2 + 3                      # the main and the next base
    bb.macro.bases = [main, OwnBase(1, (0, 0), (0, 0), False, patches=8)]
    assert workers_for(bb) == 16 * 2 + 3                      # the next base is under construction
    bb.macro.bases[1].completed = True
    assert workers_for(bb) == 16 * 3 + 3
    assert workers_for(bb, cap=40) == 40


def test_zerg_floating_without_larva_adds_a_hatchery():
    from types import SimpleNamespace as NS
    from blackboard.sections import MacroState
    hatch, larva = int(U.Zerg_Hatchery), int(U.Zerg_Larva)
    units = {hatch: 2, larva: 0}
    w = NS(minerals=900, count=lambda t: units.get(t, 0), count_completed=lambda t: units.get(t, 0))
    bb = NS(world=w, macro=MacroState())
    p = GreedyPlanner()
    p.race = int(Race.Zerg)
    assert p._larva_starved(bb, hatch)
    units[larva] = 2
    assert not p._larva_starved(bb, hatch)
    units[larva], bb.macro.pending = 0, {hatch: 1}
    assert not p._larva_starved(bb, hatch)            # one already on the way
    p.race = int(Race.Terran)
    bb.macro.pending = {}
    assert not p._larva_starved(bb, hatch)


def test_army_trains_leave_money_for_tech():
    g = make_game()
    bot = _bot("mech_expand")
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    sx, sy = g.self_player.start_location
    for t in (U.Terran_Barracks, U.Terran_Factory, U.Terran_Refinery):
        w.add(t, sx * 32 + 300, sy * 32 + 200)
    for _ in range(3):
        w.add(U.Terran_Supply_Depot, sx * 32 + 400, sy * 32 + 300)
    bot.game = g
    bot.on_start(g)
    Sim(w).run(bot, 24 * 60, skip=8)
    w.minerals, w.gas = 250, 100
    sim = Sim(w)
    sim.run(bot, 8, skip=8)
    kinds = {(it.kind, it.type_id) for it in bot.bb.plan.items}
    assert ("addon", int(U.Terran_Machine_Shop)) in kinds or _n(w, U.Terran_Machine_Shop) > 0


def test_zerglings_are_a_larva_filler():
    from adjutant.techtree import TechTree
    from bwbot import Race
    tree = TechTree(make_game(self_race=Race.Zerg, enemy_race=Race.Terran))
    assert tree.supply(int(U.Zerg_Zergling)) == 1
    assert tree.fillers(int(U.Zerg_Larva), int(Race.Zerg))[0] == int(U.Zerg_Zergling)


def test_larva_starved_zerg_with_a_bank_adds_a_macro_hatchery():
    from bwbot import Race
    g = make_game(self_race=Race.Zerg, enemy_race=Race.Terran)
    bot = _bot("hydra_3hatch")
    w = FakeWorld(g, minerals=900)
    w.standard_start(12)
    w.units = [u for u in w.units if u["type"] != int(U.Zerg_Larva)]
    sx, sy = g.self_player.start_location
    w.add(U.Zerg_Spawning_Pool, sx * 32 + 200, sy * 32 + 100)
    w.add(U.Zerg_Hydralisk_Den, sx * 32 + 260, sy * 32 + 100)
    w.frame = 24 * 60 * 7
    bot.game = g
    bot.on_start(g)
    Sim(w).run(bot, 16, skip=8)
    planned = ("build", int(U.Zerg_Hatchery)) in {(it.kind, it.type_id) for it in bot.bb.plan.items}
    assert planned or bot.bb.macro.pending_count(U.Zerg_Hatchery) == 1
