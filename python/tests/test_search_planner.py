import os

from adjutant.buildsearch import BuildSearch, Target
from adjutant.components.search_planner import SearchPlanner
from adjutant.econsim import Action, EconSim, SimState
from adjutant.techtree import TechTree
from blackboard.recorder import Recorder
from bwbot import Race, UnitType as U

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"

SCV, CC, DEPOT, RAX, REF = (int(U.Terran_SCV), int(U.Terran_Command_Center), int(U.Terran_Supply_Depot),
                            int(U.Terran_Barracks), int(U.Terran_Refinery))
FACT, SHOP, TANK, VULT = (int(U.Terran_Factory), int(U.Terran_Machine_Shop), int(U.Terran_Siege_Tank_Tank_Mode),
                          int(U.Terran_Vulture))


def _start(minerals=50, workers=4, supply=(8, 20)):
    g = make_game()
    sim = EconSim(TechTree(g), int(Race.Terran), m_rate=0.05, g_rate=0.04)
    s = SimState(0, float(minerals), 0.0, workers, 0, supply[0], supply[1], {CC: 1, SCV: workers},
                 {CC: 1, SCV: workers}, {CC: [0]}, {CC: [0]}, {}, set())
    return g, sim, s


def test_income_and_when_wait_for_money():
    g, sim, s = _start(minerals=0, workers=4)
    m, gas = sim.income(s)
    assert abs(m - 0.2) < 1e-9 and gas == 0
    f = sim.when(s, Action("unit", DEPOT))            # 100 minerals at 0.2 / frame
    assert f == 500
    assert sim.when(s, Action("unit", FACT)) is None   # no barracks: not legal


def test_do_trains_on_producer_slots_and_completes():
    g, sim, s = _start(minerals=500)
    f1 = sim.do(s, Action("unit", SCV))
    f2 = sim.do(s, Action("unit", SCV))
    t = sim.tree.time(SCV)
    assert f1 == 0 and f2 == t                          # one command center: back to back
    sim.advance(s, 2 * t)
    assert s.done[SCV] == 6 and s.m_workers == 6


def test_building_takes_a_worker_and_frees_it():
    g, sim, s = _start(minerals=500)
    sim.do(s, Action("unit", DEPOT))
    assert s.m_workers == 3
    sim.advance(s, sim.travel + sim.tree.time(DEPOT) + 1)
    assert s.m_workers == 4 and s.supply_total == 20 + 16


def test_supply_blocks_until_a_depot_finishes():
    g, sim, s = _start(minerals=1000, supply=(20, 20))
    assert sim.when(s, Action("unit", SCV)) is None     # blocked, nothing coming
    sim.do(s, Action("unit", DEPOT))
    f = sim.when(s, Action("unit", SCV))
    assert f == sim.travel + sim.tree.time(DEPOT)


def test_search_reaches_a_tank_through_the_tech_chain():
    g, sim, s = _start(minerals=400, workers=10, supply=(20, 36))
    search = BuildSearch(sim, DEPOT, REF)
    target = Target({TANK: 1}, order=[Action("unit", TANK)])
    plan = search.search(s, target, budget_ms=200)
    kinds = [a.type_id for a, _ in plan.steps]
    for t in (RAX, REF, FACT, SHOP, TANK):
        assert t in kinds, (t, kinds)
    assert kinds.index(RAX) < kinds.index(FACT) < kinds.index(SHOP) < kinds.index(TANK)
    starts = [f for _, f in plan.steps]
    assert plan.makespan < 24 * 60 * 6 and starts == sorted(starts)


def test_search_beats_or_matches_both_greedy_policies():
    g, sim, s = _start(minerals=50, workers=8, supply=(16, 20))
    search = BuildSearch(sim, DEPOT, REF)
    target = Target({SCV: 14, RAX: 2, DEPOT: 2}, order=[Action("unit", SCV), Action("unit", RAX),
                                                         Action("unit", DEPOT)])
    best = search.search(s, target, budget_ms=300)
    for policy in ("order", "earliest"):
        assert best.score <= search.rollout(s, target, policy).score


def _play(template, minutes, workers=6):
    from adjutant import Adjutant
    g = make_game()
    bot = Adjutant("scripted", slots={"strategy": {"impl": "ScriptedStrategy", "template": template},
                                     "production": {"impl": "SearchPlanner", "budget_ms": 5.0}})
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    w = FakeWorld(g, minerals=50)
    w.standard_start(workers)
    bot.game = g
    bot.on_start(g)
    sim = Sim(w)
    blocked = worst = 0
    for _ in range(minutes * 2):
        sim.run(bot, 24 * 30, skip=8)
        s = bot.bb.world
        blocked = blocked + 1 if s.supply_left <= 0 and s.supply_total < 200 else 0
        worst = max(worst, blocked)
    assert all(st.failures == 0 for st in bot.sched.stats.values()), bot.sched.stats
    return bot, w, worst


def _n(w, t):
    return sum(1 for u in w.units if u["type"] == int(t) and u["player"] == 0)


def test_search_planner_plays_mech_expand():
    bot, w, worst = _play("mech_expand", 12)
    sp = next(c for c in bot.sched.components if isinstance(c, SearchPlanner))
    assert sp.stats["searches"] > 50 and sp.stats["fallbacks"] == 0
    assert worst <= 2, "supply blocked for more than a minute"
    assert _n(w, U.Terran_Factory) >= 2 and _n(w, U.Terran_Machine_Shop) >= 1
    assert _n(w, U.Terran_Command_Center) >= 2
    assert _n(w, U.Terran_Siege_Tank_Tank_Mode) + _n(w, U.Terran_Vulture) + _n(w, U.Terran_Goliath) >= 6
    assert _n(w, U.Terran_SCV) >= 20


def test_search_planner_plays_bio():
    bot, w, worst = _play("bio_2rax", 10)
    assert worst <= 2
    assert _n(w, U.Terran_Barracks) >= 2 and _n(w, U.Terran_Marine) >= 10
