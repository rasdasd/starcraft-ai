import os

from adjutant.components.planner import GreedyPlanner
from blackboard.recorder import Recorder
from bwbot import TechType as T, UnitType as U

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _bot(template):
    from adjutant import Adjutant
    bot = Adjutant("planned", slots={"strategy": {"impl": "ScriptedStrategy", "template": template}})
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


def test_bio_builds_rax_academy_and_stim():
    bot, w, worst = _play("bio_2rax", 12)
    assert worst <= 2
    assert _n(w, U.Terran_Barracks) >= 3
    assert _n(w, U.Terran_Academy) == 1
    assert _n(w, U.Terran_Marine) >= 10
    assert int(T.Stim_Packs) in w.techs or any(u.get("remaining_research_time", 0) for u in w.units)
    items = bot.bb.plan.items
    assert [it.priority for it in items] == sorted((it.priority for it in items), reverse=True)


def test_expansion_goes_to_natural_exact_tile():
    g = make_game()
    bot = _bot("mech_expand")
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    bot.game = g
    bot.on_start(g)
    Sim(w).run(bot, 24 * 10, skip=8)
    planner = next(c for c in bot.sched.components if isinstance(c, GreedyPlanner))
    assert planner.next_base(bot.bb) == g.bases[g.self_natural_id].tile


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
