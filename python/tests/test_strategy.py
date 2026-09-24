import os

import pytest

from adjutant.strategies import TEMPLATES, blend_goals, get, validate_goal, validate_template
from adjutant.techtree import TechTree
from blackboard.recorder import Recorder
from blackboard.sections import STANCES, Goal
from bwbot import Race, UnitType as U, UpgradeType as Up

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _bot(template):
    from adjutant import Adjutant
    bot = Adjutant("parity", slots={"strategy": {"impl": "ScriptedStrategy", "template": template}})
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    return bot


def test_templates_static_contracts():
    tree = TechTree(make_game())
    assert set(TEMPLATES) >= {"goliath_1fact", "bio_2rax", "anti_rush", "mech_expand"}
    for name, t in TEMPLATES.items():
        assert validate_template(t, tree) == [], name


@pytest.mark.parametrize("name", ["goliath_1fact", "bio_2rax", "anti_rush", "mech_expand"])
def test_template_goals_and_postures_valid_during_play(name):
    g = make_game()
    tree = TechTree(g)
    bot = _bot(name)
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    bot.game = g
    bot.on_start(g)
    sim = Sim(w)
    for _ in range(12):
        sim.run(bot, 24 * 30, skip=8)
        st = bot.bb.strategy
        assert st.template == name
        assert validate_goal(st.goal, tree, int(Race.Terran)) == [], (name, w.frame)
        assert st.posture.stance in STANCES
        assert st.posture.retreat_supply <= st.posture.attack_supply
    assert all(s.failures == 0 for s in bot.sched.stats.values())
    bot.on_end(False)


def test_validate_goal_catches_bad_items():
    tree = TechTree(make_game())
    bad = Goal(units={int(U.Terran_Barracks): 1, int(U.Zerg_Zergling): 4}, buildings={int(U.Terran_Machine_Shop): 1},
               addons={int(U.Terran_Factory): 1}, upgrades=[(int(Up.U_238_Shells), 2)], techs=[9999], bases=0)
    errs = validate_goal(bad, tree, int(Race.Terran))
    assert len(errs) == 7, errs


def test_blend_goals_weights_counts_and_orders_upgrades():
    a = Goal(units={1: 10}, upgrades=[(5, 1)], workers=20, bases=1)
    b = Goal(units={1: 0, 2: 8}, upgrades=[(6, 1)], workers=40, bases=3)
    m = blend_goals([a, b], [3, 1])
    assert m.units == {1: 8, 2: 2}
    assert m.upgrades == [(5, 1), (6, 1)]
    assert m.workers == 25 and m.bases == 2
    tiny = blend_goals([a, b], [0.9, 0.1])
    assert tiny.upgrades == [(5, 1)]


def test_techtree_missing_chain():
    g = make_game()
    tree = TechTree(g)
    have = {int(U.Terran_Command_Center): 1, int(U.Terran_SCV): 4}
    chain = tree.missing([int(U.Terran_Goliath)], lambda t: have.get(t, 0))
    assert chain == [int(U.Terran_Barracks), int(U.Terran_Factory), int(U.Terran_Armory), int(U.Terran_Goliath)]
    assert tree.upgrade_requires(int(Up.Charon_Boosters)) == [int(U.Terran_Machine_Shop), int(U.Terran_Armory)]
    assert get("goliath_1fact").race == int(Race.Terran)
