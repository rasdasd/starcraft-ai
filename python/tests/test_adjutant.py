import os

import pytest

from blackboard.recorder import Recorder
from bwbot import Actions, Race, UnitType as U

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _bot(profile, **over):
    from adjutant import Adjutant
    bot = Adjutant(profile, **over)
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    return bot


def _start(bot, workers=4, minerals=50, enemy=Race.Zerg):
    g = make_game(enemy_race=enemy)
    w = FakeWorld(g, minerals=minerals)
    w.standard_start(workers)
    bot.game = g
    bot.on_start(g)
    return w


def test_parity_profile_runs_and_builds():
    bot = _bot("parity")
    w = _start(bot)
    sim = Sim(w)
    sim.run(bot, 24 * 60 * 8, skip=8)
    obs = w.observe()
    assert obs.count(U.Terran_SCV) >= 12
    assert obs.count(U.Terran_Supply_Depot) >= 1
    assert obs.count(U.Terran_Barracks) >= 1
    assert obs.count(U.Terran_Factory) >= 1, sim.log
    st = bot.sched.stats
    assert all(s.failures == 0 for s in st.values()), {k: s.errors for k, s in st.items()}
    assert bot.bb.strategy.template == "goliath_1fact"
    bot.on_end(False)


def test_default_profile_plays_planned_for_zerg():
    from adjutant.components.planner import GreedyPlanner
    bot = _bot("adjutant")
    g = make_game(self_race=Race.Zerg, enemy_race=Race.Terran)
    w = FakeWorld(g, minerals=50)
    w.standard_start(4)
    bot.game = g
    bot.on_start(g)
    assert any(isinstance(c, GreedyPlanner) for c in bot.sched.components)
    Sim(w).run(bot, 24 * 60 * 4, skip=8)
    assert all(s.failures == 0 for s in bot.sched.stats.values()), {k: s.errors for k, s in bot.sched.stats.items()}
    assert w.observe().count(U.Zerg_Spawning_Pool) >= 1


def test_parity_matches_goliath_commands():
    """Same observations -> same unit commands as the goliath bot (spread training off)."""
    from goliath import Goliath
    ref = Goliath()
    ref.logger.enabled = False
    par = _bot("parity", slots={"construction": {"impl": "Construction", "spread": False}})
    g = make_game()
    w = FakeWorld(g, minerals=50)
    w.standard_start(4)
    for b in (ref, par):
        b.game = g
        b.on_start(g)
    sim = Sim(w)
    for _ in range(0, 24 * 60 * 6, 8):
        obs_a = w.observe()
        obs_b = w.observe()
        a, b = Actions(), Actions()
        a.frame_count = b.frame_count = w.frame
        ref.on_frame(obs_a, a)
        par.on_frame(obs_b, b)
        ka = sorted((c.unit, c.type, c.target, c.x, c.y, c.extra) for c in a.unit_cmds)
        kb = sorted((c.unit, c.type, c.target, c.x, c.y, c.extra) for c in b.unit_cmds)
        assert ka == kb, (w.frame, ka, kb)
        sim.apply(a)
        sim.step(8)
    assert w.observe().count(U.Terran_Factory) >= 1
