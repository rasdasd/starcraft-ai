import os

import pytest

from blackboard.recorder import Recorder
from bwbot import Race, UnitType as U

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _bot(profile="adjutant", **over):
    from adjutant import Adjutant
    bot = Adjutant(profile, **over)
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    return bot


def _run(race, enemy, minutes, workers=4, profile="adjutant"):
    bot = _bot(profile)
    g = make_game(self_race=race, enemy_race=enemy)
    w = FakeWorld(g, minerals=50)
    w.standard_start(workers)
    bot.game = g
    bot.on_start(g)
    sim = Sim(w)
    sim.run(bot, 24 * 60 * minutes, skip=8)
    st = bot.sched.stats
    assert all(s.failures == 0 for s in st.values()), {k: s.errors for k, s in st.items()}
    return bot, w, sim


@pytest.mark.parametrize("race,worker,producer", [
    (Race.Terran, U.Terran_SCV, U.Terran_Barracks),
    (Race.Protoss, U.Protoss_Probe, U.Protoss_Gateway),
    (Race.Zerg, U.Zerg_Drone, U.Zerg_Spawning_Pool),
])
def test_default_bot_macros_every_race(race, worker, producer):
    bot, w, sim = _run(race, Race.Zerg if race != Race.Zerg else Race.Terran, 8)
    obs = w.observe()
    assert obs.count(worker) >= 14, sim.log
    assert obs.count(producer) >= 1, sim.log
    assert bot.bb.macro.bases, "macro tracks our bases"
    bot.on_end(False)


def test_expanding_build_takes_its_natural():
    bot, w, sim = _run(Race.Terran, Race.Zerg, 10, profile="scripted")      # mech_expand
    g = w.game
    nat = g.base(g.self_natural_id)
    assert any(b.base_id == nat.id for b in bot.bb.macro.bases) or bot.bb.macro.expanding == nat.id, \
        (bot.bb.macro.summary(), bot.bb.strategy.template, sim.log)
