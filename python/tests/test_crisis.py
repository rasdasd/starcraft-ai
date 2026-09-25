import os

from adjutant.components.crisis import WorkerDefense
from blackboard.recorder import Recorder
from bwbot import Race, UnitType as U
from bwbot.enums import UnitCommandType as C
from bwbot.observation import UnitFlag

from fakes import FakeWorld, Sim, make_game
from test_tactics import Recording

os.environ["BWBOT_LOG"] = "0"


def _setup(enemy_race=Race.Protoss, workers=8):
    from adjutant import Adjutant
    g = make_game(enemy_race=enemy_race)
    bot = Adjutant("planned", slots={})
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    w = FakeWorld(g, minerals=50)
    w.standard_start(workers)
    sx, sy = g.self_player.start_location
    return bot, w, g, (sx * 32 + 64, sy * 32 + 48)


def _kinds(bot):
    return {t.kind for t in bot.bb.threats.active}


def _queued(bot, t) -> bool:
    """Planned at some point: owned, in production, or waiting in the production queue."""
    from adjutant.components.planner import GreedyPlanner
    p = next(c for c in bot.sched.components if isinstance(c, GreedyPlanner))
    return p.have(bot.bb, int(t)) > 0


def test_worker_rush_pulls_workers_and_defends():
    bot, w, g, (hx, hy) = _setup()
    probes = [w.add(U.Protoss_Probe, hx + 60 + i * 12, hy + 90, player=g.enemy_id) for i in range(4)]
    sim = Recording(w)
    sim.run(bot, 24 * 20, until=lambda _: False)
    assert "worker_rush" in _kinds(bot)
    assert bot.bb.threats.posture_override == "defend"
    wd = next(c for c in bot.sched.components if isinstance(c, WorkerDefense))
    assert len(wd.pulled) == 6                             # 4 probes + 2
    assert all(bot.bb.leases.owner(i) == wd.slot for i in wd.pulled)
    attacks = [c for c in sim.all() if c.unit in wd.pulled and c.type == C.Attack_Unit]
    assert attacks and {c.target for c in attacks} <= set(probes)
    # the probes leave: pull ends, workers go back to mining
    for p in probes:
        w.remove(p)
    sim.run(bot, 24 * 3)
    assert not wd.pulled and not bot.bb.leases.owned(wd.slot)


def test_early_rush_requests_bunker_and_defends():
    bot, w, g, (hx, hy) = _setup()
    w.add(U.Terran_Barracks, hx + 200, hy - 100)
    for i in range(6):
        w.add(U.Protoss_Zealot, hx + 250 + i * 20, hy + 250, player=g.enemy_id)
    sim = Recording(w)
    sim.run(bot, 24 * 2)
    assert "early_rush" in _kinds(bot)
    assert _queued(bot, U.Terran_Bunker)
    assert bot.bb.strategy.posture.stance == "defend"


def test_cloaked_units_request_scan_and_detection():
    bot, w, g, (hx, hy) = _setup(enemy_race=Race.Zerg)
    w.add(U.Terran_Barracks, hx + 200, hy - 100)
    w.add(U.Zerg_Zergling, hx + 300, hy + 300, player=g.enemy_id,
          flags=int(UnitFlag.Exists | UnitFlag.Completed | UnitFlag.Cloaked))
    sim = Recording(w)
    sim.run(bot, 24 * 2)
    assert "cloak" in _kinds(bot)
    scans = bot.bb.requests.active("scan")
    assert scans and (scans[0].x, scans[0].y) == (hx + 300, hy + 300)
    # no academy / engineering bay yet: comsat and turret requests chain their prerequisites
    assert _queued(bot, U.Terran_Academy) and _queued(bot, U.Terran_Engineering_Bay)


def test_cannon_rush_is_a_crisis():
    bot, w, g, (hx, hy) = _setup()
    w.add(U.Protoss_Pylon, hx + 200, hy + 200, player=g.enemy_id)
    w.add(U.Protoss_Photon_Cannon, hx + 250, hy + 200, player=g.enemy_id, completed=False)
    w.add(U.Protoss_Probe, hx + 230, hy + 230, player=g.enemy_id)
    Recording(w).run(bot, 24 * 2)
    assert "cannon_rush" in _kinds(bot)
    wd = next(c for c in bot.sched.components if isinstance(c, WorkerDefense))
    assert len(wd.pulled) >= 3
    assert all(s.failures == 0 for s in bot.sched.stats.values())
