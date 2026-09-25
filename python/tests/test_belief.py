import os

from blackboard.recorder import Recorder
from bwbot import EventType, Race, UnitType as U
from bwbot.observation import Event

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _setup(n_starts=2, enemy=Race.Zerg):
    from adjutant import Adjutant
    g = make_game(enemy_race=enemy, n_starts=n_starts)
    bot = Adjutant("scripted")
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    return bot, w, g


def _enemy_px(g, dx=0, dy=0):
    ex, ey = g.players[1].start_location
    return ex * 32 + 64 + dx, ey * 32 + 48 + dy


def test_tech_inference_and_kills():
    bot, w, g = _setup(enemy=Race.Terran)
    x, y = _enemy_px(g, -300, -300)
    vid = w.add(U.Terran_Vulture, x, y, player=g.enemy_id)
    w.add(U.Terran_Marine, x + 20, y, player=g.enemy_id)
    sim = Sim(w)
    sim.run(bot, 16)
    b = bot.bb.belief
    assert b.count(U.Terran_Vulture) == 1 and b.count(U.Terran_Marine) == 1
    assert {int(U.Terran_Factory), int(U.Terran_Barracks), int(U.Terran_Command_Center)} <= b.inferred
    assert int(U.Terran_Factory) in b.tech
    assert b.army_supply == 3 and b.army_pos is not None
    w.remove(vid)
    w.events.append(Event(EventType.UnitDestroy, vid, g.enemy_id, False, "", (x, y)))
    sim.run(bot, 16)
    assert b.count(U.Terran_Vulture) == 0 and b.dead == {int(U.Terran_Vulture): 1}
    assert int(U.Terran_Factory) in b.tech            # tech knowledge stays


def test_enemy_bases_starts_and_staleness():
    bot, w, g = _setup(n_starts=4)
    sim = Sim(w)
    sim.run(bot, 16)
    b = bot.bb.belief
    assert len(b.start_candidates) == 1            # every start is visible in the fake; no buildings anywhere
    hx, hy = _enemy_px(g)
    hid = w.add(U.Zerg_Hatchery, hx, hy, player=g.enemy_id)
    nx, ny = g.bases[3].center                     # enemy natural
    w.add(U.Zerg_Hatchery, nx, ny, player=g.enemy_id)
    sim.run(bot, 16)
    alive = [e for e in b.bases if e.alive]
    assert {e.base_id for e in alive} == {2, 3}
    assert b.enemy_start == g.players[1].start_location
    assert b.start_candidates == {tuple(g.players[1].start_location): 1.0}
    assert all(v <= 16 for v in b.staleness.values())
    w.remove(hid)
    sim.run(bot, 16)
    assert {e.base_id for e in b.bases if e.alive} == {3}


def test_opening_probs_rush():
    bot, w, g = _setup()
    x, y = _enemy_px(g)
    w.add(U.Zerg_Hatchery, x, y, player=g.enemy_id)
    w.add(U.Zerg_Spawning_Pool, x + 100, y, player=g.enemy_id)
    for i in range(6):
        w.add(U.Zerg_Zergling, x - 200 + i * 5, y, player=g.enemy_id)
    Sim(w).run(bot, 24 * 10)
    b = bot.bb.belief
    assert b.opening == "rush"
    assert max(b.opening_probs, key=b.opening_probs.get) == "rush"
    assert abs(sum(b.opening_probs.values()) - 1) < 0.01
    assert all(s.failures == 0 for s in bot.sched.stats.values())
