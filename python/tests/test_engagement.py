import os

import numpy as np

from adjutant import engage as E
from adjutant.components.engagement import LanchesterEngagement
from blackboard.recorder import Recorder
from bwbot import EventType, Race, UnitType as U
from bwbot.observation import Event, UnitFlag

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _side(table, counts, cloaked=()):
    side = E.side_from_counts(counts, table)
    for m in side.members:
        m.cloaked = m.info.type_id in cloaked
    return side


def test_lanchester_basics():
    g = make_game(enemy_race=Race.Zerg)
    t = E.TypeTable(g)
    big = E.evaluate(_side(t, {U.Terran_Marine: 10}), _side(t, {U.Zerg_Zergling: 3}))
    small = E.evaluate(_side(t, {U.Terran_Marine: 2}), _side(t, {U.Zerg_Zergling: 10}))
    assert big.win_prob > 0.9 and big.own_left > 0.5
    assert small.win_prob < 0.1 and small.own_left == 0
    even = E.evaluate(_side(t, {U.Terran_Marine: 5}), _side(t, {U.Terran_Marine: 5}))
    assert abs(even.win_prob - 0.5) < 1e-6
    # vultures cannot shoot up: mutalisks win for free
    air = E.evaluate(_side(t, {U.Terran_Vulture: 8}), _side(t, {U.Zerg_Mutalisk: 2}))
    assert air.win_prob == 0.0 and air.own_rate == 0.0
    # goliaths can
    gol = E.evaluate(_side(t, {U.Terran_Goliath: 6}), _side(t, {U.Zerg_Mutalisk: 3}))
    assert gol.win_prob > 0.8


def test_damage_types_armor_and_upgrades():
    g = make_game(enemy_race=Race.Protoss)
    t = E.TypeTable(g)
    tank = t.get(U.Terran_Siege_Tank_Tank_Mode).ground
    # explosive: full damage to large (dragoon), half to small (zergling-size marine)
    assert tank.per_frame(0, 3) == 2 * tank.per_frame(0, 1)
    up = np.zeros(64, np.int32)
    up[int(E.G.Terran_Infantry_Weapons)] = 3
    t_up = E.TypeTable(g, up)
    assert t_up.get(U.Terran_Marine).ground.per_hit == t.get(U.Terran_Marine).ground.per_hit + 3
    # armor never drops damage below 0.5 per hit
    assert t.get(U.Terran_Marine).ground.per_frame(100, 1) > 0


def test_cloaked_units_need_detection():
    g = make_game(enemy_race=Race.Zerg)
    t = E.TypeTable(g)
    own = _side(t, {U.Terran_Marine: 10})
    lings = _side(t, {U.Zerg_Zergling: 4}, cloaked={int(U.Zerg_Zergling)})
    blind = E.evaluate(own, lings)
    assert blind.own_rate == 0 and blind.win_prob == 0
    own.detects = True
    assert E.evaluate(own, lings).win_prob > 0.9


def test_cluster_groups_nearby_points():
    pts = np.array([[0, 0], [100, 0], [200, 0], [5000, 5000], [5100, 5000]], np.float32)
    groups = E.cluster(pts, 150)
    assert [sorted(g.tolist()) for g in groups] == [[0, 1, 2], [3, 4]]


def _bot():
    from adjutant import Adjutant
    g = make_game(enemy_race=Race.Zerg)
    bot = Adjutant("scripted", slots={})
    rec = Recorder(enabled=False)
    rows = []
    rec.record = lambda slot, kind, frame, every=0, **data: rows.append((slot, kind, data)) or True
    bot.recorder = rec
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    eng = next(c for c in bot.sched.components if isinstance(c, LanchesterEngagement))
    return bot, w, g, eng, rows


def test_engagement_component_writes_clusters_and_logs_fights():
    bot, w, g, eng, rows = _bot()
    sx, sy = g.self_player.start_location
    x0, y0 = sx * 32 + 400, sy * 32 + 400
    marines = [w.add(U.Terran_Marine, x0 + i * 16, y0) for i in range(8)]
    lings = [w.add(U.Zerg_Zergling, x0 + 120 + i * 10, y0 + 40, player=g.enemy_id) for i in range(4)]
    sim = Sim(w)
    sim.run(bot, 48)
    by = bot.bb.engagements.by_squad
    assert "c0" in by and by["c0"].contact and by["c0"].win_prob > 0.8
    assert set(by["c0"].own_ids) == set(marines) and set(by["c0"].enemy_ids) == set(lings)
    assert bot.bb.services["engage"] is eng
    # two lings and one marine die, the rest of the lings run off and are forgotten
    for uid in lings[:2]:
        w.remove(uid)
        w.events.append(Event(EventType.UnitDestroy, uid, g.enemy_id, False, "", (x0, y0)))
    w.remove(marines[0])
    w.events.append(Event(EventType.UnitDestroy, marines[0], g.self_id, False, "", (x0, y0)))
    for uid in lings[2:]:
        w.get(uid)["x"] = 100 * 32
    sim.run(bot, 24 * 8)
    fights = [d for s, k, d in rows if (s, k) == ("engagement", "fight")]
    assert len(fights) == 1
    f = fights[0]
    assert f["enemy_lost"] == {str(int(U.Zerg_Zergling)): 2} and f["own_lost"] == {str(int(U.Terran_Marine)): 1}
    assert f["won"] is True                     # 100 (two fake-data lings) vs 50 lost
    assert f["snap"]["own"] == {str(int(U.Terran_Marine)): 8}
    assert bot.bb.engagements.fights == 1


def test_global_ratio_uses_belief_counts():
    bot, w, g, eng, rows = _bot()
    sx, sy = g.self_player.start_location
    for i in range(6):
        w.add(U.Terran_Goliath, sx * 32 + 300 + i * 20, sy * 32 + 300)
    ex, ey = g.players[1].start_location
    for i in range(12):
        w.add(U.Zerg_Mutalisk, ex * 32 + i * 20, ey * 32, player=g.enemy_id)
    Sim(w).run(bot, 48)
    e = bot.bb.engagements
    assert e.global_win_prob < 0.5 and e.global_ratio < 1
    assert all(s.failures == 0 for s in bot.sched.stats.values())
