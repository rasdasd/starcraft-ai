import os

import numpy as np

from adjutant.components.scouting import Scouting
from blackboard import Priority
from blackboard.recorder import Recorder
from bwbot import Race, UnitType as U

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _setup(scouting=None):
    from adjutant import Adjutant
    g = make_game(enemy_race=Race.Zerg)
    slots = {"scouting": {"impl": "Scouting", **(scouting or {})}}
    bot = Adjutant("planned", slots=slots)
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    tiles = np.zeros((g.map_height, g.map_width), np.uint8)
    sx, sy = g.self_player.start_location
    tiles[max(0, sy - 12):sy + 16, max(0, sx - 12):sx + 16] = 3        # we only see around our main
    w.tiles = tiles
    ex, ey = g.players[1].start_location
    w.add(U.Zerg_Hatchery, ex * 32 + 64, ey * 32 + 48, player=g.enemy_id)
    sc = next(c for c in bot.sched.components if isinstance(c, Scouting))
    return bot, w, g, sc


def test_rank_targets_prefers_stale_enemy_bases():
    bot, w, g, sc = _setup({"worker_scout": False})
    Sim(w).run(bot, 24 * 130)
    ranked = sc.rank_targets(bot.bb)
    assert ranked, "everything but our main is stale"
    ids = [bid for _, _, bid in ranked]
    assert g.self_main_id not in ids
    assert ids[0] == 2                                   # enemy main (known, alive, stale)
    assert bot.bb.scouting.targets[0] == g.bases[2].center


def test_rescout_leases_unit_and_times_out():
    bot, w, g, sc = _setup({"worker_scout": False, "timeout_s": 30})
    sx, sy = g.self_player.start_location
    vid = w.add(U.Terran_Vulture, sx * 32 + 200, sy * 32 + 200)
    sim = Sim(w)
    sim.run(bot, 24 * 125)
    assert sc.unit_id == vid
    lease = bot.bb.leases.get(vid)
    assert lease.owner == sc.slot and lease.priority == int(Priority.COMBAT) + 1
    assert vid in bot.bb.scouting.scouts
    sim.run(bot, 24 * 40)                                # never arrives in the Sim -> timeout
    assert sc.unit_id is None and bot.bb.leases.owner(vid) in (None, "micro")   # back to the army


def test_periodic_scan_uses_comsat_energy():
    bot, w, g, sc = _setup({"worker_scout": False, "scan_every_s": 60})
    sx, sy = g.self_player.start_location
    w.add(U.Terran_Comsat_Station, sx * 32 + 130, sy * 32 + 48, energy=150)
    Sim(w).run(bot, 24 * 130)
    assert bot.bb.scouting.last_scan_frame > 0
    assert all(s.failures == 0 for s in bot.sched.stats.values())
