"""Macro executor: plan items -> commands, building jobs, expansions, worker pool."""
import os

import numpy as np

from blackboard import Component, Phase
from blackboard.profile import register
from blackboard.recorder import Recorder
from blackboard.sections import PlanItem
from bwbot import Actions, Race, UnitFlag, UnitType as U
from bwbot.enums import UnitCommandType as C

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


@register("TestPlan")
class FixedPlan(Component):
    """Publishes a fixed list of plan items."""
    phase = Phase.PLAN
    reads = ("world",)
    writes = ("plan",)

    def __init__(self, items=(), cancel=()) -> None:
        self.items = list(items)
        self.cancel = list(cancel)

    def tick(self, bb) -> None:
        bb.plan.items = [PlanItem(*it) for it in sorted(self.items, key=lambda it: -it[2])]
        bb.plan.cancel = list(self.cancel)


OFF = ("engagement", "strategy", "scouting", "repair", "tactics", "micro", "crisis", "worker_defense", "report")


def _bot(g, w, items=(), **macro):
    from adjutant import Adjutant
    slots = {k: None for k in OFF}
    slots["production"] = {"impl": "TestPlan", "items": list(items)}
    slots["macro"] = {"impl": "Macro", "draw": False, **macro}
    bot = Adjutant("adjutant", slots=slots)
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    return bot


def _comp(bot, slot):
    return next(c for c in bot.sched.components if c.slot == slot)


def _tick(bot, w, sim, skip=8):
    act = Actions()
    act.frame_count = w.frame
    bot.on_frame(w.observe(), act)
    sim.apply(act)
    sim.step(skip)
    return act


def _world(race=Race.Terran, workers=8, minerals=50, gas=0):
    g = make_game(self_race=race, enemy_race=Race.Terran)
    w = FakeWorld(g, minerals=minerals, gas=gas)
    w.standard_start(workers)
    return g, w


# ---------------------------------------------------------------------------- placement
def test_protoss_buildings_go_in_pylon_power_and_zerg_on_creep():
    from bwbot.observation import TileFlag, UnitTypeFlag
    from adjutant.macro.placement import find_build_tile, occupancy

    g, w = _world(Race.Protoss, 4, 500)
    g.unit_types["flags"][int(U.Protoss_Gateway)] |= int(UnitTypeFlag.RequiresPsi)
    sx, sy = g.self_player.start_location
    near = (sx + 2, sy + 6)
    obs = w.observe()
    assert find_build_tile(obs, U.Protoss_Gateway, near, occupancy(obs)) is None          # no pylon yet
    w.add(U.Protoss_Pylon, (sx + 8) * 32 + 32, (sy + 8) * 32 + 32)
    obs = w.observe()
    tx, ty = find_build_tile(obs, U.Protoss_Gateway, near, occupancy(obs))
    assert ((tx + 2 - (sx + 9)) / 7.5) ** 2 + ((ty + 1.5 - (sy + 9)) / 4.5) ** 2 <= 1

    gz, wz = _world(Race.Zerg, 4, 500)
    gz.unit_types["flags"][int(U.Zerg_Spawning_Pool)] |= int(UnitTypeFlag.RequiresCreep)
    tiles = np.full((gz.map_height, gz.map_width), 3, np.uint8)
    tiles[sy + 8:sy + 14, sx + 4:sx + 12] |= np.uint8(TileFlag.Creep)
    wz.tiles = tiles
    obs = wz.observe()
    tx, ty = find_build_tile(obs, U.Zerg_Spawning_Pool, (sx + 2, sy + 10), occupancy(obs))
    assert sy + 8 <= ty and ty + 2 <= sy + 14 and sx + 4 <= tx and tx + 3 <= sx + 12


# ---------------------------------------------------------------------------- building jobs
def test_terran_builder_pulled_off_construction_is_sent_back():
    g, w = _world(minerals=500)
    bot = _bot(g, w, [("build", int(U.Terran_Supply_Depot), 90)])
    sim, m = Sim(w), _comp(bot, "macro")
    for _ in range(40):
        _tick(bot, w, sim)
        if m.jobs and m.jobs[0].building_id is not None:
            break
    job = m.jobs[0]
    assert job.building_id is not None
    _comp(bot, "production").items = []
    scv = w.get(job.worker_id)
    scv["flags"] &= ~int(UnitFlag.Constructing)       # pulled away mid-construction
    scv["order"] = -1
    back = []
    for _ in range(3):
        act = _tick(bot, w, sim)
        back += [c for c in act.unit_cmds if c.type == C.Right_Click_Unit and c.target == job.building_id]
    assert back and back[0].unit == job.worker_id


def test_building_waits_for_money_then_finishes_and_releases_its_cost():
    g, w = _world(minerals=0)
    bot = _bot(g, w, [("build", int(U.Terran_Barracks), 90)])
    sim, m, plan = Sim(w), _comp(bot, "macro"), _comp(bot, "production")
    for _ in range(24 * 90 // 8):
        _tick(bot, w, sim)
        if m.jobs:
            plan.items = []                            # a planner counts the dispatched job
        if w.observe().count(U.Terran_Barracks, completed_only=True):
            break
    assert w.observe().count(U.Terran_Barracks, completed_only=True) == 1
    assert w.observe().count(U.Terran_Barracks) == 1
    _tick(bot, w, sim)
    assert not m.jobs and bot.bb.macro.reserved == (0, 0)


def test_gas_building_without_refinery_does_not_block_the_refinery():
    g, w = _world(minerals=400)
    sx, sy = g.self_player.start_location
    w.add(U.Terran_Barracks, sx * 32 + 200, sy * 32 + 200)
    bot = _bot(g, w, [("build", int(U.Terran_Factory), 90), ("build", int(U.Terran_Refinery), 80)])
    _tick(bot, w, Sim(w))
    m = _comp(bot, "macro")
    assert [j.unit_type for j in m.jobs] == [int(U.Terran_Refinery)]
    assert any("Factory:no gas" in b for b in bot.bb.macro.blocked)


def test_building_made_from_a_building_is_morphed_not_placed():
    g, w = _world(Race.Zerg, 4, 400)
    sx, sy = g.self_player.start_location
    w.add(U.Zerg_Spawning_Pool, sx * 32 + 200, sy * 32 + 200)
    colony = w.add(U.Zerg_Creep_Colony, sx * 32 + 260, sy * 32 + 200)
    bot = _bot(g, w, [("build", int(U.Zerg_Sunken_Colony), 90)])
    _tick(bot, w, Sim(w))
    assert not _comp(bot, "macro").jobs
    assert w.get(colony)["type"] == int(U.Zerg_Sunken_Colony)


def test_unaffordable_item_holds_money_but_a_blocked_one_does_not():
    g, w = _world(minerals=100, gas=100)
    sx, sy = g.self_player.start_location
    w.add(U.Terran_Barracks, sx * 32 + 200, sy * 32 + 200)
    marine = ("train", int(U.Terran_Marine), 50)
    bot = _bot(g, w, [("build", int(U.Terran_Factory), 90), marine], early_dispatch=False)
    act = _tick(bot, w, Sim(w))
    assert not any(c.type == C.Train for c in act.unit_cmds)            # saving for the factory
    g, w = _world(minerals=100, gas=100)
    w.add(U.Terran_Barracks, sx * 32 + 200, sy * 32 + 200)
    bot = _bot(g, w, [("train", int(U.Terran_Goliath), 90), marine])    # no factory / armory
    act = _tick(bot, w, Sim(w))
    assert any(c.type == C.Train and c.extra == int(U.Terran_Marine) for c in act.unit_cmds)


# ---------------------------------------------------------------------------- expansions
def _hall_at(w, tile):
    from adjutant.macro.placement import unit_tile
    return [u for u in w.units if u["type"] == int(U.Terran_Command_Center) and unit_tile(w.game, u) == tuple(tile)]


def test_expand_builds_the_hall_on_the_natural_site():
    g, w = _world(minerals=500)
    bot = _bot(g, w, [("expand", int(U.Terran_Command_Center), 90)])
    sim = Sim(w)
    nat = g.base(g.self_natural_id)
    assert bot.bb.macro.next_base in (None, nat.id)
    for _ in range(30):
        _tick(bot, w, sim)
        if _hall_at(w, nat.tile):
            break
    assert _hall_at(w, nat.tile)
    assert bot.bb.macro.base_count == 2 or bot.bb.macro.expanding == nat.id


def test_expansion_builder_leaves_before_the_money_is_there():
    g, w = _world(workers=12, minerals=0)
    bot = _bot(g, w, [])
    sim, m = Sim(w), _comp(bot, "macro")
    sim.run(bot, 24 * 40, skip=8)                     # mining: income known
    assert bot.bb.world.income_minerals > 600
    w.minerals = 360                                  # the walk to the natural earns the rest
    _comp(bot, "production").items = [("expand", int(U.Terran_Command_Center), 90)]
    _tick(bot, w, sim, skip=1)
    assert [j.base_id for j in m.jobs] == [g.self_natural_id]
    assert w.minerals < 400


def test_blocked_expansion_site_is_given_up_and_avoided():
    g, w = _world(minerals=1000)
    bot = _bot(g, w, [("expand", int(U.Terran_Command_Center), 90)], site_timeout_s=2)
    sim, m = Sim(w), _comp(bot, "macro")
    apply = sim.apply

    def no_hall(act):                                  # something sits on the site
        act.unit_cmds[:] = [c for c in act.unit_cmds
                            if not (c.type == C.Build and c.extra == int(U.Terran_Command_Center))]
        apply(act)
    sim.apply = no_hall
    for _ in range(80):
        _tick(bot, w, sim)
        if g.self_natural_id in m.bases.avoid:
            break
    assert g.self_natural_id in m.bases.avoid
    assert not m.jobs
    assert bot.bb.macro.next_base != g.self_natural_id


def test_cancelled_expansion_is_not_redispatched_and_bases_stay_usable():
    g, w = _world(minerals=1000)
    cc = int(U.Terran_Command_Center)
    bot = _bot(g, w, [("expand", cc, 90)])
    sim, m, plan = Sim(w), _comp(bot, "macro"), _comp(bot, "production")
    _tick(bot, w, sim, skip=1)
    assert [j.base_id for j in m.jobs] == [g.self_natural_id]
    plan.cancel = [cc]                                # a crisis at home: drop the unplaced hall
    for _ in range(10):
        _tick(bot, w, sim)
    assert not m.jobs and not m.bases.avoid
    plan.cancel = []
    _tick(bot, w, sim, skip=1)
    assert [j.base_id for j in m.jobs] == [g.self_natural_id]


def test_builder_taken_by_a_worker_pull_is_replaced():
    from blackboard import Priority
    g, w = _world(minerals=0)
    bot = _bot(g, w, [("build", int(U.Terran_Barracks), 90)], early_dispatch=False)
    sim, m = Sim(w), _comp(bot, "macro")
    w.minerals = 150
    _tick(bot, w, sim, skip=1)
    job = m.jobs[0]
    first = job.worker_id
    bot.bb.leases.lease(first, "crisis", int(Priority.CRISIS), "pull", w.frame)
    _tick(bot, w, sim, skip=1)
    assert job.worker_id not in (None, first) and job.deaths == 0


# ---------------------------------------------------------------------------- workers
def test_gas_workers_go_back_to_minerals_while_gas_is_over_banked():
    g, w = _world(workers=10)
    sx, sy = g.self_player.start_location
    w.add(U.Terran_Refinery, sx * 32 + 256, sy * 32)
    bot = _bot(g, w)
    sim, m = Sim(w), _comp(bot, "macro")

    def step():
        _tick(bot, w, sim, skip=1)
        return len(m.pool.gas)

    assert step() == 3
    w.gas, w.minerals = 800, 100
    assert step() == 0 and step() == 0
    w.gas = 400
    assert step() == 0                        # hysteresis: not until it is spent down
    w.gas = 100
    assert step() == 3


def test_workers_transfer_from_a_saturated_base_to_a_new_one():
    g, w = _world(workers=22)
    nat = g.base(g.self_natural_id)
    nx, ny = nat.tile
    w.add(U.Terran_Command_Center, nx * 32 + 64, ny * 32 + 48)
    for i in range(7):
        w.add(U.Resource_Mineral_Field, nx * 32 - 96, ny * 32 + i * 32, player=g.neutral_id, resources=1500,
              resource_group=2)
    bot = _bot(g, w)
    sim, m = Sim(w), _comp(bot, "macro")
    for _ in range(8):
        _tick(bot, w, sim)
    main = g.self_main_id
    assert m.pool.miners.get(main, 0) <= 16
    assert m.pool.miners.get(nat.id, 0) >= 6
    assert bot.bb.macro.worker_target == 2 * 8 + 2 * 7


def test_protoss_building_without_power_puts_down_a_pylon():
    from bwbot.observation import UnitTypeFlag
    g, w = _world(race=Race.Protoss, minerals=400)
    g.unit_types["flags"][int(U.Protoss_Gateway)] |= int(UnitTypeFlag.RequiresPsi)
    bot = _bot(g, w, [("build", int(U.Protoss_Gateway), 90)])
    sim, m = Sim(w), _comp(bot, "macro")
    _tick(bot, w, sim, skip=1)
    assert [j.unit_type for j in m.jobs] == [int(U.Protoss_Pylon)]
    assert any("pylon" in b for b in bot.bb.macro.blocked)


def test_full_main_builds_at_another_base():
    g, w = _world(minerals=500)
    nat = g.base(g.self_natural_id)
    nx, ny = nat.tile
    w.add(U.Terran_Command_Center, nx * 32 + 64, ny * 32 + 48)
    bot = _bot(g, w, [("build", int(U.Terran_Supply_Depot), 90)])
    sim, m = Sim(w), _comp(bot, "macro")
    mx, my = bot.bb.world.main_tile
    m.placer.failed.update((mx + dx, my + dy) for dx in range(-40, 41) for dy in range(-40, 41))
    _tick(bot, w, sim, skip=1)
    assert len(m.jobs) == 1
    tx, ty = m.jobs[0].tile
    assert abs(tx - nx) <= 32 and abs(ty - ny) <= 32


# ---------------------------------------------------------------------------- whole bot
def test_eggs_count_as_what_they_morph_into():
    from adjutant.components.perception import unit_counts
    g, w = _world(Race.Zerg, 4, 400)
    sx, sy = g.self_player.start_location
    w.add(U.Zerg_Egg, sx * 32 + 64, sy * 32 + 100, completed=False, build_type=int(U.Zerg_Overlord))
    w.add(U.Zerg_Egg, sx * 32 + 96, sy * 32 + 100, completed=False, build_type=int(U.Zerg_Zergling))
    counts, _ = unit_counts(w.observe(), g)
    assert counts[int(U.Zerg_Overlord)] == 2 and counts[int(U.Zerg_Zergling)] == 2


def test_zerg_supply_is_trained_from_larva():
    from adjutant import Adjutant
    g, w = _world(Race.Zerg, 9, 50)
    bot = Adjutant()
    bot.recorder = Recorder(enabled=False)
    bot.game = g
    bot.on_start(g)
    Sim(w).run(bot, 24 * 60 * 3, skip=8)
    kinds = {(it.kind, it.type_id) for it in bot.bb.plan.items}
    assert ("build", int(U.Zerg_Overlord)) not in kinds
    assert not any(j.unit_type == int(U.Zerg_Overlord) for j in _comp(bot, "macro").jobs)
    assert w.observe().count(U.Zerg_Overlord) >= 2
