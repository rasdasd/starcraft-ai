import os

from adjutant.components.micro import Micro
from adjutant.components.tactics import Tactics
from blackboard import Blackboard, Component, Phase
from blackboard.profile import register
from blackboard.recorder import Recorder
from blackboard.sections import Posture
from bwbot import Race, TechType, UnitType as U
from bwbot.enums import UnitCommandType as C

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


@register("FixedPosture")
class FixedPosture(Component):
    phase = Phase.DECIDE
    reads = ("world",)
    writes = ("strategy",)

    def __init__(self, stance: str = "hold", attack_supply: int = 10, harass: bool = False) -> None:
        self.posture = Posture(stance, attack_supply, 2, harass)

    def tick(self, bb: Blackboard) -> None:
        bb.strategy.posture = Posture(**self.posture.__dict__)


class Recording(Sim):
    def __init__(self, w):
        super().__init__(w)
        self.cmds = []

    def apply(self, act):
        self.cmds.append(list(act.unit_cmds))
        super().apply(act)

    def last(self):
        return self.cmds[-1] if self.cmds else []

    def all(self):
        return [c for step in self.cmds for c in step]


def _setup(stance="hold", enemy_race=Race.Zerg, **posture):
    from adjutant import Adjutant
    g = make_game(enemy_race=enemy_race)
    bot = Adjutant("planned", slots={"strategy": {"impl": "FixedPosture", "stance": stance, **posture}})
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    return bot, w, g


def _squads(bot):
    return bot.bb.squads.squads


def _home(g):
    sx, sy = g.self_player.start_location
    return sx * 32 + 64, sy * 32 + 48


def test_hold_stance_holds_at_main_choke_and_micro_leases_units():
    bot, w, g = _setup("hold")
    hx, hy = _home(g)
    ids = [w.add(U.Terran_Vulture, hx + i * 20, hy + 180) for i in range(3)]
    sim = Recording(w)
    sim.run(bot, 48)
    main = _squads(bot)["main"]
    assert main.units == set(ids) and main.order.kind == "hold"
    # a few tiles inside the main from the choke, not on the ramp
    cx, cy = g.main_choke.center
    d_choke = ((main.order.x - cx) ** 2 + (main.order.y - cy) ** 2) ** 0.5
    d_home = ((main.order.x - hx) ** 2 + (main.order.y - hy) ** 2) ** 0.5
    assert 3 * 32 <= d_choke <= 5 * 32 and d_home < ((hx - cx) ** 2 + (hy - cy) ** 2) ** 0.5
    assert all(bot.bb.leases.owner(i) == "micro" for i in ids)
    moves = [c for c in sim.all() if c.unit in ids and c.type == C.Attack_Move]
    assert moves and all((c.x, c.y) == (main.order.x, main.order.y) for c in moves)


def test_attack_stance_pushes_at_attack_supply_and_retreats_from_bad_fights():
    bot, w, g = _setup("attack", attack_supply=6)
    hx, hy = _home(g)
    ids = [w.add(U.Terran_Vulture, hx + 1500 + i * 20, hy + 1500) for i in range(4)]   # 8 supply
    sim = Recording(w)
    sim.run(bot, 48)
    main = _squads(bot)["main"]
    assert main.order.kind == "attack"
    ex, ey = g.players[1].start_location
    assert (main.order.x, main.order.y) == (ex * 32 + 64, ey * 32 + 48)
    # a big hydra ball shows up right next to the army (far from home): hopeless -> retreat
    for i in range(14):
        w.add(U.Zerg_Hydralisk, hx + 1620 + i * 12, hy + 1530, player=g.enemy_id)
    n0 = len(sim.cmds)
    sim.run(bot, 24)
    assert bot.bb.engagements.by_squad["c0"].win_prob < 0.35
    main = _squads(bot)["main"]
    assert main.order.kind == "retreat"
    later = [c for step in sim.cmds[n0:] for c in step if c.unit in ids]
    assert later and all(c.type == C.Move for c in later)


def test_home_threat_gets_a_defense_squad_sized_by_the_estimate():
    bot, w, g = _setup("hold")
    hx, hy = _home(g)
    near = [w.add(U.Terran_Goliath, hx + 150 + i * 20, hy + 150) for i in range(4)]
    far = [w.add(U.Terran_Goliath, hx + 1500 + i * 20, hy + 1500) for i in range(6)]
    w.add(U.Zerg_Zergling, hx + 200, hy + 200, player=g.enemy_id)
    Recording(w).run(bot, 32)
    sq = _squads(bot)
    assert "defense" in sq and sq["defense"].order.kind == "defend"
    assert sq["defense"].units <= set(near) and len(sq["defense"].units) >= 1
    assert set(far) <= sq["main"].units


def test_damaged_mech_walks_home_for_repair():
    bot, w, g = _setup("hold")
    hx, hy = _home(g)
    gid = w.add(U.Terran_Goliath, hx + 900, hy + 900, hit_points=20)
    ok = w.add(U.Terran_Goliath, hx + 920, hy + 900)
    Recording(w).run(bot, 32)
    sq = _squads(bot)
    assert sq["repair"].units == {gid} and sq["repair"].order.kind == "retreat"
    assert ok in sq["main"].units
    w.get(gid)["hit_points"] = 120
    Recording(w).run(bot, 16)
    assert "repair" not in _squads(bot) and gid in _squads(bot)["main"].units


def test_focus_fire_spreads_without_overkill():
    bot, w, g = _setup("hold")
    hx, hy = _home(g)
    x0, y0 = hx + 800, hy + 800
    marines = [w.add(U.Terran_Marine, x0 + i * 10, y0) for i in range(10)]
    lings = [w.add(U.Zerg_Zergling, x0 + 60 + i * 20, y0 + 60, player=g.enemy_id) for i in range(2)]
    sim = Recording(w)
    sim.run(bot, 8)
    attacks = [c for c in sim.all() if c.unit in marines and c.type == C.Attack_Unit]
    targets = {c.target for c in attacks}
    assert targets == set(lings)
    per = {t: sum(1 for c in attacks if c.target == t) for t in targets}
    assert max(per.values()) <= 7                       # 35 hp ling: six 6-damage volleys cover it


def test_vulture_kites_melee_while_on_cooldown():
    bot, w, g = _setup("hold", enemy_race=Race.Protoss)
    hx, hy = _home(g)
    x0, y0 = hx + 800, hy + 800
    v = w.add(U.Terran_Vulture, x0, y0, ground_weapon_cooldown=10)
    w.add(U.Protoss_Zealot, x0 + 40, y0, player=g.enemy_id)
    sim = Recording(w)
    sim.run(bot, 8)
    mine = [c for c in sim.all() if c.unit == v]
    assert mine and mine[0].type == C.Move and mine[0].x < x0     # away from the zealot (east of it)


def test_tanks_siege_near_enemies_and_unsiege_when_quiet():
    bot, w, g = _setup("attack", attack_supply=100)
    w.techs.add(TechType.Tank_Siege_Mode)
    hx, hy = _home(g)
    x0, y0 = hx + 800, hy + 800
    t = w.add(U.Terran_Siege_Tank_Tank_Mode, x0, y0)
    ling = w.add(U.Zerg_Zergling, x0 + 300, y0, player=g.enemy_id)
    sim = Recording(w)
    sim.run(bot, 8)
    assert any(c.unit == t and c.type == C.Siege for c in sim.all())
    w.get(t)["type"] = int(U.Terran_Siege_Tank_Siege_Mode)
    w.remove(ling)
    sim.run(bot, 24 * 6)
    assert any(c.unit == t and c.type == C.Unsiege for c in sim.all())


def test_micro_throttles_repeated_commands():
    bot, w, g = _setup("hold")
    hx, hy = _home(g)
    ids = [w.add(U.Terran_Vulture, hx + 900 + i * 20, hy + 900) for i in range(3)]
    sim = Recording(w)
    sim.run(bot, 8 * 6)                                  # 6 decisions inside the repeat window... mostly
    n = sum(1 for c in sim.all() if c.unit in ids)
    assert n <= 3 * 2
    micro = next(c for c in bot.sched.components if isinstance(c, Micro))
    assert micro.sent >= 3
    assert all(s.failures == 0 for s in bot.sched.stats.values())
