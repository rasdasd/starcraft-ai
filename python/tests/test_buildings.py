from bwbot import Actions, UnitFlag, UnitType as U
from bwbot.enums import UnitCommandType as C

from fakes import FakeWorld, Sim, make_game
from mybot.buildings import CONSTRUCTING, BuildingManager
from mybot.macro import Placer
from mybot.production import ProductionManager
from mybot.state import Memory, perceive
from mybot.workers import WorkerManager


def _tick(w, sim, pm, bm, wm, placer):
    act = Actions()
    s = perceive(w.observe(), w.game, Memory())
    pm.update(s, act, bm)
    bm.update(s, act, wm, placer)
    sim.apply(act)
    sim.step(8)
    return act


def test_terran_builder_pulled_off_construction_is_sent_back():
    g = make_game()
    w = FakeWorld(g, minerals=500)
    w.standard_start(4)
    sim, pm, bm, wm, placer = Sim(w), ProductionManager(), BuildingManager(), WorkerManager(), Placer()
    pm.ensure_build(U.Terran_Supply_Depot, bm)
    for _ in range(40):
        _tick(w, sim, pm, bm, wm, placer)
        if bm.tasks and bm.tasks[0].status == CONSTRUCTING:
            break
    task = bm.tasks[0]
    assert task.status == CONSTRUCTING
    scv = w.get(task.worker_id)
    scv["flags"] &= ~int(UnitFlag.Constructing)       # pulled away mid-construction
    scv["order"] = -1
    back = []
    for _ in range(3):
        act = _tick(w, sim, pm, bm, wm, placer)
        back += [c for c in act.unit_cmds if c.type == C.Right_Click_Unit and c.target == task.building_id]
    assert back and back[0].unit == task.worker_id
