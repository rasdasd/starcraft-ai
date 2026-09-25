from bwbot import Actions, Race, UnitType as U

from fakes import FakeWorld, Sim, make_game
from mybot.state import Memory, perceive
from mybot.workers import WorkerManager


def test_gas_workers_go_back_to_minerals_while_gas_is_over_banked():
    g = make_game(self_race=Race.Terran, enemy_race=Race.Terran)
    w = FakeWorld(g, minerals=50)
    w.standard_start(10)
    sx, sy = g.self_player.start_location
    w.add(U.Terran_Refinery, sx * 32 + 256, sy * 32)
    wm, sim = WorkerManager(), Sim(w)

    def step():
        act = Actions()
        wm.update(perceive(w.observe(), g, Memory()), act)
        sim.apply(act)
        return len(wm.gas)

    assert step() == 3
    w.gas, w.minerals = 800, 100
    assert step() == 0 and step() == 0
    w.gas = 400
    assert step() == 0                        # hysteresis: not until it is spent down
    w.gas = 100
    assert step() == 3
