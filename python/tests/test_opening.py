from types import SimpleNamespace

import numpy as np

from bwbot import UnitType as U
from mybot.opening import Opening, OpeningStep


def _state(counts: dict, supply: int):
    return SimpleNamespace(count=lambda t: counts.get(int(t), 0), supply_used=supply)


def test_starting_units_do_not_satisfy_opening_steps():
    base = np.zeros(256, dtype=np.int32)
    base[int(U.Zerg_Hatchery)] = 1
    base[int(U.Zerg_Overlord)] = 1
    op = Opening([OpeningStep(9, U.Zerg_Overlord), OpeningStep(12, U.Zerg_Hatchery)], base=base)
    have = {int(U.Zerg_Hatchery): 1, int(U.Zerg_Overlord): 1}
    assert op.next_build(_state(have, 8)) is None and op.i == 0
    assert op.next_build(_state(have, 9)) == U.Zerg_Overlord
    have[int(U.Zerg_Overlord)] = 2
    assert op.next_build(_state(have, 12)) == U.Zerg_Hatchery
    have[int(U.Zerg_Hatchery)] = 2
    assert op.next_build(_state(have, 12)) is None and op.done
