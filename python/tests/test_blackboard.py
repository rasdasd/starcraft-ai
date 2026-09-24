import pytest

from blackboard import Blackboard, Component, ContractError, Phase, Priority, Scheduler, validate
from blackboard.bot import BlackboardBot
from blackboard.sections import STANDARD_SCHEMA, Belief

from fakes import FakeWorld, make_game


class Rec(Component):
    def __init__(self, slot, phase, reads=(), writes=(), log=None, period=0, triggers=(), order=0, boom=False):
        self.slot, self.phase, self.reads, self.writes = slot, phase, tuple(reads), tuple(writes)
        self.period_frames, self.triggers, self.order = period, tuple(triggers), order
        self.log = log if log is not None else []
        self.boom = boom

    def tick(self, bb):
        self.log.append((self.slot, bb.frame))
        if self.boom:
            raise RuntimeError("boom")


def _board():
    g = make_game()
    w = FakeWorld(g)
    w.standard_start()
    return g, w, Blackboard(g)


def _decide(bb, sched, w, frame):
    from bwbot import Actions
    act = Actions()
    bb.begin_decision(w.observe(frame), act)
    sched.run(bb)
    return act


def test_sections_are_typed_and_accessible():
    _, _, bb = _board()
    assert isinstance(bb.belief, Belief)
    assert set(bb.sections) == set(STANDARD_SCHEMA)
    with pytest.raises(AttributeError):
        _ = bb.nope


def test_validate_rejects_two_writers_and_unknown_sections():
    a = Rec("a", Phase.SENSE, writes=["world"])
    b = Rec("b", Phase.SENSE, writes=["world"])
    with pytest.raises(ContractError):
        validate([a, b], STANDARD_SCHEMA)
    with pytest.raises(ContractError):
        validate([Rec("c", Phase.SENSE, reads=["bogus"])], STANDARD_SCHEMA)
    with pytest.raises(ContractError):
        validate([Rec("d", Phase.DECIDE, reads=["truth"])], STANDARD_SCHEMA)
    validate([Rec("e", Phase.REPORT, reads=["truth"])], STANDARD_SCHEMA)


def test_phase_and_dependency_order():
    log = []
    comps = [
        Rec("act", Phase.ACT, reads=["plan"], log=log),
        Rec("tactics", Phase.DECIDE, reads=["strategy", "engagements"], writes=["squads"], log=log),
        Rec("engage", Phase.DECIDE, reads=["belief"], writes=["engagements"], log=log, order=5),
        Rec("strategy", Phase.DECIDE, reads=["belief"], writes=["strategy"], log=log, order=9),
        Rec("perception", Phase.SENSE, writes=["world"], log=log),
        Rec("plan", Phase.PLAN, reads=["strategy"], writes=["plan"], log=log),
    ]
    _, w, bb = _board()
    s = Scheduler(comps, strict=True)
    s.start(bb)
    _decide(bb, s, w, 0)
    order = [n for n, _ in log]
    assert order[0] == "perception"
    assert order.index("strategy") < order.index("tactics")
    assert order.index("engage") < order.index("tactics")
    assert order[-2:] == ["plan", "act"]


def test_periods_and_triggers():
    log = []
    c = Rec("slow", Phase.DECIDE, period=48, triggers=["enemy_seen"], log=log)
    _, w, bb = _board()
    s = Scheduler([c], strict=True)
    s.start(bb)
    for f in (0, 8, 16, 48, 56):
        _decide(bb, s, w, f)
    assert [f for _, f in log] == [0, 48]
    bb.raise_event("enemy_seen")
    _decide(bb, s, w, 64)
    assert [f for _, f in log] == [0, 48, 64]


def test_failure_switches_to_fallback():
    log = []
    good = Rec("strategy", Phase.DECIDE, writes=["strategy"], log=log)
    bad = Rec("strategy", Phase.DECIDE, writes=["strategy"], log=log, boom=True)
    bad.fallback = good
    _, w, bb = _board()
    s = Scheduler([bad], max_failures=2)
    s.start(bb)
    for f in range(0, 40, 8):
        _decide(bb, s, w, f)
    assert s.stats["strategy"].replaced_by == "Rec"
    assert s.slot("strategy") is good
    assert bb.stats["fallback.strategy"] == 1


def test_replacing_unowned_section_is_a_violation():
    class Rogue(Component):
        slot, phase, writes = "rogue", Phase.DECIDE, ("squads",)

        def tick(self, bb):
            bb.put("belief", Belief())

    _, w, bb = _board()
    s = Scheduler([Rogue()], strict=True)
    s.start(bb)
    with pytest.raises(ContractError):
        _decide(bb, s, w, 0)


def test_time_budget_defers_periodic_components():
    import time

    class Sleepy(Component):
        slot, phase, period_frames = "sleepy", Phase.DECIDE, 8

        def tick(self, bb):
            time.sleep(0.02)

    log = []
    late = Rec("late", Phase.PLAN, period=8, log=log)
    _, w, bb = _board()
    s = Scheduler([Sleepy(), late], time_budget_ms=5)
    s.start(bb)
    _decide(bb, s, w, 0)      # first decision: nothing is deferred
    _decide(bb, s, w, 8)
    assert len(log) == 1 and s.stats["late"].deferred == 1


def test_blackboard_bot_runs_components():
    log = []
    g, w, _ = _board()

    def factory():
        return [Rec("perception", Phase.SENSE, writes=["world"], log=log)]

    from blackboard.recorder import Recorder
    bot = BlackboardBot(factory, show_hud=True, strict=True, recorder=Recorder(enabled=False))
    bot.game = g
    bot.on_start(g)
    from bwbot import Actions
    act = Actions()
    bot.on_frame(w.observe(0), act)
    assert log == [("perception", 0)]
    assert act.draws                     # HUD drawn
    bot.on_end(True)
