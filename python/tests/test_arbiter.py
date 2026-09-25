from bwbot import Actions

from blackboard import Blackboard, Budget, Priority, RequestQueue, UnitLeases
from blackboard.arbiter import CommandBus, Request


def test_leases_priority_and_release():
    L = UnitLeases()
    assert L.lease(5, "tactics", Priority.COMBAT)
    assert not L.lease(5, "scout", Priority.SCOUT)
    assert L.lease(5, "crisis", Priority.CRISIS)
    assert L.owner(5) == "crisis"
    L.release(5, owner="tactics")          # not the owner: no-op
    assert L.owner(5) == "crisis"
    L.lease(6, "crisis", Priority.CRISIS)
    L.release_owner("crisis")
    assert L.owner(5) is None and L.owner(6) is None
    L.lease(7, "a", 1)
    L.sync([8])
    assert len(L) == 0


def test_budget_reservations_block_lower_priority():
    b = Budget()
    b.reset(300, 100, 10)
    b.reserve("cc", 250, 0, priority=Priority.CONSTRUCTION)
    assert b.can_afford(100, 0, priority=Priority.CRISIS)
    assert not b.can_afford(100, 0, priority=Priority.PRODUCTION)
    assert b.can_afford(50, 0, priority=Priority.PRODUCTION)
    b.spend(50, 0, 1)
    assert b.available(Priority.CRISIS) == (250, 100, 9)
    b.release("cc")
    assert b.can_afford(200, 0)


def test_request_queue_dedup_ttl_and_order():
    q = RequestQueue()
    q.post(Request("production", "crisis", Priority.CRISIS, type_id=125, item="build"), frame=0, ttl=10)
    q.post(Request("production", "strategy", Priority.PRODUCTION, type_id=0, item="train"), frame=0, ttl=100)
    q.post(Request("production", "crisis", Priority.CRISIS, type_id=125, item="build"), frame=5, ttl=10)
    assert len(q) == 2
    assert [r.source for r in q.active("production")] == ["crisis", "strategy"]
    q.expire(16)
    assert [r.source for r in q.active("production")] == ["strategy"]
    q.withdraw("strategy")
    assert len(q) == 0


def test_scoped_actions_respect_leases_and_flush_sorts():
    L = UnitLeases()
    bus = CommandBus(L)
    root = Actions()
    L.lease(1, "tactics", Priority.COMBAT)
    workers = bus.scope(root, "workers", Priority.ECONOMY)
    tactics = bus.scope(root, "tactics", Priority.COMBAT)
    crisis = bus.scope(root, "crisis", Priority.CRISIS)
    workers.move(1, 0, 0)                  # dropped: unit leased by tactics
    workers.move(2, 0, 0)
    tactics.attack_move(1, 10, 10)
    tactics.move(2, 5, 5)                  # conflicts with workers on unit 2; tactics wins at flush
    crisis.move(3, 1, 1)
    root.stop(4)                           # untagged -> NORMAL
    assert bus.dropped == 1
    bus.flush(root)
    got = [(c.unit, getattr(c, "owner", "")) for c in root.unit_cmds]
    assert got == [(3, "crisis"), (1, "tactics"), (2, "tactics"), (4, "")]
    assert bus.conflicts == 1
    # APM trim keeps the most important
    assert [c.unit for c in root.unit_cmds[:2]] == [3, 1]


def test_same_owner_multiple_commands_kept():
    bus = CommandBus(UnitLeases())
    root = Actions()
    p = bus.scope(root, "production", Priority.PRODUCTION)
    p.train(9, 0)
    p.train(9, 0)
    bus.flush(root)
    assert len(root.unit_cmds) == 2


def test_board_request_helper():
    from fakes import FakeWorld, make_game
    g = make_game()
    w = FakeWorld(g, minerals=400)
    w.standard_start()
    bb = Blackboard(g)
    bb.begin_decision(w.observe(0), Actions())
    bb.request("production", "crisis", ttl=24, type_id=125, item="build", priority=Priority.CRISIS)
    assert bb.requests.active("production")[0].type_id == 125
    assert bb.budget.minerals == 400
