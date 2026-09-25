"""Adapters that run the old mybot/goliath decision code inside slots (parity and baselines)."""
from __future__ import annotations

from typing import Optional

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from blackboard.sections import PlanItem, Squad, SquadOrder
from mybot.combat import CombatCommander
from mybot.policy import Attack, Build, BuildAddon, Rally, Research, Train, Upgrade
from mybot.scout import ScoutManager

from .. import compat

POLICIES = {
    "goliath": ("goliath.policy", "GoliathPolicy"),
    "marine": ("mybot.policy", "ScriptedPolicy"),
}


def _make_policy(name: str):
    import importlib
    mod, cls = POLICIES.get(name, (name.rpartition(":")[0], name.rpartition(":")[2]))
    return getattr(importlib.import_module(mod), cls)()


@register("LegacyPolicyPlanner")
class LegacyPolicyPlanner(Component):
    """Production slot filled by a mybot Policy: its intents become plan items in the order the
    policy emitted them; Attack/Rally goes to `plan.army_order` for the legacy CombatCommander."""

    phase = Phase.PLAN
    reads = ("world", "belief", "strategy")
    writes = ("plan",)

    def __init__(self, policy: str = "goliath") -> None:
        self.policy_name = policy
        self.policy = _make_policy(policy)

    def on_start(self, bb: Blackboard) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def tick(self, bb: Blackboard) -> None:
        s = compat.state(bb)
        intents = self.policy.decide(s)
        plan = bb.plan
        plan.items = []
        plan.army_order = None
        trains: dict[int, PlanItem] = {}
        for it in intents:
            if isinstance(it, Build):
                plan.items.append(PlanItem("build", int(it.unit_type), Priority.PRODUCTION, "policy"))
            elif isinstance(it, Train):
                t = int(it.unit_type)
                if t in trains:
                    trains[t].count += 1
                else:
                    trains[t] = PlanItem("train", t, Priority.PRODUCTION, "policy")
                    plan.items.append(trains[t])
            elif isinstance(it, BuildAddon):
                plan.items.append(PlanItem("addon", int(it.unit_type), Priority.PRODUCTION, "policy"))
            elif isinstance(it, Upgrade):
                plan.items.append(PlanItem("upgrade", int(it.upgrade_type), Priority.PRODUCTION, "policy"))
            elif isinstance(it, Research):
                plan.items.append(PlanItem("research", int(it.tech_type), Priority.PRODUCTION, "policy"))
            elif isinstance(it, (Attack, Rally)):
                plan.army_order = it

    def on_end(self, bb: Blackboard, won: bool) -> None:
        state = bb.cache.get("state") or bb.world.state
        if hasattr(self.policy, "on_game_end"):
            self.policy.on_game_end(state, won)

    def describe(self) -> dict:
        return {"impl": self.name, "policy": self.policy_name}


@register("LegacyScout")
class LegacyScout(Component):
    """mybot ScoutManager: one worker to the enemy start(s)."""

    phase = Phase.ACT
    reads = ("world", "belief")
    writes = ("scouting",)
    priority = Priority.SCOUT
    order = 10

    def __init__(self) -> None:
        self.scout = ScoutManager()

    def on_start(self, bb: Blackboard) -> None:
        self.scout.on_start(bb.game, bb.services["info"])

    def tick(self, bb: Blackboard) -> None:
        s = compat.state(bb)
        wm = compat.worker_manager(bb)
        self.scout.update(s, bb.act, wm, bb.services["info"])
        sc = bb.scouting
        sc.scouts = {}
        if self.scout.worker_id is not None:
            bb.leases.lease(self.scout.worker_id, self.slot, self.priority, "scout", bb.frame)
            dest = self.scout.targets[self.scout.ti] if self.scout.ti < len(self.scout.targets) else (0, 0)
            sc.scouts[self.scout.worker_id] = (dest[0] * 32 + 64, dest[1] * 32 + 48)
        else:
            bb.leases.release_owner(self.slot)
        sc.initial_done = self.scout.done


@register("LegacyCombat")
class LegacyCombat(Component):
    """mybot CombatCommander driven by `plan.army_order` (legacy planners) or the posture."""

    phase = Phase.ACT
    reads = ("world", "belief", "plan", "strategy")
    writes = ("squads",)
    priority = Priority.COMBAT
    order = 40

    def __init__(self) -> None:
        self.combat = CombatCommander()

    def tick(self, bb: Blackboard) -> None:
        s = compat.state(bb)
        order = bb.plan.army_order if bb.plan.army_order is not None else self._order_from_posture(bb, s)
        stance = self.combat.update(s, bb.act, order, bb.services["info"])
        ids = {int(u["id"]) for u in s.army}
        x, y = (order.x, order.y) if order is not None else (0, 0)
        bb.squads.squads = {"main": Squad("main", ids, SquadOrder(stance, x, y), self.priority)}
        bb.squads.army_order = order

    @staticmethod
    def _order_from_posture(bb: Blackboard, s) -> Optional[object]:
        stance = bb.strategy.posture.stance
        if stance in ("attack", "all_in", "contain"):
            if s.enemy_buildings:
                return Attack(s.enemy_buildings[0][1], s.enemy_buildings[0][2])
            if s.enemy_start is not None:
                return Attack(s.enemy_start[0] * 32 + 64, s.enemy_start[1] * 32 + 48)
        return Rally(*s.rally_point)
