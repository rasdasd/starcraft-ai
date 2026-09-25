"""REPORT phase: periodic snapshots for training/analysis (strategy/belief/fight rows are written
by their own components)."""
from __future__ import annotations

from blackboard import Blackboard, Component, Phase
from blackboard.profile import register


@register("Snapshots")
class Snapshots(Component):
    phase = Phase.REPORT
    reads = ("world", "belief", "strategy", "squads", "truth", "plan", "macro")
    period_frames = 24 * 10

    def tick(self, bb: Blackboard) -> None:
        w, b, st = bb.world, bb.belief, bb.strategy
        row = dict(
            min=w.minerals, gas=w.gas, sup=w.supply_used, cap=w.supply_total, wrk=len(w.workers),
            army=w.army_supply, inc=[round(w.income_minerals), round(w.income_gas)], bases=len(w.depots),
            tmpl=st.template, stance=st.posture.stance,
            b_army=round(b.army_supply, 1), b_air=round(b.air, 1), b_open=b.opening,
            b_counts={str(k): round(v, 1) for k, v in b.counts.items() if v},
            plan=[f"{it.kind}:{bb.game.type_name(it.type_id) if it.kind in ('build', 'addon', 'train') else it.type_id}"
                  f"@{it.priority}" for it in bb.plan.items[:6]],
            notes=list(bb.plan.notes)[:6],
            own={_short(bb, t): int(n) for t, n in enumerate(w.counts) if n},
        )
        m = bb.macro
        row["jobs"] = list(m.jobs)
        row["blocked"] = list(m.blocked)[:6]
        row["sat"] = [[b.base_id, b.miners, 2 * b.patches, b.gas_workers] for b in m.bases]
        row["mbases"] = m.base_count
        t = bb.truth
        if t.enabled:
            row["t_counts"] = {str(k): v for k, v in t.counts.items()}
            row["t_army"] = t.army_supply
            row["t_bases"] = len(t.bases)
        bb.record("report", "snap", **row)


def _short(bb: Blackboard, type_id) -> str:
    name = bb.game.type_name(int(type_id))
    for p in ("Terran_", "Protoss_", "Zerg_"):
        if name.startswith(p):
            return name[len(p):]
    return name
