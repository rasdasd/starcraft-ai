"""REPORT phase: periodic snapshots for training/analysis (strategy/belief/fight rows are written
by their own components)."""
from __future__ import annotations

from blackboard import Blackboard, Component, Phase
from blackboard.profile import register


@register("Snapshots")
class Snapshots(Component):
    phase = Phase.REPORT
    reads = ("world", "belief", "strategy", "squads", "truth")
    period_frames = 24 * 10

    def tick(self, bb: Blackboard) -> None:
        w, b, st = bb.world, bb.belief, bb.strategy
        row = dict(
            min=w.minerals, gas=w.gas, sup=w.supply_used, cap=w.supply_total, wrk=len(w.workers),
            army=w.army_supply, inc=[round(w.income_minerals), round(w.income_gas)], bases=len(w.depots),
            tmpl=st.template, stance=st.posture.stance,
            b_army=round(b.army_supply, 1), b_air=round(b.air, 1), b_open=b.opening,
            b_counts={str(k): round(v, 1) for k, v in b.counts.items() if v},
        )
        t = bb.truth
        if t.enabled:
            row["t_counts"] = {str(k): v for k, v in t.counts.items()}
            row["t_army"] = t.army_supply
            row["t_bases"] = len(t.bases)
        bb.record("report", "snap", **row)
