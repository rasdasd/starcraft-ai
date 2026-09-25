"""On-screen debug overlay: one line per section summary, plus slot timings and fallbacks."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bwbot.commands import Actions

if TYPE_CHECKING:
    from .board import Blackboard
    from .scheduler import Scheduler

SECTIONS = ("world", "meta", "belief", "threats", "strategy", "engagements", "squads", "plan", "scouting")


def draw(bb: "Blackboard", sched: "Scheduler", act: Actions, x: int = 10, y: int = 10, title: str = "") -> None:
    line = 12
    act.draw_text_screen(x, y, f"{title} dec {bb.decision} {sched.last_decision_ms:.1f}ms "
                               f"(max {sched.max_decision_ms:.1f}) leases {len(bb.leases)} req {len(bb.requests)} "
                               f"cmd-drop {bb.bus.dropped}")
    y += line
    for name in SECTIONS:
        sec = bb.sections.get(name)
        summary = getattr(sec, "summary", None)
        if summary is None:
            continue
        try:
            text = summary()
        except Exception as e:     # a HUD bug must never break the decision
            text = f"<{type(e).__name__}>"
        act.draw_text_screen(x, y, f"{name}: {text}"[:160])
        y += line
    slots = []
    for key, st in sched.stats.items():
        tag = f"{key} {st.last_ms:.1f}"
        if st.replaced_by:
            tag += f"->{st.replaced_by}"
        elif st.failures:
            tag += f" !{st.failures}"
        slots.append(tag)
    act.draw_text_screen(x, y, ("slots: " + "  ".join(slots))[:200])
