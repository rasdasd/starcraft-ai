"""Built-in profiles (which implementation fills each slot). Override with a JSON file via
`BWBOT_PROFILE_FILE` / `BWBOT_PROFILE_JSON`, or pick one with `BWBOT_PROFILE=<name or path>`."""
from __future__ import annotations

from . import components  # noqa: F401  (register implementations)

PARITY = {
    "name": "parity",
    "slots": {
        "perception": {"impl": "Perception"},
        "meta": {"impl": "MetaAnalysis"},
        "belief": {"impl": "LegacyBelief"},
        "strategy": {"impl": "ScriptedStrategy", "template": "goliath_1fact"},
        "production": {"impl": "LegacyPolicyPlanner", "policy": "goliath"},
        "construction": {"impl": "Construction"},
        "scouting": {"impl": "LegacyScout"},
        "repair": {"impl": "Repair", "unit_types": ["Terran_Goliath"], "structures": False},
        "economy": {"impl": "Economy"},
        "tactics": {"impl": "LegacyCombat"},
        "report": {"impl": "Snapshots"},
    },
    "config": {"time_budget_ms": 40},
}

BUILTINS: dict[str, dict] = {
    "parity": PARITY,
    "adjutant": {"base": "parity", "name": "adjutant"},
    # strategy goal -> greedy tech-tree planner (instead of the goliath policy's own build order)
    "planned": {"base": "parity", "name": "planned", "slots": {
        "strategy": {"impl": "ScriptedStrategy", "template": "mech_expand"},
        "production": {"impl": "GreedyPlanner"},
        "repair": {"impl": "Repair"},
    }},
    "rules": {"base": "planned", "name": "rules", "slots": {"strategy": {"impl": "RuleSelector"}}},
    # data collection for the strategy win model
    "explore": {"base": "planned", "name": "explore", "slots": {"strategy": {"impl": "Explore"}}},
    "learned": {"base": "planned", "name": "learned", "slots": {
        "strategy": {"impl": "LearnedStrategy", "model": "strategy.npz", "mode": "select", "epsilon": 0.05}}},
    "blend": {"base": "planned", "name": "blend", "slots": {
        "strategy": {"impl": "LearnedStrategy", "model": "strategy.npz", "mode": "blend"}}},
}
