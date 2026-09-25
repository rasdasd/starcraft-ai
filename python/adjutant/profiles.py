"""Built-in profiles (which implementation fills each slot). Override with a JSON file via
`BWBOT_PROFILE_FILE` / `BWBOT_PROFILE_JSON`, or pick one with `BWBOT_PROFILE=<name or path>`.

`adjutant` is the bot; every other profile swaps a few of its slots (a different planner, strategy
selector, or a learned / data-collecting variant). The strategy selector picks among the build
templates for our race (`adjutant/builds/<race>/`).
"""
from __future__ import annotations

from . import components  # noqa: F401  (register implementations)

ADJUTANT = {
    "name": "adjutant",
    "slots": {
        "perception": {"impl": "Perception"},
        "meta": {"impl": "MetaAnalysis"},
        "belief": {"impl": "ScriptedBelief"},
        "engagement": {"impl": "Engagement"},
        "strategy": {"impl": "RuleSelector"},
        "production": {"impl": "GreedyPlanner"},
        "macro": {"impl": "Macro"},
        "scouting": {"impl": "Scouting"},
        "repair": {"impl": "Repair"},
        "tactics": {"impl": "Tactics"},
        "micro": {"impl": "Micro"},
        "crisis": {"impl": "Crisis"},
        "worker_defense": {"impl": "WorkerDefense"},
        "report": {"impl": "Snapshots"},
    },
    "config": {"time_budget_ms": 40},
}

BUILTINS: dict[str, dict] = {
    "adjutant": ADJUTANT,
    # one fixed build template (see adjutant/builds); e.g. BWBOT_PROFILE_JSON='{"slots": {"strategy": {"template": "x"}}}'
    "scripted": {"base": "adjutant", "name": "scripted", "slots": {
        "strategy": {"impl": "ScriptedStrategy", "template": "mech_expand"}}},
    # production v2: build-order search over the economy simulator
    "search": {"base": "adjutant", "name": "search", "slots": {"production": {"impl": "SearchPlanner"}}},
    # data collection for the strategy win model
    "explore": {"base": "adjutant", "name": "explore", "slots": {"strategy": {"impl": "Explore"}}},
    # belief training data: full map information -> `truth`, hidden again by Perception's fog filter
    "truth": {"base": "explore", "name": "truth", "config": {"complete_map_information": True},
              "slots": {"belief_log": {"impl": "BeliefLog"}}},
    # combat training data: random tactics options 15% of the time (fight rows are always logged)
    "explore_combat": {"base": "adjutant", "name": "explore_combat", "slots": {
        "tactics": {"impl": "LearnedTactics", "epsilon": 0.15}}},
    # experimental RL micro: explore_micro collects transitions (and uses micro.npz if present), rl_micro plays it
    "explore_micro": {"base": "adjutant", "name": "explore_micro", "slots": {
        "micro": {"impl": "RLMicro", "epsilon": 0.2}}},
    "rl_micro": {"base": "adjutant", "name": "rl_micro", "slots": {
        "micro": {"impl": "RLMicro", "epsilon": 0.0, "margin": 0.1, "log_frac": 0.0}}},
    "learned_combat": {"base": "adjutant", "name": "learned_combat", "slots": {
        "engagement": {"impl": "LearnedEngagement"},
        "tactics": {"impl": "LearnedTactics"}}},
    "learned": {"base": "adjutant", "name": "learned", "slots": {
        "belief": {"impl": "LearnedBelief"},
        "engagement": {"impl": "LearnedEngagement"},
        "tactics": {"impl": "LearnedTactics"},
        "strategy": {"impl": "LearnedStrategy", "model": "strategy.npz", "mode": "select", "epsilon": 0.05}}},
    "blend": {"base": "adjutant", "name": "blend", "slots": {
        "strategy": {"impl": "LearnedStrategy", "model": "strategy.npz", "mode": "blend"}}},
}
