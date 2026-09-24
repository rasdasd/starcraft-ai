import os

import numpy as np

from adjutant import engage as E
from adjutant.components.engagement import LearnedEngagement
from adjutant.components.tactics import LearnedTactics, Tactics
from adjutant.learn.combat import (ENGAGE_SPEC, OPTIONS, TACTICS_SPEC, engage_dataset, engage_features,
                                   tactics_dataset, tactics_features, tactics_labels)
from blackboard.models import LinearModel
from blackboard.recorder import Recorder
from bwbot import Race, UnitType as U

from fakes import FakeWorld, Sim, make_game
import test_tactics  # noqa: F401  (registers FixedPosture)

os.environ["BWBOT_LOG"] = "0"


def _bot(slots=None, stance="hold"):
    from adjutant import Adjutant
    g = make_game(enemy_race=Race.Zerg)
    slots = {"strategy": {"impl": "FixedPosture", "stance": stance}, **(slots or {})}
    bot = Adjutant("planned", slots=slots)
    rec = Recorder(enabled=False)
    rows = []
    rec.record = lambda slot, kind, frame, every=0, **data: rows.append((slot, kind, dict(data, f=frame))) or True
    bot.recorder = rec
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    return bot, w, g, rows


def _const_model(spec, feature: str, weight: float, task: str) -> LinearModel:
    W = np.zeros((spec.dim, 1), np.float32)
    W[spec.names.index(feature), 0] = weight
    return LinearModel(W, np.zeros(1, np.float32), task=task, mu=np.zeros(spec.dim), sd=np.ones(spec.dim),
                       labels=("y",), spec=spec)


def test_engage_features_match_spec():
    g = make_game(enemy_race=Race.Zerg)
    t = E.TypeTable(g)
    own = E.side_from_counts({int(U.Terran_Marine): 4}, t)
    enemy = E.side_from_counts({int(U.Zerg_Zergling): 6}, t)
    x = engage_features(own, enemy, E.evaluate(own, enemy), {"frame": 24 * 300, "b_army": 10, "own_army": 4})
    assert x.shape == (ENGAGE_SPEC.dim,) and np.isfinite(x).all()
    assert x[ENGAGE_SPEC.names.index("hidden_supply")] > 0          # 10 believed vs 3 visible
    for o in OPTIONS:
        assert tactics_features({}, o, None).shape == (TACTICS_SPEC.dim,)


def test_learned_engagement_falls_back_and_uses_the_model(tmp_path):
    bot, w, g, rows = _bot({"engagement": {"impl": "LearnedEngagement", "model": str(tmp_path / "none.npz")}})
    eng = next(c for c in bot.sched.components if isinstance(c, LearnedEngagement))
    assert eng.model is None
    own = E.side_from_counts({int(U.Terran_Marine): 8}, eng.own_table)
    enemy = E.side_from_counts({int(U.Zerg_Zergling): 2}, eng.enemy_table)
    assert eng.evaluate(own, enemy).win_prob == E.evaluate(own, enemy).win_prob
    # a model that only looks at "hidden_supply": lots of unseen enemy army -> the fight looks bad
    eng.model = _const_model(ENGAGE_SPEC, "hidden_supply", -20.0, "binary")
    eng.ctx = {"frame": 24 * 400, "b_army": 40.0, "own_army": 8.0}
    assert eng.evaluate(own, enemy).win_prob < 0.01
    assert bot.bb.services["engage"] is eng


def test_fight_rows_carry_features_and_build_a_dataset(tmp_path):
    bot, w, g, rows = _bot()
    sx, sy = g.self_player.start_location
    x0, y0 = sx * 32 + 400, sy * 32 + 400
    [w.add(U.Terran_Marine, x0 + i * 16, y0) for i in range(8)]
    lings = [w.add(U.Zerg_Zergling, x0 + 120 + i * 10, y0 + 40, player=g.enemy_id) for i in range(4)]
    from bwbot import EventType
    from bwbot.observation import Event
    sim = Sim(w)
    sim.run(bot, 48)
    for uid in lings:
        w.remove(uid)
        w.events.append(Event(EventType.UnitDestroy, uid, g.enemy_id, False, "", (x0, y0)))
    sim.run(bot, 24 * 8)
    fights = [d for s, k, d in rows if (s, k) == ("engagement", "fight")]
    assert fights and len(fights[0]["snap"]["x"]) == ENGAGE_SPEC.dim and fights[0]["snap"]["fh"] == ENGAGE_SPEC.hash
    # the dataset reader on a recorded game
    rec = Recorder(directory=tmp_path, enabled=True)
    rec.start({"game_id": "g1", "side": "a"})
    for f in fights:
        rec.record("engagement", "fight", 100, **{k: v for k, v in f.items() if k != "f"})
    rec.finish(True, 200)
    X, y, groups, skipped, n = engage_dataset([tmp_path])
    assert X.shape == (len(fights), ENGAGE_SPEC.dim) and y.tolist() == [1.0] * len(fights)


def test_tactics_labels_are_value_trade_over_the_horizon():
    rows = [{"f": 0, "lost": [0, 0], "value": 800}, {"f": 24 * 30, "lost": [100, 0], "value": 700},
            {"f": 24 * 50, "lost": [100, 500], "value": 700}]
    labs = tactics_labels(rows, [100, 500], True, 24 * 45, 0.0)
    assert labs[0] == (500 - 100) / 1000
    assert labs[1] == 500 / 900
    assert labs[2] == 0.0
    assert tactics_labels(rows[:1], None, None, 24 * 45, 0.3) == [None]


def test_tactics_logs_decisions_with_losses(tmp_path):
    bot, w, g, rows = _bot()
    sx, sy = g.self_player.start_location
    ids = [w.add(U.Terran_Vulture, sx * 32 + 200 + i * 20, sy * 32 + 200) for i in range(3)]
    ling = w.add(U.Zerg_Zergling, sx * 32 + 1500, sy * 32 + 1500, player=g.enemy_id)
    sim = Sim(w)
    sim.run(bot, 24 * 6)
    from bwbot import EventType
    from bwbot.observation import Event
    w.remove(ids[0])
    w.events.append(Event(EventType.UnitDestroy, ids[0], g.self_id, False, "", (0, 0)))
    w.remove(ling)
    w.events.append(Event(EventType.UnitDestroy, ling, g.enemy_id, False, "", (0, 0)))
    sim.run(bot, 24 * 6)
    dec = [d for s, k, d in rows if (s, k) == ("tactics", "decision")]
    assert len(dec) >= 2 and all(d["opt"] == "hold" and len(d["x"]) == TACTICS_SPEC.dim for d in dec)
    t = next(c for c in bot.sched.components if isinstance(c, Tactics))
    assert t.lost == [75.0, 50.0]                       # fake prices: vulture 75, zergling 50
    assert dec[-1]["lost"] == [75, 50]
    rec = Recorder(directory=tmp_path, enabled=True)
    rec.start({"game_id": "g1", "side": "a"})
    for d in dec:
        rec.record("tactics", "decision", d["f"], **{k: v for k, v in d.items() if k != "f"})
    rec.record("tactics", "final", 24 * 60, lost=[150, 50])
    rec.finish(False, 24 * 60)
    X, y, groups, opts, skipped, n = tactics_dataset([tmp_path], horizon_s=45, win_weight=0.3)
    assert len(X) == len(dec) and set(opts) == {"hold"} and (y < 0).all()


def test_learned_tactics_overrides_scripted_hold_when_the_model_prefers_attack():
    bot, w, g, rows = _bot({"tactics": {"impl": "LearnedTactics", "margin": 0.1}})
    lt = next(c for c in bot.sched.components if isinstance(c, LearnedTactics))
    lt.model = _const_model(TACTICS_SPEC, "opt_attack", 1.0, "regression")
    sx, sy = g.self_player.start_location
    [w.add(U.Terran_Vulture, sx * 32 + 200 + i * 20, sy * 32 + 200) for i in range(4)]
    Sim(w).run(bot, 48)
    main = bot.bb.squads.squads["main"]
    ex, ey = g.players[1].start_location
    assert main.order.kind == "attack" and (main.order.x, main.order.y) == (ex * 32 + 64, ey * 32 + 48)
    assert lt.scores["attack"] > lt.scores["hold"]
    dec = [d for s, k, d in rows if (s, k) == ("tactics", "decision")]
    assert dec and dec[-1]["opt"] == "attack" and dec[-1]["learned"] is True
    # no model: exactly the scripted hold
    lt.model = None
    lt.choice = None
    Sim(w).run(bot, 24)
    assert bot.bb.squads.squads["main"].order.kind == "hold"
