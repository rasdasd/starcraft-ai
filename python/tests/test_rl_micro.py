import os

import numpy as np

from adjutant.components.rl_micro import RLMicro
from adjutant.learn.micro import ACTIONS, MICRO_SPEC, MICRO_STATE, STATE_NAMES, fitted_q, micro_dataset, sa
from blackboard.models import LinearModel
from blackboard.recorder import Recorder
from bwbot import EventType, Race, UnitType as U
from bwbot.enums import UnitCommandType as C
from bwbot.observation import Event

from fakes import FakeWorld, make_game
from test_tactics import Recording  # noqa: F401  (also registers FixedPosture)

os.environ["BWBOT_LOG"] = "0"


def _bot(micro: dict):
    from adjutant import Adjutant
    g = make_game(enemy_race=Race.Zerg)
    bot = Adjutant("planned", slots={"strategy": {"impl": "FixedPosture", "stance": "hold"},
                                     "micro": {"impl": "RLMicro", **micro}})
    rec = Recorder(enabled=False)
    rows = []
    rec.record = lambda slot, kind, frame, every=0, **data: rows.append((slot, kind, dict(data, f=frame))) or True
    bot.recorder = rec
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    rl = next(c for c in bot.sched.components if isinstance(c, RLMicro))
    return bot, w, g, rows, rl


def _fight(w, g):
    sx, sy = g.self_player.start_location
    x0, y0 = sx * 32 + 300, sy * 32 + 300
    marines = [w.add(U.Terran_Marine, x0 + i * 16, y0) for i in range(4)]
    lings = [w.add(U.Zerg_Zergling, x0 + i * 16, y0 + 60, player=g.enemy_id) for i in range(4)]
    return marines, lings


def test_rl_micro_without_model_plays_scripted_focus_and_logs_transitions():
    bot, w, g, rows, rl = _bot({"epsilon": 0.0, "log_frac": 1.0, "decide_frames": 8})
    assert rl.model is None
    marines, lings = _fight(w, g)
    sim = Recording(w)
    sim.run(bot, 48)
    atk = [c for c in sim.all() if c.unit in marines and c.type == C.Attack_Unit]
    assert atk and all(c.target in lings for c in atk)
    assert rl.counts[ACTIONS.index("focus")] > 0 and sum(rl.counts) == rl.counts[ACTIONS.index("focus")]
    # a marine takes damage and a zergling dies, then the fight ends
    next(u for u in w.units if u["id"] == marines[0])["hit_points"] = 10
    w.remove(lings[0])
    w.events.append(Event(EventType.UnitDestroy, lings[0], g.enemy_id, False, "", (0, 0)))
    sim.run(bot, 16)
    for uid in lings[1:]:
        w.remove(uid)
    sim.run(bot, 24 * 4)
    steps = [d for s, k, d in rows if (s, k) == ("micro", "step")]
    assert steps and all(len(d["x"]) == MICRO_STATE and d["fh"] == MICRO_SPEC.hash for d in steps)
    assert any(d["x2"] is None for d in steps) and any(d["x2"] is not None for d in steps)
    assert not rl.pending                                  # every open transition closed once the fight ended
    assert any(d["r"] != 0 for d in steps)


def test_rl_micro_follows_the_q_model():
    bot, w, g, rows, rl = _bot({"epsilon": 0.0, "log_frac": 0.0})
    W = np.zeros((MICRO_SPEC.dim, 1), np.float32)
    W[MICRO_SPEC.names.index("act_back"), 0] = 1.0
    rl.model = LinearModel(W, np.zeros(1, np.float32), task="regression", mu=np.zeros(MICRO_SPEC.dim),
                           sd=np.ones(MICRO_SPEC.dim), labels=("q",), spec=MICRO_SPEC)
    marines, lings = _fight(w, g)
    sim = Recording(w)
    sim.run(bot, 48)
    assert rl.counts[ACTIONS.index("back")] == sum(rl.counts) > 0
    moves = [c for c in sim.all() if c.unit in marines and c.type == C.Move]
    assert moves and not [c for c in sim.all() if c.unit in marines and c.type == C.Attack_Unit]
    assert not [r for r in rows if r[0] == "micro"]         # log_frac 0: nothing recorded


def test_fitted_q_learns_state_dependent_actions(tmp_path):
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.random((n, MICRO_STATE)).astype(np.float32)
    A = rng.integers(0, len(ACTIONS), n)
    hp = X[:, STATE_NAMES.index("hp")]
    # healthy units should fight (focus), hurt ones back off; everything else is neutral
    R = np.where(A == ACTIONS.index("focus"), np.where(hp > 0.5, 1.0, -1.0),
                 np.where(A == ACTIONS.index("back"), np.where(hp > 0.5, -0.5, 0.5), 0.0)).astype(np.float32)
    done = np.ones(n, bool)
    m, rep = fitted_q(X, A, R, X, done, gamma=0.8, iters=2, hidden=(16,), epochs=80, log=lambda *_: None)
    healthy, hurt = np.full(MICRO_STATE, 0.5, np.float32), np.full(MICRO_STATE, 0.5, np.float32)
    healthy[STATE_NAMES.index("hp")], hurt[STATE_NAMES.index("hp")] = 0.95, 0.05
    q = lambda x: m.predict(sa(np.stack([x] * len(ACTIONS)), np.arange(len(ACTIONS)))).reshape(-1)
    assert ACTIONS[int(np.argmax(q(healthy)))] == "focus"
    assert ACTIONS[int(np.argmax(q(hurt)))] == "back"
    # the dataset reader
    rec = Recorder(directory=tmp_path, enabled=True)
    rec.start({"game_id": "g1", "side": "a"})
    for i in range(5):
        rec.record("micro", "step", i * 12, uid=1, a=int(A[i]), r=float(R[i]), x=X[i].tolist(),
                   x2=None if i == 4 else X[i + 1].tolist(), fh=MICRO_SPEC.hash, dt=12)
    rec.finish(True, 100)
    Xd, Ad, Rd, X2d, dd, groups, skipped, games = micro_dataset([tmp_path])
    assert Xd.shape == (5, MICRO_STATE) and dd.tolist() == [False] * 4 + [True] and games == 1
