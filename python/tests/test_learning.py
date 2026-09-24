import json

import numpy as np
import pytest

from blackboard import Component, Phase
from blackboard.profile import REGISTRY, build, register, resolve
from blackboard.recorder import Recorder, iter_games
from blackboard.datasets import matrix, outcome, rows
from blackboard.models import FeatureMismatch, FeatureSpec, load_model, try_load
from blackboard.train import fit

SPEC = FeatureSpec("toy", 1, ("a", "b", "c"))


def test_recorder_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BWBOT_GAME_ID", "g1")
    r = Recorder(tmp_path, enabled=True)
    r.start({"map": "m", "seed": 7})
    assert r.record("strategy", "dec", 0, x=np.arange(3), choice=1)
    assert not r.record("strategy", "dec", 4, every=24, x=[0, 0, 0], choice=0)
    assert r.record("belief", "snap", 8, every=24, army=3.5)
    assert not r.record("belief", "snap", 16, every=24, army=3.5)
    r.finish(True, 100)
    games = list(iter_games(tmp_path))
    assert len(games) == 1 and games[0]["header"]["seed"] == 7 and games[0]["header"]["game_id"] == "g1"
    pairs = list(rows([tmp_path], "strategy", "dec"))
    X, y, g = matrix(pairs, "x", lambda gm, row: float(outcome(gm)))
    assert X.shape[1] == 3 and y[0] == 1.0 and g[0] == "g1"


def test_truncated_log_is_tolerated(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"t": "header", "game_id": "x"}) + "\n" + '{"t": "rec", "slot"', encoding="utf-8")
    assert list(iter_games(tmp_path)) == []
    assert len(list(iter_games(tmp_path, finished_only=False))) == 1


@pytest.mark.parametrize("hidden", [(), (16,)])
def test_fit_binary_save_load(tmp_path, hidden):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 3)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(np.float32)
    m, rep = fit(X, y, task="binary", hidden=hidden, spec=SPEC, epochs=100, lr=0.03)
    assert rep.val_metric > 0.9
    path = m.save(tmp_path / "m.npz")
    m2 = load_model(path, SPEC)
    np.testing.assert_allclose(m.predict(X[:5]), m2.predict(X[:5]), rtol=1e-5)
    assert 0.0 <= float(m2.predict(X[0])) <= 1.0


def test_fit_softmax_and_regression():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 3)).astype(np.float32)
    y = np.argmax(X, axis=1)
    m, rep = fit(X, y, task="softmax", labels=["a", "b", "c"], epochs=150, lr=0.05)
    assert rep.val_metric > 0.85 and m.predict(X[0]).shape == (3,)
    yr = 2 * X[:, 0] + 1
    m, rep = fit(X, yr, task="regression", epochs=150, lr=0.05)
    assert rep.val_metric > 0.95


def test_feature_mismatch_rejected(tmp_path):
    X = np.random.default_rng(0).standard_normal((50, 3))
    m, _ = fit(X, (X[:, 0] > 0).astype(float), spec=SPEC, epochs=5)
    path = m.save(tmp_path / "m.npz")
    other = FeatureSpec("toy", 2, ("a", "b", "c"))
    with pytest.raises(FeatureMismatch):
        load_model(path, other)
    model, msg = try_load(path, other)
    assert model is None and "rejected" in msg
    model, msg = try_load(tmp_path / "missing.npz", SPEC)
    assert model is None and "not found" in msg


def test_profile_inheritance_and_fallback(tmp_path):
    @register("ToyA")
    class ToyA(Component):
        phase = Phase.DECIDE
        writes = ("strategy",)

        def __init__(self, k=0):
            self.k = k

    @register("ToyB")
    class ToyB(ToyA):
        pass

    builtins = {
        "base": {"name": "base", "slots": {"strategy": {"impl": "ToyA", "k": 1}, "extra": {"impl": "ToyB"}}},
        "child": {"base": "base", "name": "child",
                  "slots": {"strategy": {"impl": "ToyB", "k": 2, "fallback": {"impl": "ToyA"}}, "extra": None}},
    }
    spec = resolve("child", builtins)
    comps = build(spec)
    assert len(comps) == 1
    c = comps[0]
    assert type(c).__name__ == "ToyB" and c.k == 2 and c.slot == "strategy"
    assert type(c.fallback).__name__ == "ToyA" and c.fallback.slot == "strategy"
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"base": "base", "config": {"time_budget_ms": 30}}), encoding="utf-8")
    spec = resolve(str(f), builtins)
    assert spec["config"]["time_budget_ms"] == 30 and len(build(spec)) == 2
    for k in ("ToyA", "ToyB"):
        REGISTRY.pop(k)
