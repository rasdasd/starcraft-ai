import os

import numpy as np

from adjutant.components.strategy import LearnedStrategy, RuleSelector, load_strategy_model
from adjutant.learn import train as learn_train
from adjutant.learn.features import CTX_NAMES, CTX_SPEC, strategy_context
from blackboard.recorder import Recorder, read_game
from bwbot import Race

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _bot(profile="scripted", strategy=None, recorder=None, enemy=Race.Zerg):
    from adjutant import Adjutant
    slots = {"strategy": strategy} if strategy else {}
    bot = Adjutant(profile, slots=slots)
    bot.recorder = recorder or Recorder(enabled=False)
    bot.strict = True
    g = make_game(enemy_race=enemy)
    bot.game = g
    bot.on_start(g)
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    return bot, w


def _strategy(bot):
    return next(c for c in bot.sched.components if c.slot == "strategy")


def test_context_has_spec_width():
    bot, w = _bot()
    Sim(w).run(bot, 24 * 5)
    x = strategy_context(bot.bb)
    assert x.shape == (CTX_SPEC.dim,) and len(CTX_NAMES) == CTX_SPEC.dim
    assert np.isfinite(x).all()


def test_rule_selector_defaults_and_rules():
    bot, w = _bot(strategy={"impl": "RuleSelector"})
    Sim(w).run(bot, 24 * 5)
    rs = _strategy(bot)
    assert isinstance(rs, RuleSelector)
    assert bot.bb.strategy.template == "bio_2rax"             # vs Zerg default
    bot.bb.belief.opening = "rush"
    assert rs.rule(bot.bb) == "anti_rush"
    bot.bb.belief.opening = "unknown"
    bot.bb.belief.air = 5
    assert rs.rule(bot.bb) == "goliath_1fact"


def test_explore_records_ctx_rows(tmp_path):
    rec = Recorder(directory=tmp_path, enabled=True)
    bot, w = _bot(strategy={"impl": "Explore", "seed": 3}, recorder=rec)
    Sim(w).run(bot, 24 * 70)
    bot.on_end(True)
    g = read_game(rec.path)
    rows = [r for r in g["rows"] if r["slot"] == "strategy" and r["k"] == "ctx"]
    assert len(rows) >= 3                   # switch + periodic rows
    assert all(len(r["x"]) == CTX_SPEC.dim for r in rows)
    assert rows[0]["reason"] == "switch" and rows[0]["template"] in ("goliath_1fact", "bio_2rax", "anti_rush",
                                                                       "mech_expand")
    assert g["header"]["meta"]


def _synthetic_logs(d, n=80, seed=0):
    """bio_2rax wins vs Zerg (enemy_Z=1), mech_expand wins otherwise; the others lose half the time."""
    rng = np.random.default_rng(seed)
    bot, w = _bot()
    Sim(w).run(bot, 24 * 5)
    base = strategy_context(bot.bb)
    iz = CTX_NAMES.index("enemy_Z")
    templates = ["bio_2rax", "mech_expand", "goliath_1fact"]
    for i in range(n):
        rec = Recorder(directory=d, enabled=True)
        os.environ["BWBOT_GAME_ID"] = f"syn{i}"
        try:
            rec.start({"bot": "adjutant:explore"})
        finally:
            del os.environ["BWBOT_GAME_ID"]
        zerg = bool(rng.random() < 0.5)
        t = templates[i % 3]
        for f in range(0, 24 * 60 * 6, 24 * 30):
            x = base + rng.normal(0, 0.05, CTX_SPEC.dim).astype(np.float32)
            x[iz] = float(zerg)
            rec.record("strategy", "ctx", f, x=x.tolist(), template=t)
        if t == "goliath_1fact":
            won = bool(rng.random() < 0.5)
        else:
            won = (t == "bio_2rax") == zerg
        rec.finish(won, 24 * 60 * 6)


def test_train_and_learned_select_and_blend(tmp_path):
    logs = tmp_path / "logs"
    _synthetic_logs(logs)
    out = tmp_path / "strategy.npz"
    assert learn_train.main(["strategy", "--logs", str(logs), "--out", str(out), "--epochs", "400"]) == 0
    model, templates, msg = load_strategy_model(str(out))
    assert model is not None, msg
    assert templates == ["bio_2rax", "goliath_1fact", "mech_expand"]

    bot, w = _bot(strategy={"impl": "LearnedStrategy", "model": str(out), "mode": "select"}, enemy=Race.Zerg)
    Sim(w).run(bot, 24 * 5)
    ls = _strategy(bot)
    assert isinstance(ls, LearnedStrategy) and ls.model is not None
    assert bot.bb.strategy.template == "bio_2rax"
    # anti_rush never appears in the logs but is scored by its descriptor
    assert set(bot.bb.strategy.values) == {"anti_rush", "bio_2rax", "goliath_1fact", "mech_expand"}

    bot, w = _bot(strategy={"impl": "LearnedStrategy", "model": str(out), "mode": "blend"}, enemy=Race.Protoss)
    Sim(w).run(bot, 24 * 5)
    assert bot.bb.strategy.template == "mech_expand"
    probs = bot.bb.strategy.values
    assert probs["mech_expand"] > probs["bio_2rax"]


def test_learned_without_model_uses_rules(tmp_path):
    bot, w = _bot(strategy={"impl": "LearnedStrategy", "model": str(tmp_path / "missing.npz")}, enemy=Race.Protoss)
    Sim(w).run(bot, 24 * 5)
    assert _strategy(bot).model is None
    assert bot.bb.strategy.template == "mech_expand"
    assert all(s.failures == 0 for s in bot.sched.stats.values())


def test_learned_hysteresis_keeps_current_within_margin(tmp_path):
    logs = tmp_path / "logs"
    _synthetic_logs(logs, n=60, seed=1)
    out = tmp_path / "strategy.npz"
    learn_train.main(["strategy", "--logs", str(logs), "--out", str(out)])
    bot, w = _bot(strategy={"impl": "LearnedStrategy", "model": str(out), "margin": 2.0}, enemy=Race.Zerg)
    ls = _strategy(bot)
    Sim(w).run(bot, 24 * 5)
    first = bot.bb.strategy.template
    bot.bb.meta.features["enemy_Z"] = 0.0          # context now favours another template
    ls._last = -10 ** 9
    Sim(w).run(bot, 24 * 60)
    assert bot.bb.strategy.template == first          # margin 2.0: never worth switching
