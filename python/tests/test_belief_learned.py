import os

from adjutant.components.belief_learned import LearnedBelief
from adjutant.learn import train as learn_train
from adjutant.learn.belief import BELIEF_SPEC, TECH, UNITS, join_opponents, template_at
from blackboard.recorder import Recorder, read_game
from bwbot import Race, UnitType as U

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"
HIDDEN = 0b10          # visible to the enemy only


def _truth_game(tmp, i, frames=24 * 120):
    from adjutant import Adjutant
    g = make_game(enemy_race=Race.Zerg)
    rec = Recorder(directory=tmp, enabled=True)
    os.environ["BWBOT_GAME_ID"] = f"truth{i}"
    try:
        bot = Adjutant("truth", slots={"strategy": {"impl": "Explore", "seed": i}})
        bot.recorder = rec
        bot.strict = True
        bot.game = g
        bot.on_start(g)
    finally:
        del os.environ["BWBOT_GAME_ID"]
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    ex, ey = g.players[1].start_location
    px, py = ex * 32 + 64, ey * 32 + 48
    w.add(U.Zerg_Hatchery, px, py, player=g.enemy_id)
    w.add(U.Zerg_Spawning_Pool, px + 120, py, player=g.enemy_id)
    w.add(U.Zerg_Spire, px - 120, py, player=g.enemy_id, visible_mask=HIDDEN)
    for k in range(8):
        w.add(U.Zerg_Zergling, px + k * 8, py + 150, player=g.enemy_id, visible_mask=HIDDEN)
    Sim(w).run(bot, frames)
    bot.on_end(i % 2 == 0)
    return bot, rec.path


def test_truth_rows_fog_and_widths(tmp_path):
    bot, path = _truth_game(tmp_path, 0)
    b, t = bot.bb.belief, bot.bb.truth
    assert t.enabled and t.counts[int(U.Zerg_Zergling)] == 8 and int(U.Zerg_Spire) in t.buildings
    assert b.count(U.Zerg_Zergling) == 0 and int(U.Zerg_Spire) not in b.tech     # fog filter held
    rows = [r for r in read_game(path)["rows"] if r["slot"] == "belief" and r["k"] == "sample"]
    assert len(rows) >= 5
    r = rows[-1]
    assert len(r["x"]) == BELIEF_SPEC.dim and len(r["yc"]) == len(UNITS) and len(r["yt"]) == len(TECH)
    assert r["yt"][TECH.index(int(U.Zerg_Spire))] == 1 and r["yb"] == 1


def test_train_belief_and_learned_predictions(tmp_path):
    logs = tmp_path / "logs"
    for i in range(6):
        _truth_game(logs, i)
    out = tmp_path / "models"
    assert learn_train.main(["belief", "--logs", str(logs), "--out", str(out), "--epochs", "300"]) == 0
    for h in ("counts", "tech", "bases"):
        assert (out / f"belief_{h}.npz").is_file()

    from adjutant import Adjutant
    g = make_game(enemy_race=Race.Zerg)
    bot = Adjutant("planned", slots={"belief": {"impl": "LearnedBelief", "models": str(out)}})
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    lb = next(c for c in bot.sched.components if isinstance(c, LearnedBelief))
    assert set(lb.heads) >= {"counts", "tech", "bases"}, lb.messages
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    ex, ey = g.players[1].start_location
    w.add(U.Zerg_Hatchery, ex * 32 + 64, ey * 32 + 48, player=g.enemy_id)
    w.add(U.Zerg_Spawning_Pool, ex * 32 + 184, ey * 32 + 48, player=g.enemy_id)
    Sim(w).run(bot, 24 * 60)
    b = bot.bb.belief
    assert b.predicted["counts"][int(U.Zerg_Zergling)] > 3
    assert b.count(U.Zerg_Zergling) > 3                     # raised above what was seen (0)
    assert int(U.Zerg_Spire) in b.tech and int(U.Zerg_Spire) in b.inferred
    assert all(s.failures == 0 for s in bot.sched.stats.values())


def test_learned_belief_without_models_is_scripted(tmp_path):
    from adjutant import Adjutant
    g = make_game()
    bot = Adjutant("planned", slots={"belief": {"impl": "LearnedBelief", "models": str(tmp_path)}})
    bot.recorder = Recorder(enabled=False)
    bot.game = g
    bot.on_start(g)
    lb = next(c for c in bot.sched.components if isinstance(c, LearnedBelief))
    w = FakeWorld(g)
    w.standard_start(4)
    Sim(w).run(bot, 24 * 5)
    assert lb.heads == {} and bot.bb.belief.predicted == {}


def test_label_join_other_side_template():
    def game(side, switches):
        return {"header": {"game_id": "g1", "side": side}, "path": f"g1-{side}.jsonl", "end": {"won": side == "a"},
                "rows": [{"slot": "strategy", "k": "switch", "f": f, "to": t} for f, t in switches]}
    a = game("a", [(0, "bio_2rax")])
    b = game("b", [(0, "mech_expand"), (5000, "goliath_1fact")])
    lone = {"header": {"game_id": "g2", "side": "a"}, "path": "g2-a.jsonl", "rows": [], "end": {"won": True}}
    opp = join_opponents([a, b, lone])
    assert "g2-a.jsonl" not in opp
    assert template_at(opp["g1-a.jsonl"], 100) == "mech_expand"
    assert template_at(opp["g1-a.jsonl"], 6000) == "goliath_1fact"
    assert template_at(opp["g1-b.jsonl"], 10) == "bio_2rax"
