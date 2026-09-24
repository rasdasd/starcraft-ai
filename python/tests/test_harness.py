import argparse
import io
import random

from adjutant.learn.report import print_report, summarize, wilson
from harness.selfplay import Player, judge, schedule


def test_player_parse():
    p = Player.parse("adjutant@explore/Terran")
    assert (p.spec, p.profile, p.race, p.name) == ("adjutant", "explore", "Terran", "adjutant@explore")
    z = Player.parse("sparring.zerg:Pool/Zerg")
    assert (z.spec, z.profile, z.race) == ("sparring.zerg:Pool", None, "Zerg")
    assert Player.parse("sparring.protoss:Dragoon").race == "Protoss"
    assert Player.parse("goliath").race == "Terran"


def test_judge():
    win = {"won": True, "frame": 9000, "total": 100}
    loss = {"won": False, "frame": 9000, "total": 900}
    assert judge(win, loss, 28800) == ("a", "elimination")
    assert judge(None, loss, 28800) == ("b", "crash_a")
    assert judge(None, None, 28800) == (None, "crash")
    at_limit_a = {"won": False, "frame": 28800, "total": 500}
    at_limit_b = {"won": False, "frame": 28800, "total": 700}
    assert judge(at_limit_a, at_limit_b, 28800) == ("b", "score")
    assert judge(at_limit_a, dict(at_limit_b, total=500), 28800) == (None, "draw")


def test_schedule_alternates_maps_and_samples_pool():
    args = argparse.Namespace(pool=None, p1=["adjutant"], p2=["goliath", "mybot"], games=6,
                              maps=["m1", "m2"], swap=False)
    games = schedule(args, random.Random(1))
    assert [g.map for g in games] == ["m1", "m2"] * 3
    assert all(g.a.spec == "adjutant" and g.b.spec in ("goliath", "mybot") for g in games)


def _row(winner, a="adj", b="gol", map_="maps/(2)X.scx", template="goliath_1fact", reason="elimination"):
    return {"a": {"name": a, "template": template}, "b": {"name": b}, "map": map_, "winner": winner,
            "reason": reason}


def test_report_tallies():
    rows = [_row("a"), _row("a"), _row("b"), _row(None, reason="draw"), _row(None, reason="crash")]
    s = summarize(rows)
    t = s["player"]["adj"]
    assert (t.wins, t.losses, t.draws) == (2, 1, 1)
    assert abs(t.rate - 0.625) < 1e-9
    assert s["matchup"]["gol vs adj"].losses == 2
    assert s["template"]["adj / goliath_1fact"].n == 4
    assert s["reasons"]["crash"] == 1
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    buf = io.StringIO()
    print_report(rows, out=buf)
    assert "Opening template" in buf.getvalue()
