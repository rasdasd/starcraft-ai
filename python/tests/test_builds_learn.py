import json
import random

from adjutant.learn import builds as B
from adjutant.strategies import TEMPLATES, BuildSpec


def test_mutations_of_every_builtin_are_valid_and_new():
    rng = random.Random(1)
    for name, t in sorted(TEMPLATES.items()):
        if not isinstance(t, BuildSpec):
            continue
        for _ in range(10):
            d = B.mutate(t.to_doc(), rng)
            assert d is not None, name
            spec = BuildSpec(d)
            assert spec.parent == name and spec.name.startswith(name.split("~")[0] + "~")
            assert B._digest(d) != B._digest(t.to_doc())
            sup = [s[0] for s in d.get("opening", [])]
            assert sup == sorted(sup)


def test_stats_and_prune_only_drop_losing_generated_builds(tmp_path):
    rng = random.Random(2)
    parent = TEMPLATES["mech_expand"].to_doc()
    kids = [B.mutate(parent, rng) for _ in range(2)]
    bdir = tmp_path / "builds"
    bdir.mkdir()
    for d in kids:
        (bdir / f"{d['name'].replace('~', '__')}.json").write_text(json.dumps(d))
    (bdir / "mine.json").write_text(json.dumps({**parent, "name": "mine"}))
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    rows = []
    for i in range(8):
        for who, won in ((kids[0]["name"], False), (kids[1]["name"], i % 2 == 0), ("mine", False)):
            log = run / "logs" / f"{who.replace('~', '_')}-{i}.jsonl"
            log.write_text(json.dumps({"t": "rec", "slot": "report", "k": "snap", "f": 100, "tmpl": who}) + "\n")
            rows.append({"a": {"result": {"log": str(log)}}, "b": {"name": "Opp"}, "winner": "a" if won else "b"})
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    st = B.stats([run])
    assert st[kids[0]["name"]]["games"] == 8 and st[kids[1]["name"]]["wins"] == 4
    B.main(["prune", str(run), "--dir", str(bdir), "--min-games", "6", "--below", "0.3"])
    left = {p.name for p in bdir.glob("*.json")}
    assert f"{kids[0]['name'].replace('~', '__')}.json" not in left
    assert f"{kids[1]['name'].replace('~', '__')}.json" in left and "mine.json" in left
