import json
import os

import pytest

from adjutant.strategies import TEMPLATES, BuildSpec, SpecError, load_dirs, reload, validate_goal, validate_template
from adjutant.strategies.spec import Expr
from adjutant.techtree import TechTree
from blackboard.recorder import Recorder
from bwbot import Race, UnitType as U

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def test_builtin_builds_cover_all_races_and_validate():
    tree = TechTree(make_game())
    races = {t.race for t in TEMPLATES.values()}
    assert races >= {int(Race.Terran), int(Race.Protoss), int(Race.Zerg)}
    for name, t in TEMPLATES.items():
        assert validate_template(t, tree) == [], name


def test_expressions_reject_unsafe_code():
    for src in ("__import__('os')", "x.y", "a[0]", "(lambda: 1)()", "open('f')"):
        with pytest.raises(SpecError):
            doc = {"name": "bad", "race": "Terran", "goal": {"workers": src}}
            BuildSpec(doc)
    assert Expr("min(3, 1 + 1) if 2 > 1 else 0")({"min": min}) == 2


def test_spec_errors_are_reported():
    with pytest.raises(SpecError, match="unknown keys"):
        BuildSpec({"name": "x", "race": "Terran", "goall": {}})
    with pytest.raises(SpecError, match="UnitType"):
        BuildSpec({"name": "x", "race": "Terran", "goal": {"units": {"Zergling": 4}}})
    with pytest.raises(SpecError, match="race"):
        BuildSpec({"name": "x", "race": "Elf"})


def _bot(template, race=Race.Terran, enemy=Race.Zerg):
    from adjutant import Adjutant
    bot = Adjutant("scripted", slots={"strategy": {"impl": "ScriptedStrategy", "template": template}})
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    g = make_game(self_race=race, enemy_race=enemy)
    bot.game = g
    bot.on_start(g)
    w = FakeWorld(g, minerals=50)
    w.standard_start(6)
    return bot, w, g


def test_user_build_dir_phases_and_hold(tmp_path, monkeypatch):
    doc = {
        "name": "my_vultures", "race": "Terran", "tags": ["harass"], "default_vs": ["Zerg"],
        "opening": [[9, "Supply_Depot"], [11, "Barracks"], [12, "Refinery"], [14, "Factory"]],
        "attack_supply": 6, "hold_if": "minute < 3",
        "goal": {"workers": 20, "buildings": {"Factory": 2}, "units": {"Vulture": 12},
                 "upgrades": ["Ion_Thrusters"]},
        "phases": [{"when": "minute >= 2", "units": {"Vulture": 4, "Goliath": "2 + enemy_air"}, "bases": 2}],
    }
    (tmp_path / "mine.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("BWBOT_BUILDS", str(tmp_path))
    try:
        reload()
        assert "my_vultures" in TEMPLATES and "mech_expand" in TEMPLATES
        bot, w, g = _bot("my_vultures")
        sim = Sim(w)
        sim.run(bot, 24 * 30, skip=8)
        goal = bot.bb.strategy.goal
        assert goal.units == {int(U.Terran_Vulture): 12} and goal.bases == 1 and goal.workers == 20
        sim.run(bot, 24 * 100, skip=8)
        goal = bot.bb.strategy.goal
        assert goal.units[int(U.Terran_Vulture)] == 4 and goal.units[int(U.Terran_Goliath)] == 2 and goal.bases == 2
        assert validate_goal(goal, TechTree(g), int(Race.Terran)) == []
        bot.bb.world.army_supply = 10
        from blackboard.sections import Posture
        bot.bb.frame = 24 * 60
        assert TEMPLATES["my_vultures"].posture(bot.bb, Posture(stance="attack")).stance == "hold"
    finally:
        monkeypatch.delenv("BWBOT_BUILDS")
        reload()
    assert "my_vultures" not in TEMPLATES


def test_scripted_strategy_falls_back_for_other_races():
    bot, w, g = _bot("goliath_1fact", race=Race.Protoss, enemy=Race.Terran)
    Sim(w).run(bot, 24 * 5)
    assert TEMPLATES[bot.bb.strategy.template].race == int(Race.Protoss)
    assert bot.bb.strategy.template == "dragoon_2gate"


@pytest.mark.parametrize("name,race", [("dragoon_2gate", Race.Protoss), ("zealot_2gate", Race.Protoss),
                                       ("hydra_3hatch", Race.Zerg), ("ling_9pool", Race.Zerg)])
def test_other_race_goals_are_valid(name, race):
    bot, w, g = _bot(name, race=race, enemy=Race.Terran)
    tree = TechTree(g)
    t = TEMPLATES[name]
    for minute in (1, 4, 8):
        bot.bb.frame = bot.bb.world.frame = 24 * 60 * minute
        assert validate_goal(t.goal(bot.bb), tree, int(race)) == [], (name, minute)


def test_load_dirs_keeps_later_dirs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for d, n in ((a, 10), (b, 20)):
        (d / "x.json").write_text(json.dumps({"name": "x", "race": "Zerg", "goal": {"workers": n}}), encoding="utf-8")
    assert load_dirs([a, b])["x"].base["workers"] == 20
