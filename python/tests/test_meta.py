import os

from adjutant.components.meta import ANALYZERS, META_FEATURES, clean_map_name, matchup, register_analyzer
from adjutant.mapgraph import MapGraph
from blackboard.recorder import Recorder
from bwbot import Race, UnitType as U
from bwbot.observation import MapChoke

from fakes import FakeWorld, Sim, make_game

os.environ["BWBOT_LOG"] = "0"


def _bot(g):
    from adjutant import Adjutant
    bot = Adjutant("parity")
    bot.recorder = Recorder(enabled=False)
    bot.strict = True
    bot.game = g
    bot.on_start(g)
    return bot


def test_mapgraph_routes_through_chokes_and_falls_back():
    g = make_game()
    mg = MapGraph(g)
    main, nat = g.bases[0], g.bases[1]
    d = mg.base_distance(main.id, nat.id)
    c = g.chokes[0].center
    straight = ((main.center[0] - c[0]) ** 2 + (main.center[1] - c[1]) ** 2) ** 0.5 + \
               ((nat.center[0] - c[0]) ** 2 + (nat.center[1] - c[1]) ** 2) ** 0.5
    assert abs(d - straight) < 1e-6
    assert mg.connected(main.area_id, nat.area_id)
    assert not mg.connected(main.area_id, g.bases[2].area_id)       # fake map: starts not linked
    far = mg.base_distance(main.id, g.bases[2].id)
    assert far > 0


def test_clean_map_name():
    assert clean_map_name("\x06B\x05e\x06n\x05z\x06e\x05n\x06e\x051.1") == "Benzene1.1"
    assert clean_map_name("\x07Destination \x051.1") == "Destination 1.1"


def test_mapgraph_multi_hop():
    g = make_game()
    # link the two naturals directly
    g.chokes.append(MapChoke(9, 2, 4, (60 * 32, 60 * 32), 128, False))
    mg = MapGraph(g)
    assert mg.connected(1, 3)
    d = mg.base_distance(0, 2)
    assert d >= ((g.bases[0].center[0] - g.bases[2].center[0]) ** 2 +
                 (g.bases[0].center[1] - g.bases[2].center[1]) ** 2) ** 0.5 - 1e-6


def test_meta_fills_section_features_and_detects_random_race():
    g = make_game(enemy_race=Race.Random)
    calls = []

    @register_analyzer("probe")
    def probe(game, graph):
        calls.append(game.map_name)
        return {"ok": 1}

    try:
        bot = _bot(g)
        m = bot.bb.meta
        assert m.map_name == "Fake Map" and m.n_starts == 2 and m.n_bases == 5
        assert m.rush_distance > 0 and m.natural_distance > 0 and m.main_choke_width == 96
        assert list(m.features) == list(META_FEATURES)
        assert m.features["enemy_random"] == 1.0 and m.enemy_race == int(Race.Unknown)
        assert m.map_analysis["probe"] == {"ok": 1} and calls == ["Fake Map"]
        assert m.map_analysis["bases"]["by_distance"][0]["id"] == g.self_main_id
        w = FakeWorld(g)
        w.standard_start(4)
        ex, ey = g.players[1].start_location
        w.add(U.Zerg_Hatchery, ex * 32 + 64, ey * 32 + 48, player=g.enemy_id)
        Sim(w).run(bot, 16, skip=8)
        assert m.enemy_race == int(Race.Zerg) and m.features["enemy_Z"] == 1.0
        assert matchup(m.self_race, m.enemy_race) == "TvZ"
    finally:
        ANALYZERS.pop("probe", None)
