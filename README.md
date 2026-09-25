# starcraft-ai

A StarCraft: Brood War bot environment where the bot "brain" is a normal 64-bit Python process and
the only game-coupled code is a small C++ shim. Runs against real StarCraft 1.16.1 + BWAPI 4.4.0 on
Windows (for watching / competitions) and against headless OpenBW in WSL2 (for fast, unattended
games). Both use the same shim source, the same wire protocol, and the same Python bot.

```
Windows                                             WSL2 Ubuntu
+----------------------------+                       +------------------------------------+
| StarCraft.exe 1.16.1 (x86) |                       | BWAPILauncher (OpenBW, headless)   |
|   + BWAPI.dll              |                       |   + shim_module.so (AI module)     |
+-------------+--------------+                       +-----------------+------------------+
              | shared memory + pipe (BWAPI client mode)                | TCP :8765 (WSL -> Windows localhost)
+-------------v--------------+                                         |
| shim.exe (x86, C++)        |                                         |
+-------------+--------------+                                         |
              | TCP 127.0.0.1:8765, length-prefixed FlatBuffers        |
+-------------v---------------------------------------------------------v------------------+
| python -m bwbot.run examples.basic_terran     (64-bit Python 3.12, numpy; torch optional) |
+-------------------------------------------------------------------------------------------+
```

Key decisions:

- The shim is the only 32-bit / game-coupled piece. On Windows it runs as a separate **BWAPI client
  process** (`shim.exe`), so a bot crash never kills StarCraft. On OpenBW (whose BWAPI fork has no
  client mode) the same code is loaded as an **AI module** (`shim_module.so`).
- The shim is the TCP server; Python connects. This direction works across the WSL2 boundary.
- Lockstep: per decision frame the shim sends `Frame`, waits for `Commands`, applies them, advances.
  `frame_skip` lets the bot act every N game frames. Static data (map grids, unit type table,
  players) is sent once per match in `GameStart`.
- FlatBuffers for serialization: `UnitState` is a fixed-layout struct vector that Python views as a
  numpy structured array with **zero copies**. Typical frame ≈ 25-60 KB; serialize ≈ 0.25 ms,
  round trip ≈ 0.3 ms, ~850 decisions/s end to end on this machine.
- BWAPI 4.4.0 (not 5.x) because OpenBW's fork and all existing client bindings target 4.x.

## Layout

```
game/                 (git-ignored) StarCraft 1.16.1 + BWAPI 4.4.0 runtime, maps/BroodWar/{sscai,aiide,cog}/, replays
proto/bw.fbs          wire protocol (FlatBuffers schema) - the single source of truth
shim/                 C++ shim: CMakeLists.txt, src/, third_party/bwapi + bwem-community
  build/              Win32 build: shim.exe, shim_module.dll
  build-openbw/       Linux build (from WSL): shim_module.so
python/               bwbot package (framework) + mybot/ + goliath/ + learned/ + examples/, .venv/
  blackboard/         blackboard framework: sections, scheduler, arbiters, recorder, numpy models
  adjutant/           blackboard bot: components/, strategies/, learn/ (per-slot features + training)
  harness/            self-play (OpenBW LAN) and published-bot match runners
  tests/              unit tests (synthetic games; no StarCraft needed)
scripts/              setup_windows.ps1, build_shim.ps1, run_native.ps1, gen_proto.ps1|sh, gen_enums.py,
                      setup_wsl.sh, run_openbw.sh, freeze_bot.ps1, pack_competition.ps1
docs/competition.md   AIIDE/BASIL run-folder layout and organizer notes
wsl/                  (git-ignored) openbw/ + bwapi/ checkouts and build, game/ dir for BWAPILauncher
tools/                (git-ignored) flatc
```

## Setup

### Windows (native StarCraft)

Prereqs: Windows 10/11, Python 3.12 x64 on PATH, winget. Visual Studio 2022 (Community or Build
Tools) with the *Desktop development with C++* workload; `setup_windows.ps1` installs Build Tools
via winget if no MSVC is found.

```powershell
scripts\setup_windows.ps1    # MSVC check, flatc, BWAPI 4.4.0 sources, game download (~100 MB), bwapi.ini, venv, codegen
scripts\build_shim.ps1       # -> shim\build\shim.exe (Win32) and shim_module.dll
scripts\run_native.ps1       # shim + StarCraft(BWAPI injected) + example bot, each in its own window
scripts\run_native.ps1 -Stop # kill everything
```

Day-to-day, use the root launcher instead; it works from a Windows shell *and* from WSL (via
`powershell.exe` interop) and drives `run_native.ps1`:

```
python run.py                                   # example bot in a visible StarCraft window
python run.py --bot mybots.zerg --speed 42 --map "maps/BroodWar/sscai/(4)Python.scx"
python run.py --bot mybot --games 5              # exactly 5 games, then StarCraft/shim/bot shut down
python run.py --stop
python run.py --help                            # all options
```

The game archive is David Churchill's `scbw_bwapi440.zip` (StarCraft 1.16.1 redistributed with
Blizzard's permission for AI research; includes BWAPI 4.4.0, Injectory launcher, and the SSCAI,
AIIDE, and COG map packs).

`game/bwapi-data/bwapi.ini` is configured for unattended single-player games: `auto_menu =
SINGLE_PLAYER`, Terran vs Random on `(2)Destination.scx`, `auto_restart = ON`, windowed, sound off,
replays into `game/maps/replays/bwbot/`. `run_native.ps1 -Map/-Race/-EnemyRace` override it.
Single-player StarCraft pauses when it loses focus; the shim detects that and resumes the game.

### WSL2 (OpenBW, headless)

Prereqs: WSL2 with Ubuntu (22.04/24.04), and the Windows setup done first (OpenBW reads the
`.mpq` files from `game/`).

```powershell
wsl -d Ubuntu -u root -- bash scripts/setup_wsl.sh   # apt deps, clone+build openbw/bwapi, build shim_module.so, set up wsl/game
wsl -d Ubuntu -- bash scripts/run_openbw.sh          # headless BWAPILauncher + shim, listening on :8765
python\.venv\Scripts\python -m bwbot.run examples.basic_terran --no-gui --max-frames 20000
```

`run_openbw.sh --map ... --race ... --enemy-race ... --ui --port ...`. Set `OPENBW_ENABLE_UI=1`
when running `setup_wsl.sh` to also build the SDL2 viewer (then `--ui` shows the game). Python can
run from Windows (WSL2 forwards `localhost`) or inside WSL (`pip install -e python` there).

OpenBW has no built-in computer opponent: the enemy in single-player just sits idle. Use
`--max-frames` to end games, or run two shim instances for self-play (see `OPENBW_LAN_MODE` in the
[OpenBW BWAPI README](https://github.com/OpenBW/bwapi)). Measured here: ~1800 game frames/s at
`frame_skip=2` (≈75x realtime), bounded by Python round trips.

## Writing a bot

Start from `python/mybot/` — a runnable starter bot (`python run.py --bot mybot`) with a swappable
`Policy` on top of UAlbertaBot-shaped managers. `python run.py --bot goliath` is the same stack
with a mech opener.

| file | role | ML analogue |
|---|---|---|
| `mybot/state.py` | `perceive(obs, mem) -> State`: counts, supply, army/worker/enemy arrays, fog buildings, BWEM natural / main choke; `State.as_features()` | feature extraction |
| `mybot/policy.py` | `Policy.decide(State) -> [Train, Build, Attack, Rally, …]`; `ScriptedPolicy` walks `opening.MARINE` | the model |
| `mybot/opening.py` | supply-gated build lists (`MARINE`, `GOLIATH`) shared by policies | - |
| `mybot/information.py` | fog memory for enemy buildings; guessed / seen enemy start | - |
| `mybot/opponent.py` | race, first-seen timings, opening guess, proxy flag | opponent features |
| `mybot/learned.py` | teacher opening + opponent prior, or `LinearPolicy` from `models/policy.npz` | the model |
| `mybot/logger.py` / `train.py` | JSONL decisions; `python -m mybot.train` fits a numpy softmax | dataset / train |
| `mybot/production.py` | queue + train / addon / upgrade (one building at a time until started) | - |
| `mybot/buildings.py` | construction state machine (reserve, assign SCV, re-issue, retry tile) | - |
| `mybot/workers.py` | mineral / gas / build / repair / scout jobs | - |
| `mybot/scout.py` | one SCV to the other start, then watch | - |
| `mybot/combat.py` | defend / push / hunt-air; no leave-home under 3 army | - |
| `mybot/macro.py` | reserved-tile placer (mineral-line exclusion, geyser refineries, BWAPI legality) | - |
| `mybot/bot.py` | `MyBot`: ticks managers each decision and draws a HUD | environment glue |

To change the opener edit `opening.MARINE` / `opening.GOLIATH`. For the learning loop:

```
python run.py --bot learned              # Goliath teacher + opponent prior; logs to logs/
python -m mybot.train --logs logs --out models/policy.npz
python run.py --bot learned              # loads models/policy.npz (or BWBOT_MODEL / bwapi-data/read/policy.npz)
```

The model only outputs a tactic (`hold` / `defend` / `push` / `hunt_air`) and the next building.
Managers still place tiles, assign workers, and the runner still trims to `apm_budget`.

The minimal version of the same thing, without the layers:

```python
from bwbot import Bot, ClientConfig, UnitType, UnitFlag, run

class MyBot(Bot):
    config = ClientConfig(frame_skip=2, local_speed=0)   # act every 2 frames, run the game at max speed

    def on_start(self, game):          # game: GameInfo (map grids, players, unit type table)
        print(game.map_name, game.self_race, game.start_locations)

    def on_frame(self, obs, act):      # obs: Observation, act: Actions
        for scv in obs.idle(obs.my_completed(UnitType.Terran_SCV)):
            patch = obs.nearest(obs.minerals_fields, scv["x"], scv["y"])
            if patch is not None:
                act.gather(scv, patch)
        for cc in obs.my_completed(UnitType.Terran_Command_Center):
            if cc["train_queue_count"] == 0 and obs.minerals >= 50:
                act.train(cc, UnitType.Terran_SCV)
        act.draw_text_screen(10, 10, f"frame {obs.frame_count} minerals {obs.minerals}")

    def on_end(self, is_winner): ...

run(MyBot())                           # or: python -m bwbot.run mymodule:MyBot
```

- `obs.units` is a numpy structured array (`bwbot.UNIT_DTYPE`: id, type, player, x, y, hit_points,
  shields, energy, flags, order, target, cooldowns, timers, ...). It views the receive buffer,
  so copy it if you keep it past the frame. Filters: `obs.my_units`, `obs.enemy_units`,
  `obs.my_units_of_type(t)`, `obs.idle(units)`, `obs.unit(id)`, `obs.tiles / visible / explored`.
- `obs.me` (`PlayerState`): minerals, gas, supply, per-type unit counts, upgrade levels, research.
- `obs.events`: BWAPI events since the last decision (`UnitCreate`, `UnitDestroy`, `ReceiveText`, ...).
- `bot.apm` (`ApmMeter`): actions per *game* minute that the bot issued - `current` (trailing minute),
  `average` (whole game), `total`, `last` (this decision). Unit commands count; game commands and
  drawing don't. The runner draws it on screen (`--no-apm-hud` to hide) and logs it; `obs.game_apm`
  is the game's own counter for cross-checking. `Bot.apm_budget` (default 400) caps trailing-minute
  unit commands; the runner trims `act.unit_cmds` after `on_frame` (`--apm-budget N`, `0` = unlimited).
- `act.*` mirrors `BWAPI::UnitCommandType` (`move`, `attack`, `build`, `train`, `research`, `use_tech_pos`,
  ...) plus game commands (`set_local_speed`, `send_text`, `leave_game`, ...) and debug drawing.
- `bwbot.enums` mirrors BWAPI's numeric enums (`UnitType`, `Order`, `TechType`, `UpgradeType`,
  `WeaponType`, `Race`, ...), generated from the vendored headers by `scripts/gen_enums.py`.
- `python -m bwbot.run <module[:Class]> [--frame-skip N] [--speed MS] [--no-gui] [--games N]
  [--max-frames N] [--apm-budget N] [--cheat-map-info] [--host H --port P]`.

ML frameworks are optional extras and never imported by the core:
`pip install -e "python[torch]"`, `"python[onnx]"`, `"python[tensorrt]"`.

## Adjutant (blackboard bot)

`python/blackboard/` is a bot-agnostic framework: typed board sections, components that each fill
one *slot*, a phased scheduler (SENSE → DECIDE → PLAN → ACT → REPORT) with per-component periods,
event triggers, a 40 ms decision budget and fallback to a scripted teacher when a component keeps
failing, plus arbiters for unit leases, money and command priority (commands are sorted by priority
before the runner's APM trim). `python/adjutant/` is the Terran bot built on it.

```
python run.py --bot adjutant                 # default profile
python run.py --bot adjutant:Parity          # the goliath bot, rebuilt from adapters over the mybot managers
BWBOT_PROFILE_FILE=my.json python run.py --bot adjutant
```

A profile says which implementation fills each slot. Override any part with JSON, for example:

```json
{"base": "adjutant", "slots": {"strategy": {"impl": "ScriptedStrategy", "template": "bio_2rax"}}}
```

The HUD shows one line per section and the per-slot timings. Each game writes a slot-tagged JSONL
log (`logs/adjutant/`, or `bwapi-data/write/adjutant-logs/` in tournaments) that the trainers read.
Models are numpy `.npz` files (no torch at inference), looked up in `bwapi-data/read/`, the
competition pack's `AI/models/`, then `python/models/`. Tests: `cd python; .venv\Scripts\python -m pytest`.

Many games unattended: `harness.selfplay` (parallel headless OpenBW games in WSL, Python bots
including Zerg/Protoss sparring bots), `harness.winematch` (published bots such as Locutus,
Stardust or Pluto on real StarCraft under Wine in WSL, several games at once) and
`harness.botmatch` (the same on native Windows, one slow game at a time). See
[docs/harness.md](docs/harness.md).

### Builds

A build is a JSON file: race, tags, the enemy races it is the default against, an opening of
`[supply, type]` steps, and a goal (workers, bases, buildings, addons, units, upgrades, techs)
whose numbers can be expressions over the board, e.g. `"Factory": "min(6, 2 + 2 * max(0, bases - 1))"`
or `"Goliath": "6 + 2 * enemy_air"`. Optional `phases` override parts of the goal once their `when`
expression holds; `attack_if` / `retreat_if` / `hold_if` set the army posture. The format is in
`python/adjutant/strategies/spec.py` and the built-ins (Terran, Protoss, Zerg) are in
`python/adjutant/builds/`.

Put your own builds in `builds/` at the repo root (or any folder in `BWBOT_BUILDS`); a file with
the same name as a built-in replaces it. They are loaded at start, checked (a broken file is
reported and skipped), and competed for by every selector: `RuleSelector` by tags (`rush_safe`,
`anti_air`) and `default_vs`, `Explore` at random, `LearnedStrategy` by predicted win rate. The
strategy model describes builds by their tags and opening shape instead of by name, so it also
scores builds it has never seen; no retraining is needed to add one. To always play one build, set
`BWBOT_PROFILE_JSON='{"slots": {"strategy": {"impl": "ScriptedStrategy", "template": "my_build"}}}'`
(or a path to such a file) with a profile whose production slot follows the goal (`planned`,
`search`); `parity` plays the goliath bot's own build.

Generated builds: `python -m adjutant.learn.builds mutate mech_expand --n 4` writes variants
(opening timings and order, an extra production building, attack/retreat supply, goal counts) to
`python/models/builds/` with a `parent` field. Play them with `explore` (it tries every build of
our race), then `... builds stats runs/X runs/Y` for the win rate per build and `... builds prune
runs/X --below 0.25 --min-games 6` to delete the generated ones that lose (hand-written builds are
never touched), and retrain the strategy model on the same runs.

### Profiles and training

`adjutant` (the default) is `parity` when playing Terran, `planned` as Protoss or Zerg (the goliath
policy is Terran-only; see `race_profiles` in `adjutant.profiles`). `parity` is the goliath bot's
managers behind the blackboard. `planned` swaps
in the new components (scripted belief, scouting, engagement evaluator, tactics + micro, crisis
defense, greedy tech-tree planner); `search` replaces the planner with the build-order search over
the economy simulator. The learned profiles build on `planned`, and each learned slot falls back to
its scripted behaviour when its model is missing.

Train one component at a time: collect games with its data profile, train, then evaluate the
learned profile against the scripted one (commands run from `python/`, in WSL):

| Component | Collect with | Train | Play with |
|---|---|---|---|
| Strategy win model | `explore` | `python -m adjutant.learn.train strategy --logs runs/X --out models/strategy.npz` | `learned`, `blend` |
| Belief | `truth` (full map info, fog-filtered) | `... train belief --logs runs/X --out models` | `learned` |
| Engagement predictor | any `planned`-based profile | `... train engage --logs runs/X --out models/engage.npz` | `learned_combat` |
| Tactics value model | `explore_combat` | `... train tactics --logs runs/X --out models/tactics.npz` | `learned_combat` |
| RL micro (experimental) | `explore_micro` | `... train micro --logs runs/X --out models/micro.npz` | `rl_micro` |

```
python -m harness.selfplay --p1 adjutant@explore_combat --p2 adjutant --games 48 --parallel 6 --run-id ec1
python -m adjutant.learn.train tactics --logs ../runs/ec1 --out models/tactics.npz
python -m harness.selfplay --p1 adjutant@learned_combat --p2 adjutant --games 24 --parallel 6 --run-id lc1
```

RL micro runs batch reinforcement learning. `RLMicro` picks one of five fight actions per unit
(`focus`, `nearest`, `kite`, `back`, `stay`) and logs transitions whose reward is the local
hit-point trade weighted by unit value. The trainer runs fitted Q iteration over those transitions.
Repeat collect and train: `explore_micro` loads the current `micro.npz`, so each round explores
around the latest policy. Learned micro only overrides the scripted action when its Q value is
higher by `margin`.

## Protocol

Defined in `proto/bw.fbs`; regenerate with `scripts/gen_proto.ps1` (Windows) or `scripts/gen_proto.sh`
(WSL) after edits. Every message is `uint32 LE length` + a FlatBuffer with root `Envelope { msg:
Message }`, file identifier `BWB1`.

```
shim -> bot   Hello         protocol_version, shim_version, backend ("bwapi-4.4.0" | "openbw")
bot  -> shim  ClientConfig  frame_skip, include_bullets/tiles/players, local_speed, gui, complete_map_information, user_input
shim -> bot   GameStart     map name/size/hash, ground_height, buildable, walkable (walk tiles), region_id,
                            start_locations, players, self/enemy/neutral ids, unit/weapon/upgrade/tech type tables,
                            BWEM areas/bases/chokes, start_bases, self_main_id, self_natural_id
loop:
shim -> bot   Frame         frame_count, players (resources, supply, counts), units:[UnitState], bullets, events,
                            tiles (visible/explored/creep bits), nuke_dots, serialize_us, last_roundtrip_us
bot  -> shim  Commands      unit_commands (unit, UnitCommandType, target, x, y, extra, queued),
                            game_commands (SetLocalSpeed, SendText, LeaveGame, ...), draws
shim -> bot   GameEnd       is_winner, frame_count
```

A `ClientConfig` may also be sent mid-game in place of `Commands` to change cadence/speed; the
shim applies it and keeps waiting for the `Commands` of the current frame.

All ids are BWAPI's: unit ids are `Unit::getID()`, type/order ids are the `*Types::Enum` values.
`-1` means none.

## Shim

`shim/src/`:

- `Transport.{h,cpp}` - framed TCP server (abstract interface; a shared-memory transport can be added).
- `Serializer.{h,cpp}` - BWAPI `Game`/`Unit`/`Player` -> FlatBuffers (`GameStart`, `Frame`, `GameEnd`).
- `CommandApplier.{h,cpp}` - `Commands` -> `Unit::issueCommand`, `Game::set*`, `draw*`. Draws are
  re-issued on skipped frames so overlays persist when `frame_skip > 1`.
- `Bridge.{h,cpp}` - handshake, per-frame exchange, reconnect handling, timing stats
  (`--stats-every N`; prints serialize/rtt/apply averages and decisions/s).
- `main_client.cpp` - Windows `shim.exe` (BWAPI client loop, modelled on STARTcraft).
- `main_module.cpp` - `shim_module.{dll,so}` AI module (used by OpenBW; also loadable by native BWAPI
  via `bwapi.ini [ai] ai = bwapi-data/AI/shim_module.dll`).

Options: `--host`, `--port`, `--stats-every`, `--verbose`, `--no-auto-resume`, or env
`BWBOT_HOST`, `BWBOT_PORT`, `BWBOT_STATS_EVERY`, `BWBOT_VERBOSE`, `BWBOT_AUTO_RESUME`.

CMake: `-DSHIM_BACKEND=bwapi` (default; Win32 only) builds the vendored BWAPI 4.4.0 `BWAPILIB` +
`BWAPIClient` and links them statically. `-DSHIM_BACKEND=openbw -DOPENBW_BWAPI_SRC=... -DOPENBW_BWAPI_BUILD=...`
compiles against the OpenBW fork (`SHIM_OPENBW` guards the few API differences, e.g. the fork's
`countdownTimer()` throws). FlatBuffers headers are fetched by CMake; `-DFLATC_EXECUTABLE` skips
building `flatc`.

## Troubleshooting

- **Shim prints `Game table mapping not found`**: normal while StarCraft is not running yet; it retries.
- **Bot connects but frame count stays 0**: single-player StarCraft is paused (window lost focus).
  The shim auto-resumes; click the game window if you disabled that.
- **"Starcraft Tips" dialog on game start**: StarCraft stores *Show Tips at Startup* in the registry
  (`HKCU\Software\Blizzard Entertainment\Starcraft`, `tip`). `setup_windows.ps1` and `run_native.ps1`
  set it to 0 before every launch; if you start the game some other way, untick the box once or run
  `reg add "HKCU\Software\Blizzard Entertainment\Starcraft" /v tip /t REG_DWORD /d 0 /f`.
- **`bind failed 10048`**: another shim owns the port; `scripts\run_native.ps1 -Stop` or use `--port`.
- **OpenBW build errors about `uint32_t`**: the fork predates GCC 13; the scripts add `-include cstdint`.
- **Python from Windows can't reach WSL**: WSL2 localhost forwarding is on by default; check
  `wsl -d Ubuntu -- ss -ltnp | grep 8765` and Windows firewall.
- StarCraft crashes on Windows > 10 are rare with `wmode.dll` injected (windowed mode); the shim
  reconnects automatically when the game restarts.

## References

- Competition zip (AIIDE/BASIL): `scripts\pack_competition.ps1` → `dist/competition/<Name>/`. See
  [docs/competition.md](docs/competition.md). `BotName.dll` is the 32-bit shim module; `bot.exe` is
  the frozen 64-bit Python brain.
- BWAPI 4.4.0: https://github.com/bwapi/bwapi (vendored under `shim/third_party/bwapi`, LGPL)
- BWEM-community: https://github.com/N00byEdge/BWEM-community (vendored under
  `shim/third_party/bwem-community`, MIT/X11)
- OpenBW: https://github.com/OpenBW/openbw and https://github.com/OpenBW/bwapi
- STARTcraft (launch procedure, client loop reference): https://github.com/davechurchill/STARTcraft
- Game files mirror: https://davechurchill.ca/starcraft/
- FlatBuffers: https://flatbuffers.dev
