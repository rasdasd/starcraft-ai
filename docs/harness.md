# Game harnesses

Two ways to play many games unattended. Both write `runs/<run-id>/results.jsonl` in the same
format, and `python -m adjutant.learn.report runs/<run-id> [...]` summarizes them (win rates by
player, matchup, map and opening template, with 95% Wilson intervals).

## Self-play in WSL (OpenBW, headless)

Fast (about 1500 frames/s per game) and parallel. Every player is a Python bot from this repo.

```
# inside WSL, after scripts/setup_wsl.sh
cd /mnt/c/starcraft-ai/python
~/.venvs/bwbot/bin/python -m harness.selfplay --p1 adjutant --p2 goliath --p2 sparring.zerg:Pool --games 40 --parallel 4
```

- Player syntax: `module[:Class][@profile][/Race]`, e.g. `adjutant@explore`, `sparring.protoss:Dragoon`.
  The race defaults to the bot class's `race` attribute (sparring bots set it), else Terran.
- `--pool pool.json`: a list of `{"name", "spec", "profile", "race", "env", "args", "weight"}`.
- `--max-frames` (default 20 game minutes): at the limit both brains leave and the higher score
  wins. OpenBW keeps no BWAPI score during the game, so `harness.brain` uses a material score
  (resources gathered + value owned + 2x value killed).
- A side whose brain crashes loses; per-game launcher and brain output is in `runs/<id>/games/<n>/`,
  next to the recorder logs (`logs/<game_id>-a.jsonl`, `-b.jsonl`, sharing one game id) and replays.

Mechanics: two `BWAPILauncher` processes per game meet in an OpenBW LAN game over a private unix
socket (`OPENBW_LAN_MODE=LOCAL`, `OPENBW_LOCAL_PATH`), each hosting the shim module on its own
port with `BWBOT_NO_SPAWN=1`; the harness starts one `python -m harness.brain` per side.

### Sparring bots

`sparring.zerg` (9-pool speedlings), `sparring.zerg:Hydra`, `sparring.protoss` (2-gate zealots),
`sparring.protoss:Dragoon`. Small scripted bots that give self-play Zerg and Protoss opponents.

## Published bots on Windows (native StarCraft 1.16.1)

The Tournament Manager route without Docker. One game at a time, at roughly 1-2x normal speed.

```
cd python
.venv\Scripts\python.exe -m harness.botmatch --list
.venv\Scripts\python.exe -m harness.botmatch --p1 adjutant --opponent "Dave Churchill" --opponent Stardust --games 6
```

- Opponents are downloaded from the SSCAIT API into `bots/<name>/` (git-ignored) on first use.
- Each side gets its own StarCraft folder under `runs/botmatch/inst_a|inst_b` (hard links to `game/`)
  with its own `bwapi-data/BWAPI.dll`, so the opponent runs on the BWAPI version it was built for.
  Our side hosts a "Local PC" game; the opponent joins (`game = JOIN_FIRST`).
- BWAPI 4.1.2 and older find `bwapi.ini` through `HKCU\Software\Blizzard Entertainment\Starcraft\InstallPath`,
  so the harness points it at each side's folder before launching it and restores it afterwards.
- BWTA2 bots need `libgmp-10.dll` and `libmpfr-4.dll`; the harness copies them into `game/` from
  BWMirror's jar the first time. The VC++ 2013 x86 runtime must be installed (it usually is).
- Only our brain reports a result. Games that reach `--max-frames` are judged on BWAPI scores when
  the enemy's score is visible, otherwise they count as draws; set the limit high enough for
  games to finish.
- Close any running StarCraft first; the Local PC network would pick it up.
- Supported opponents: AI_MODULE (DLL) bots. EXE/JAVA client bots are started best-effort.
