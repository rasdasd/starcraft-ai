# Game harnesses

Three ways to play many games unattended. All write `runs/<run-id>/results.jsonl` in the same
format, and `python -m adjutant.learn.report runs/<run-id> [...]` summarizes them (win rates by
player, matchup, map and opening template, with 95% Wilson intervals).

## Self-play in WSL (OpenBW, headless)

Fast (about 1500 frames/s per game) and parallel. Every player is a Python bot from this repo.

```
# inside WSL, after scripts/setup_wsl.sh
cd /mnt/c/starcraft-ai/python
~/.venvs/bwbot/bin/python -m harness.selfplay --p1 adjutant --p2 adjutant/Zerg --p2 adjutant@search/Protoss --games 40 --parallel 4
```

- Player syntax: `module[:Class][@profile][/Race]`, e.g. `adjutant@explore`, `adjutant/Protoss`.
  The race defaults to Terran.
- `--pool pool.json`: a list of `{"name", "spec", "profile", "race", "env", "args", "weight"}`.
- `--max-frames` (default 20 game minutes): at the limit both brains leave and the higher score
  wins. OpenBW keeps no BWAPI score during the game, so `harness.brain` uses a material score
  (resources gathered + value owned + 2x value killed).
- A side whose brain crashes loses; per-game launcher and brain output is in `runs/<id>/games/<n>/`,
  next to the recorder logs (`logs/<game_id>-a.jsonl`, `-b.jsonl`, sharing one game id) and replays.

Mechanics: two `BWAPILauncher` processes per game meet in an OpenBW LAN game over a private unix
socket (`OPENBW_LAN_MODE=LOCAL`, `OPENBW_LOCAL_PATH`), each hosting the shim module on its own
port with `BWBOT_NO_SPAWN=1`; the harness starts one `python -m harness.brain` per side.

## Published bots under Wine in WSL (fast, parallel)

The recommended way to play published bots. Real StarCraft 1.16.1 runs under Wine; each game slot
puts the two clients in their own Linux network namespaces on a private bridge, so they meet in a
UDP LAN game (the sc-docker layout without Docker). Measured on a 12600K: 250 frames/s for one
game, about 130 frames/s each with 4 games at once (native Windows is capped at about 64 frames/s
and one game).

```
wsl -d Ubuntu -u root -- bash scripts/setup_wine.sh      # once: wine, wine32, xvfb
wsl -d Ubuntu -u root --cd /mnt/c/starcraft-ai/python -- /home/<you>/.venvs/bwbot/bin/python \
    -m harness.winematch --opponent Locutus --opponent Stardust --opponent Pluto --games 12 --parallel 4
```

- Needs root (network namespaces); `wsl -u root` needs no password. The Windows firewall is not
  involved: all game traffic stays on WSL's virtual bridges.
- Bots, `bwapi.ini`, and results are shared with `harness.botmatch`. Opponents not in `bots/` are
  fetched from SSCAIT. Bots from elsewhere need a hand-written `bots/<name>/bot.json` plus `AI/` and
  `BWAPI.dll`, as for Pluto (the release zip unpacked into `AI/`, BWAPI 4.4.0 from `game/`).
- Both clients load the Tournament Manager module (`LocalSpeed 0`, `FrameSkip 256`); without it the
  opponent plays at normal speed (15 frames/s). Wine renders with GDI (`WINEMATCH_RENDERER=gl` to
  compare), which is about 30% faster than OpenGL on llvmpipe.
- Slot `k` uses bridge `scbr<k>` (10.77.k.0/24), namespaces `sc<k>a` / `sc<k>b`, Xvfb displays
  `:100+2k` / `:101+2k` started inside the namespaces (X abstract sockets are per network namespace,
  and WSLg owns `/tmp/.X11-unix`), and instances plus Wine prefixes under `/root/sc/s<k>`.
- A game that has not started after `--lobby-timeout` seconds (a crashed client) is recorded as
  `no_start`. Per-game Wine, Xvfb and brain output is in `runs/<id>/games/<n>/`.
- BWAPI 4.1.2 crashes in StarCraft's character-creation screen, so every instance gets a
  multiplayer character file up front.

## Published bots on Windows (native StarCraft 1.16.1)

The Tournament Manager route without Docker. Two clients on one Windows machine can only meet over
"Local PC", which paces a game at about 64 frames/s and allows one game at a time; prefer
`harness.winematch`. Allow StarCraft.exe through the firewall if you use `--lan-mode` UDP
(port 6111); the brain talks to the shim over TCP on localhost (8790 and up).

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
