# Competition pack (AIIDE / BASIL / SSCAIT)

Tournaments load a 32-bit `BWAPI::AIModule` DLL. The brain in this repo is 64-bit Python, so a
submission is **two processes**:

1. `BotName.dll` — `shim_module.dll` renamed. It is the AI module, runs BWEM, and talks FlatBuffers
   over `127.0.0.1`.
2. `bot.exe` — frozen 64-bit Python (`mybot` or `goliath`). The DLL starts it unless
   `run_proxy.bat` already did.

BWEB is not included. Walls are not the limiter yet.

## Folder the tournament copies

```
BotName/
  README.md                 this file, plus compile notes
  AI/
    BotName.dll             copy of shim/build/shim_module.dll
    bot.exe                 or bot/bot.exe (PyInstaller onedir)
    run_proxy.bat           starts bot.exe (AIIDE proxy hook; CWD = StarCraft root)
  read/                     tournament read I/O (empty)
  write/                    tournament write I/O (empty)
```

`bwapi.ini` on the client is set by the tournament manager:

```
ai = bwapi-data/AI/BotName.dll
```

Working directory is the StarCraft folder. `run_proxy.bat` must use **relative** paths and must
not `cd`.

## Build the pack

From the repo root, after `scripts\setup_windows.ps1` and `scripts\build_shim.ps1`:

```
scripts\pack_competition.ps1                  # BotName=bwbot, policy=mybot
scripts\pack_competition.ps1 -Name Goliath -Bot goliath
```

Output: `dist/competition/<Name>/`. Zip that folder (and a copy of this source tree) for the
organizers.

## Environment

| var | meaning |
|---|---|
| `BWBOT_PORT` | TCP port (default 8765) |
| `BWBOT_BOT` | path to the frozen bot if it is not next to the DLL |
| `BWBOT_BOT_SPEC` | module passed to `bot.exe` (`mybot` or `goliath`) |
| `BWBOT_NO_SPAWN` | set to `1` if `run_proxy.bat` already started the bot |

The DLL looks for `bot.exe` then `bot\bot.exe` in its own directory.

## What to tell the organizers

- Race: Terran
- BWAPI: 4.4.0
- Bot type: AIModule DLL + proxy Python brain
- Learns using file I/O: optional. Logs go to `bwapi-data/write/bwbot-logs/`. A fitted
  `policy.npz` can be placed in `bwapi-data/read/` or pointed at with `BWBOT_MODEL`.
- Libraries: BWAPI 4.4.0 (vendored), BWEM-community (MIT/X11, Igor Dimitrijevic), FlatBuffers,
  numpy. Not a Steamhammer / UAlbertaBot fork.
- Compile: Visual Studio 2022 C++ (Win32), CMake, Ninja. `scripts\build_shim.ps1`. Python 3.12 x64
  + `pip install -e python` + PyInstaller for the brain.

## BASIL / SSCAIT

Same zip. They load the DLL from `bwapi-data/AI/`. If the ladder has no `run_proxy.bat` hook, the
DLL spawn path is the one that matters — keep `bot.exe` next to the DLL and do not set
`BWBOT_NO_SPAWN`.
