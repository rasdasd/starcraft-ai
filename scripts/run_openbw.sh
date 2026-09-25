#!/usr/bin/env bash
# Start a headless OpenBW game hosting the shim AI module (listens for the Python bot on TCP).
#
#   wsl -d Ubuntu -- bash scripts/run_openbw.sh [--port 8765] [--map maps/BroodWar/sscai/(4)Python.scx]
#                                               [--race Terran] [--enemy-race Zerg] [--ui] [--stats-every N]
#
# Then, from Windows or WSL:  python -m bwbot.run adjutant --no-gui --speed 0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAME="$ROOT/wsl/game"
LAUNCHER="$ROOT/wsl/bwapi/build/bin/BWAPILauncher"
LIBDIR="$ROOT/wsl/bwapi/build/lib"

PORT=8765; HOST=0.0.0.0; MAP=""; RACE=""; ENEMY=""; UI=0; STATS=500
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --map) MAP="$2"; shift 2 ;;
    --race) RACE="$2"; shift 2 ;;
    --enemy-race) ENEMY="$2"; shift 2 ;;
    --ui) UI=1; shift ;;
    --stats-every) STATS="$2"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

[[ -x "$LAUNCHER" ]] || { echo "BWAPILauncher missing - run scripts/setup_wsl.sh" >&2; exit 1; }
[[ -f "$GAME/StarDat.mpq" || -L "$GAME/StarDat.mpq" ]] || { echo "mpq files missing in $GAME - run scripts/setup_wsl.sh" >&2; exit 1; }

# Keep the module in sync with the latest build.
cp -f "$ROOT/shim/build-openbw/shim_module.so" "$GAME/bwapi-data/AI/shim_module.so"

export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}"
export BWBOT_HOST="$HOST" BWBOT_PORT="$PORT" BWBOT_STATS_EVERY="$STATS"
export OPENBW_ENABLE_UI="$UI"
[[ -n "$MAP" ]]   && export BWAPI_CONFIG_AUTO_MENU__MAP="$MAP"
[[ -n "$RACE" ]]  && export BWAPI_CONFIG_AUTO_MENU__RACE="$RACE"
[[ -n "$ENEMY" ]] && export BWAPI_CONFIG_AUTO_MENU__ENEMY_RACE="$ENEMY"

cd "$GAME"
echo "[run_openbw] BWAPILauncher in $GAME, shim on $HOST:$PORT (ui=$UI)"
exec "$LAUNCHER"
