#!/usr/bin/env bash
# Build OpenBW (headless Brood War engine) + its BWAPI fork + the shim AI module inside WSL2/Linux.
#
#   wsl -d Ubuntu -- bash scripts/setup_wsl.sh            # as your user (needs passwordless sudo for apt), or
#   wsl -d Ubuntu -u root -- bash scripts/setup_wsl.sh    # as root
#
# Layout (all under the repo, git-ignored):
#   wsl/openbw            openbw/openbw checkout
#   wsl/bwapi             openbw/bwapi checkout (develop-openbw branch)
#   wsl/bwapi/build       BWAPILauncher, libBWAPI.so, libBWAPILIB.so, ...
#   shim/build-openbw     shim_module.so (AI module loaded by BWAPILauncher)
#   wsl/game              working dir for BWAPILauncher (mpq files, bwapi-data/, maps/)
#
# Environment knobs: OPENBW_ENABLE_UI=1 to also build the SDL2 viewer (default 0 = headless only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WSL="$ROOT/wsl"
UI="${OPENBW_ENABLE_UI:-0}"
JOBS="$(nproc)"
mkdir -p "$WSL"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
step "apt packages"
SUDO=""
if [[ $EUID -ne 0 ]]; then SUDO="sudo"; fi
PKGS=(build-essential cmake ninja-build git pkg-config python3 python3-venv python3-pip ca-certificates curl unzip)
if [[ "$UI" == "1" ]]; then PKGS+=(libsdl2-dev libsdl2-mixer-dev libsdl2-image-dev); fi
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get -qq update
$SUDO apt-get -qq install -y "${PKGS[@]}" >/dev/null
gcc --version | head -1
cmake --version | head -1

# ---------------------------------------------------------------------------
step "clone openbw/openbw and openbw/bwapi"
if [[ ! -d "$WSL/openbw/.git" ]]; then
  git clone --depth 1 https://github.com/OpenBW/openbw.git "$WSL/openbw"
else
  echo "  (exists) $WSL/openbw"
fi
if [[ ! -d "$WSL/bwapi/.git" ]]; then
  git clone --depth 1 --branch develop-openbw https://github.com/OpenBW/bwapi.git "$WSL/bwapi"
else
  echo "  (exists) $WSL/bwapi"
fi

# ---------------------------------------------------------------------------
step "build OpenBW + BWAPI fork (BWAPILauncher)"
BW_BUILD="$WSL/bwapi/build"
mkdir -p "$BW_BUILD"
cmake -S "$WSL/bwapi" -B "$BW_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENBW_DIR="$WSL/openbw" \
  -DOPENBW_ENABLE_UI="$UI" \
  -DCMAKE_CXX_FLAGS="-w -include cstdint" \
  >/dev/null
# (-include cstdint: the fork predates GCC 13's stricter libstdc++ headers; uint32_t is used unqualified)
cmake --build "$BW_BUILD" --target BWAPILauncher BWAPILIB -j"$JOBS"
ls -la "$BW_BUILD/bin/BWAPILauncher" "$BW_BUILD"/lib/libBWAPILIB.* 2>/dev/null || true

# ---------------------------------------------------------------------------
step "build shim AI module against the OpenBW BWAPI fork"
SHIM_BUILD="$ROOT/shim/build-openbw"
cmake -S "$ROOT/shim" -B "$SHIM_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSHIM_BACKEND=openbw \
  -DOPENBW_BWAPI_SRC="$WSL/bwapi" \
  -DOPENBW_BWAPI_BUILD="$BW_BUILD" \
  >/dev/null
cmake --build "$SHIM_BUILD" -j"$JOBS"
ls -la "$SHIM_BUILD/shim_module.so"

# flatc is built as part of the shim build (FLATC_EXECUTABLE unset) - keep a copy for gen_proto.sh
FLATC="$(find "$SHIM_BUILD" -type f -name flatc -perm -u+x | head -n1 || true)"
if [[ -n "$FLATC" ]]; then
  mkdir -p "$WSL/flatbuffers" && cp -f "$FLATC" "$WSL/flatbuffers/flatc"
  echo "  flatc -> $WSL/flatbuffers/flatc"
fi

# ---------------------------------------------------------------------------
step "game archive ($ROOT/game)"
GAME_SRC="$ROOT/game"
if ! find "$GAME_SRC" -maxdepth 1 -iname 'stardat.mpq' 2>/dev/null | grep -q .; then
  # Native Linux (no Windows setup): same archive setup_windows.ps1 uses (StarCraft 1.16.1 +
  # BWAPI 4.4.0 + map packs, redistributed with Blizzard's permission for AI research).
  ZIP="$WSL/scbw_bwapi440.zip"
  [[ -f "$ZIP" ]] || curl -fL --retry 3 -o "$ZIP" https://davechurchill.ca/starcraft/files/startcraft/scbw_bwapi440.zip
  mkdir -p "$GAME_SRC"
  unzip -q -o "$ZIP" -d "$GAME_SRC"
  # the archive may contain a single top-level folder
  inner="$(find "$GAME_SRC" -mindepth 1 -maxdepth 1 -type d | head -n1)"
  if [[ -n "$inner" ]] && ! find "$GAME_SRC" -maxdepth 1 -iname 'stardat.mpq' | grep -q . \
     && find "$inner" -maxdepth 1 -iname 'stardat.mpq' | grep -q .; then
    shopt -s dotglob; mv "$inner"/* "$GAME_SRC"/; rmdir "$inner"; shopt -u dotglob
  fi
else
  echo "  (exists) $GAME_SRC"
fi

step "OpenBW game directory ($WSL/game)"
GAME="$WSL/game"
mkdir -p "$GAME/bwapi-data/AI" "$GAME/bwapi-data/read" "$GAME/bwapi-data/write" "$GAME/maps/replays"
for f in STARDAT.MPQ BROODAT.MPQ patch_rt.mpq; do
  src="$(find "$GAME_SRC" -maxdepth 1 -iname "$f" | head -n1 || true)"
  if [[ -z "$src" ]]; then echo "!! $f not found in $GAME_SRC - run scripts/setup_windows.ps1 first" >&2; exit 1; fi
  # OpenBW looks for these exact names (case-sensitive on Linux).
  case "${f,,}" in
    stardat.mpq) dst=StarDat.mpq ;;
    broodat.mpq) dst=BrooDat.mpq ;;
    *) dst=Patch_rt.mpq ;;
  esac
  ln -sfn "$src" "$GAME/$dst"
done
ln -sfn "$GAME_SRC/maps/BroodWar" "$GAME/maps/BroodWar"
cp -f "$SHIM_BUILD/shim_module.so" "$GAME/bwapi-data/AI/shim_module.so"

cat > "$GAME/bwapi-data/bwapi.ini" <<EOF
; OpenBW BWAPILauncher configuration (client mode is not available on OpenBW, so the shim is
; loaded in-process as an AI module). Any key can be overridden with env vars, e.g.
; BWAPI_CONFIG_AUTO_MENU__MAP=maps/BroodWar/sscai/(4)Python.scx
[ai]
ai = bwapi-data/AI/shim_module.so
ai_dbg = bwapi-data/AI/shim_module.so
tournament =

[auto_menu]
auto_menu = SINGLE_PLAYER
character_name = BwBot
pause_dbg = OFF
lan_mode = Local Area Network (UDP)
auto_restart = ON
map = maps/BroodWar/sscai/(2)Destination.scx
game =
mapiteration = RANDOM
race = Terran
enemy_count = 1
enemy_race = Random
game_type = MELEE
save_replay = maps/replays/openbw_%MAP%_%BOTRACE%vs%ENEMYRACES%_\$Y\$b\$d_\$H\$M\$S.rep
wait_for_min_players = 2
wait_for_max_players = 8
wait_for_time = 60000

[config]
holiday = OFF
shared_memory = OFF

[window]
windowed = ON
left = 0
top = 0
width = 640
height = 480

[starcraft]
sound = OFF
screenshots = gif
drop_players = ON
EOF

step "Linux brain venv"
if [[ -n "${SUDO_USER:-}" ]]; then
  sudo -u "$SUDO_USER" bash "$ROOT/scripts/setup_brain.sh"
else
  bash "$ROOT/scripts/setup_brain.sh"
fi

step "done"
cat <<EOF
Run a headless game:
  wsl -d Ubuntu -- bash scripts/run_openbw.sh          # starts BWAPILauncher + shim module on port 8765
  python -m bwbot.run adjutant --no-gui   # from Windows or WSL; connects to localhost:8765

Note: OpenBW has no built-in computer opponent - the single-player enemy just sits there.
EOF
