#!/usr/bin/env bash
# Linux/WSL Python brain: venv (outside /mnt/* under WSL, where exec/IO is slow), deps, generated code.
#
#   bash scripts/setup_brain.sh            # venv at $BWBOT_VENV, default ~/.venvs/bwbot
#   BWBOT_VENV=/opt/bwbot bash scripts/setup_brain.sh
#
# The repo's python/ dir is put on the venv's path with a .pth file (no editable install, so nothing
# is written into a Windows-shared checkout). Works with Python >= 3.10. The chosen venv path is
# recorded in wsl/brain_venv so the self-play harness can find it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${BWBOT_VENV:-$HOME/.venvs/bwbot}"
PY="${PYTHON:-python3}"

"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 required, found {sys.version.split()[0]}")
EOF

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> venv $VENV"
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q "numpy>=1.26" "flatbuffers>=24.3" "pytest>=8"

SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$ROOT/python" > "$SITE/bwbot_repo.pth"

if [[ ! -f "$ROOT/python/bwbot/generated/bw.py" ]]; then
  echo "==> gen_proto"
  bash "$ROOT/scripts/gen_proto.sh"
fi

mkdir -p "$ROOT/wsl"
echo "$VENV" > "$ROOT/wsl/brain_venv"
"$VENV/bin/python" -c "import bwbot, blackboard, adjutant; print('brain ok:', bwbot.__file__)"
echo "brain venv: $VENV   (python: $VENV/bin/python)"
