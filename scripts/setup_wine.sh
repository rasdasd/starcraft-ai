#!/usr/bin/env bash
# Packages for harness.winematch (real StarCraft 1.16.1 under Wine in WSL2, UDP LAN between
# network namespaces). Run as root after scripts/setup_windows.ps1 (game/, shim_module.dll) and
# scripts/setup_brain.sh (the Linux venv):
#
#   wsl -d Ubuntu -u root -- bash scripts/setup_wine.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
dpkg --add-architecture i386
apt-get -qq update
apt-get -qq install -y wine wine32 wine64 xvfb x11-utils iproute2 >/dev/null
wine --version
echo "run: wsl -d Ubuntu -u root --cd <repo>/python -- <venv>/bin/python -m harness.winematch --opponent Locutus"
