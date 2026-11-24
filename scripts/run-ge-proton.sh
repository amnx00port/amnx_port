#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
app="$repo/target/x86_64-pc-windows-msvc/release/victory.exe"
title_exe="${1:-${VICTORY_TITLE_EXE:-/mnt/data/Games/GTA6/GTA6.Xbox.v1.0.2.1/GTA6.Xbox.v1.0.2.1/game_scarlett_master.exe}}"

if [[ -z "${GE_PROTON:-}" || ! -x "$GE_PROTON/proton" ]]; then
  echo "GE-Proton not found. Enter nix-shell or set GE_PROTON=/path/to/GE-Proton." >&2
  exit 1
fi
if [[ ! -f "$app" ]]; then
  echo "Windows build missing; building it now..." >&2
  (cd "$repo" && cargo xwin build --release -p victory --target x86_64-pc-windows-msvc)
fi
if [[ ! -f "$title_exe" ]]; then
  echo "Title executable not found: $title_exe" >&2
  echo "Pass its path as the first argument or set VICTORY_TITLE_EXE." >&2
  exit 1
fi

# The title path is an argument to a Windows process, so express it through
# Wine's Z: mapping rather than relying on Unix-path argument conversion.
case "$title_exe" in
  [A-Za-z]:\\*) title_exe_win="$title_exe" ;;
  *) title_exe_win="Z:$(printf '%s' "$title_exe" | sed 's,/,\\,g')" ;;
esac

export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="${STEAM_COMPAT_DATA_PATH:-$HOME/.local/share/victory/ge-proton-compatdata}"
export WINEDLLOVERRIDES="${WINEDLLOVERRIDES:-d3d12=n,b;d3d12core=n,b;dxgi=n,b}"

mkdir -p "$STEAM_COMPAT_DATA_PATH"
if [[ -f "$repo/victory.toml" ]]; then
  repo_win="Z:$(printf '%s' "$repo" | sed 's,/,\\,g')"
  export VICTORY_CONFIG="$repo_win\\victory.toml"
fi
# vkd3d-proton writes its pipeline cache relative to the process working
# directory. Keep it (and VIctory's default config lookup) outside the repo.
cd "$STEAM_COMPAT_DATA_PATH"
exec "$GE_PROTON/proton" run "$app" run "$title_exe_win" --execute
