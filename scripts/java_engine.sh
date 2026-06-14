#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
game_dir="$repo_root/sciencebirdsgames/Linux"
jar_path="$game_dir/game_playing_interface.jar"

if [[ ! -f "$jar_path" ]]; then
  echo "Missing $jar_path" >&2
  echo "Prepare the Linux engine assets before running this script." >&2
  exit 1
fi

cd "$game_dir"
exec java -jar ./game_playing_interface.jar --dev "$@"
