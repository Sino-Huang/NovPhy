#!/usr/bin/env bash
set -euo pipefail

worktree="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
editor="${UNITY_2019_4_41F2:-$HOME/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity}"
project="${MIGRATED_UNITY_PROJECT:-$worktree/tasks/task_template_designer}"
stage="${NOVPHY_PHYSICS_STAGE:-$worktree/sciencebirdsgames/physics-v1}"
compat="${UNITY_LTS_LIBS:-/tmp/opencode/unity-2019.4-libssl1.1/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib}"
expected_editor_sha="32252cb8eca087743e500596e093061a906203703915c2d3c2fb2f8a372bc150"
expected_project="$worktree/tasks/task_template_designer"
interface_jar="/mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux/game_playing_interface.jar"
config_source="/mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux/config.xml"
serverbackup_source="/mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux/serverbackup"

[[ "$(realpath "$project")" == "$(realpath "$expected_project")" ]] || { echo "refusing non-migrated Unity project: $project" >&2; exit 2; }
[[ "$(sha256sum "$editor" | awk '{print $1}')" == "$expected_editor_sha" ]] || { echo "pinned Unity executable checksum mismatch" >&2; exit 2; }
[[ "$(realpath -m "$stage")" != "$(realpath /mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux)" ]] || { echo "refusing production player output" >&2; exit 2; }
package_inputs="$(mktemp "${TMPDIR:-/tmp}/novphy_physics_package_inputs_XXXXXX")"
trap 'rm -f "$package_inputs"' EXIT
python "$worktree/scripts/package_physics_player.py" --payload "$project" --stage "$stage" --worktree "$worktree" \
  --migration-provenance "$worktree/.omo/evidence/world-model-physics-instrumentation/task-2-migration-provenance.json" --check-worktree-only --write-package-inputs "$package_inputs"

mkdir -p "$stage"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/novphy_physics_build_XXXXXX")"
trap 'rm -rf "$build_root"; rm -f "$package_inputs"' EXIT
payload="$build_root/payload"
mkdir -p "$payload"
export NOVPHY_BUILD_OUTPUT="$payload/9001.x86_64"

DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 \
LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$editor" -batchmode -nographics -projectPath "$project" \
  -executeMethod NovPhyBuild.BuildPhysicsLinux -quit \
  -logFile "$stage/unity-build.log"

cp "$interface_jar" "$payload/"
mv "$payload/9001.x86_64" "$payload/9001-player.x86_64"
cp "$worktree/scripts/9001-player-wrapper.sh" "$payload/9001.x86_64"
chmod +x "$payload/9001.x86_64" "$payload/9001-player.x86_64"
sed 's#9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00026_0_1_010101_0_1.xml#9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels/3_9_6_1.xml#' \
  "$config_source" > "$payload/config.xml"
cp "$serverbackup_source" "$payload/"
python "$worktree/scripts/package_physics_player.py" \
  --payload "$payload" \
  --stage "$stage" \
  --worktree "$worktree" \
  --migration-provenance "$worktree/.omo/evidence/world-model-physics-instrumentation/task-2-migration-provenance.json" \
  --unity-executable "$editor" \
  --interface-jar "$interface_jar" \
  --config-source "$config_source" \
  --serverbackup-source "$serverbackup_source" \
  --package-inputs "$package_inputs"
