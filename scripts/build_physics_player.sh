#!/usr/bin/env bash
set -euo pipefail

worktree="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
physics_v2=false
observation_v1=false
aligned_observation_v1=false
print_stage=false
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --physics-v2) physics_v2=true ;;
    --observation-v1) observation_v1=true ;;
    --aligned-observation-v1) aligned_observation_v1=true ;;
    --print-stage) print_stage=true ;;
    *) echo "usage: $0 [--physics-v2|--observation-v1|--aligned-observation-v1] [--print-stage]" >&2; exit 2 ;;
  esac
  shift
done
profile_count=0
[[ "$physics_v2" == true ]] && profile_count=$((profile_count + 1))
[[ "$observation_v1" == true ]] && profile_count=$((profile_count + 1))
[[ "$aligned_observation_v1" == true ]] && profile_count=$((profile_count + 1))
if [[ "$profile_count" -gt 1 ]]; then
  echo "player capture profiles are mutually exclusive" >&2
  exit 2
fi
editor="${UNITY_2019_4_41F2:-$HOME/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity}"
project="${MIGRATED_UNITY_PROJECT:-$worktree/tasks/task_template_designer}"
if [[ "$aligned_observation_v1" == true ]]; then
  stage="$worktree/sciencebirdsgames/aligned-observation-v1"
  capture_schema="physics_capture_v2_engine_v1"
elif [[ "$observation_v1" == true ]]; then
  stage="$worktree/sciencebirdsgames/observation-v1"
  capture_schema="observation_capture_engine_v1"
elif [[ "$physics_v2" == true ]]; then
  stage="$worktree/sciencebirdsgames/physics-v2"
  capture_schema="physics_capture_v2_engine_v1"
else
  stage="${NOVPHY_PHYSICS_STAGE:-$worktree/sciencebirdsgames/physics-v1}"
  capture_schema="physics_capture_v1"
fi
if [[ "$print_stage" == true ]]; then
  printf '%s\n' "$stage"
  exit 0
fi
compat="${UNITY_LTS_LIBS:-/tmp/opencode/unity-2019.4-libssl1.1/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib}"
expected_project="$worktree/tasks/task_template_designer"
interface_jar="/mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux/game_playing_interface.jar"
config_source="/mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux/config.xml"
serverbackup_source="/mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux/serverbackup"

[[ "$(realpath "$project")" == "$(realpath "$expected_project")" ]] || { echo "refusing non-migrated Unity project: $project" >&2; exit 2; }
[[ -x "$editor" ]] || { echo "Unity executable is missing: $editor" >&2; exit 2; }
[[ "$(realpath -m "$stage")" != "$(realpath /mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux)" ]] || { echo "refusing production player output" >&2; exit 2; }
package_inputs="$(mktemp "${TMPDIR:-/tmp}/novphy_physics_package_inputs_XXXXXX")"
trap 'rm -f "$package_inputs"' EXIT
python "$worktree/scripts/package_physics_player.py" --payload "$project" --stage "$stage" --worktree "$worktree" \
  --check-worktree-only --write-package-inputs "$package_inputs" \
  --capture-schema "$capture_schema"

mkdir -p "$stage"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/novphy_physics_build_XXXXXX")"
trap 'rm -rf "$build_root"; rm -f "$package_inputs"' EXIT
payload="$build_root/payload"
mkdir -p "$payload"
export NOVPHY_BUILD_OUTPUT="$payload/9001.x86_64"

unity_log="$build_root/unity-build.log"
unity_exit=0
DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 \
LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$editor" -batchmode -nographics -projectPath "$project" \
  -executeMethod NovPhyBuild.BuildPhysicsLinux -quit \
  -logFile "$unity_log" || unity_exit=$?

# Publish the log even when the build fails, so a failure stays diagnosable,
# but never publish Unity's licensing identity lines.
bash "$worktree/scripts/redact_unity_log.sh" "$unity_log" "$stage/unity-build.log"
[[ "$unity_exit" -eq 0 ]] || { echo "Unity build failed with exit $unity_exit; see $stage/unity-build.log" >&2; exit "$unity_exit"; }

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
  --unity-executable "$editor" \
  --interface-jar "$interface_jar" \
  --config-source "$config_source" \
  --serverbackup-source "$serverbackup_source" \
  --package-inputs "$package_inputs" \
  --capture-schema "$capture_schema"
