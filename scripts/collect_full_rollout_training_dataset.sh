#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

plan_dir="${PLAN_DIR:-data/rollout_dataset_plan_$(date +%Y%m%d_%H%M%S)}"
out_root="${OUT_ROOT:-}"
display_id="${DISPLAY_ID:-:149}"
count="${ROLLOUT_COUNT:-12}"
fps="${ROLLOUT_FPS:-30}"
duration="${ROLLOUT_DURATION:-5}"
display_label="${display_id#:}"
xvnc_log="${XVNC_LOG:-/tmp/novphy_rollout_xvnc_${USER:-user}_${display_label}.log}"
workers="${WORKERS:-6}"
agent_port_base="${AGENT_PORT_BASE:-2004}"
game_port_base="${GAME_PORT_BASE:-9001}"
resume="${RESUME:-}"
train_target="${TRAIN_TARGET_PER_BUCKET:-100}"
dev_target="${DEV_TARGET_PER_BUCKET:-20}"
test_target="${TEST_TARGET_PER_BUCKET:-0}"
seed="${PARTITION_SEED:-novphy-rollout-dataset-v1}"
expected_buckets="${EXPECTED_BUCKETS:-80}"
level_type_prefix="${LEVEL_TYPE_PREFIX:-type010}"
proc_root="${NOVPHY_PROC_ROOT:-/proc}"
x11_tmp_root="${NOVPHY_X11_TMP_ROOT:-/tmp}"
include_test=0
train_only=0
physics_capture="${PHYSICS_CAPTURE_V1:-}"
physics_player_dir="${PHYSICS_PLAYER_DIR:-}"
physics_player_archive="${PHYSICS_PLAYER_ARCHIVE:-}"
physics_smoke_marker="${PHYSICS_SMOKE_MARKER:-}"
show_help=0

usage() {
  cat <<'EOF'
Collect the full NovPhy selected-split rollout dataset.

This script inventories an existing output root, publishes a capped deterministic
train/dev collection plan by default, starts Xvnc, then runs only the generated
script. Pass --include-test to additionally select test levels.
Pass --train-only to select just the train split, for a scoped inventory too small to
fund a leakage-free dev split (mutually exclusive with --include-test).
Failed levels are logged to PLAN_DIR/failed_levels.tsv; collection continues to
later levels, then exits nonzero if any level failed.

Environment overrides:
  PLAN_DIR          Output directory for partitions and generated commands
  OUT_ROOT          Required existing output root for collected rollout data
  DISPLAY_ID        X display for Xvnc and desktop capture, default :149
  ROLLOUT_COUNT     Rollouts per episode, default 12
  ROLLOUT_FPS       Capture FPS, default 30
  ROLLOUT_DURATION  Minimum post-shot capture duration, default 5
  WORKERS           Parallel rollout workers, default 6. Workers use isolated X displays, engine dirs, agent ports, and game ports.
  AGENT_PORT_BASE   First worker agent port, default 2004; later workers add 10
  GAME_PORT_BASE    First worker game port, default 9001; later workers add 10
  RESUME=1          Required. Preserve existing output paths and schedule only absent paths.
  TRAIN_TARGET_PER_BUCKET
                    Episodes per normal/novel bucket for train, default 100
  DEV_TARGET_PER_BUCKET
                    Episodes per normal/novel bucket for dev, default 20
  TEST_TARGET_PER_BUCKET
                    Episodes per normal/novel bucket for test, default 0; used only with --include-test
  PARTITION_SEED    Deterministic planner seed, default novphy-rollout-dataset-v1
  EXPECTED_BUCKETS  Declared level-inventory bucket count, default 80 (the production
                    inventory). Lower it only for a deliberately scoped inventory such
                    as the single-level staged physics player; the default fails closed
                    on a truncated production inventory.
  LEVEL_TYPE_PREFIX Level-type directory prefix to inventory, default type010 (the
                    production naming). The staged physics player ships its level
                    under type2.
  NOVPHY_ALLOW_NETWORK_LISTENERS=1
                     Required with WORKERS>1 after confirming the host is isolated/firewalled.
  XVNC_LOG          Xvnc log path, default /tmp/novphy_rollout_xvnc_${USER}_${DISPLAY_ID-without-colon}.log
  NOVPHY_YES=1      Skip the confirmation prompt
  PHYSICS_CAPTURE_V1=1
                    Opt into enriched physics_capture_v1 collection. Requires
                    PHYSICS_PLAYER_DIR (containing player.tar and a smoke marker)
                    or PHYSICS_PLAYER_ARCHIVE plus PHYSICS_SMOKE_MARKER, and a
                    separate OUT_ROOT from the active cohort.
  PHYSICS_PLAYER_DIR
                    Explicit staged enriched-player directory; defaults to
                    <dir>/player.tar and <dir>/physics_capture_v1_smoke.json
  PHYSICS_PLAYER_ARCHIVE
                    Explicit staged player archive when PHYSICS_PLAYER_DIR is unset
  PHYSICS_SMOKE_MARKER
                    Fresh successful physics_capture_v1 smoke JSON marker

Example:
  source ~/cd_novphy && RESUME=1 OUT_ROOT=/absolute/path/to/existing/output/root NOVPHY_YES=1 scripts/collect_full_rollout_training_dataset.sh
  source ~/cd_novphy && RESUME=1 TEST_TARGET_PER_BUCKET=20 OUT_ROOT=/absolute/path/to/existing/output/root NOVPHY_YES=1 scripts/collect_full_rollout_training_dataset.sh --include-test
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --include-test)
      include_test=1
      ;;
    --train-only)
      train_only=1
      ;;
    --help|-h)
      show_help=1
      ;;
    *)
      echo "Unknown launcher option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$train_only" == "1" && "$include_test" == "1" ]]; then
  echo "--train-only and --include-test are mutually exclusive." >&2
  exit 2
fi

if [[ "$show_help" == "1" ]]; then
  if [[ "$physics_capture" == "1" ]]; then
    if [[ -z "$out_root" ]]; then
      echo "PHYSICS_CAPTURE_V1=1 requires OUT_ROOT, even for --help." >&2
      exit 2
    fi
    active_root_canonical="$(realpath -m -- "$repo_root/data/novphy_rollouts_dataset_20260708_171531")"
    help_root_canonical="$(realpath -m -- "$out_root")"
    if [[ "$help_root_canonical" == "$active_root_canonical" ]]; then
      echo "PHYSICS_CAPTURE_V1 cannot target the active cohort root: $help_root_canonical" >&2
      exit 2
    fi
    if [[ -n "$physics_player_dir" ]]; then
      [[ -z "$physics_player_archive" ]] || { echo "PHYSICS_PLAYER_DIR and PHYSICS_PLAYER_ARCHIVE are mutually exclusive." >&2; exit 2; }
      [[ -f "$physics_player_dir/player.tar" ]] && physics_player_archive="$physics_player_dir/player.tar" || physics_player_archive="$physics_player_dir"
      [[ -n "$physics_smoke_marker" ]] || physics_smoke_marker="$physics_player_dir/physics_capture_v1_smoke.json"
    fi
    if [[ -z "$physics_player_archive" || -z "$physics_smoke_marker" || ! -e "$physics_player_archive" || ! -f "$physics_smoke_marker" ]]; then
      echo "PHYSICS_CAPTURE_V1 requires a staged player archive/directory and a valid smoke marker." >&2
      exit 2
    fi
    echo "PHYSICS_CAPTURE_V1 staging is valid." >&2
    exit 0
  fi
  usage
  exit 0
fi

parse_port_base() {
  local name="$1" value="$2" normalized
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a decimal integer in 1..65535." >&2
    return 1
  fi
  normalized="$value"
  while [[ ${#normalized} -gt 1 && "${normalized:0:1}" == "0" ]]; do
    normalized="${normalized:1}"
  done
  if [[ ${#normalized} -gt 5 || ( ${#normalized} -eq 5 && "$normalized" > "65535" ) || "$normalized" == "0" ]]; then
    echo "$name must be a decimal integer in 1..65535." >&2
    return 1
  fi
  REPLY=$((10#$normalized))
}

if ! parse_port_base "AGENT_PORT_BASE" "$agent_port_base"; then
  exit 2
fi
agent_port_base="$REPLY"

if ! parse_port_base "GAME_PORT_BASE" "$game_port_base"; then
  exit 2
fi
game_port_base="$REPLY"

if [[ "$resume" != "1" ]]; then
  echo "This launcher requires RESUME=1." >&2
  exit 2
fi

if ! [[ "$workers" =~ ^[0-9]+$ ]]; then
  echo "WORKERS must be a positive integer." >&2
  exit 2
fi

while [[ ${#workers} -gt 1 && "${workers:0:1}" == "0" ]]; do
  workers="${workers:1}"
done
if [[ ${#workers} -gt 4 || "$workers" -lt 1 ]]; then
  echo "WORKERS must be at least 1." >&2
  exit 2
fi

workers=$((10#$workers))
final_agent_port=$((agent_port_base + (workers - 1) * 10))
final_game_port=$((game_port_base + (workers - 1) * 10))
if [[ "$final_agent_port" -gt 65535 ]]; then
  echo "AGENT_PORT_BASE final worker port exceeds 65535." >&2
  exit 2
fi
if [[ "$final_game_port" -gt 65535 ]]; then
  echo "GAME_PORT_BASE final worker port exceeds 65535." >&2
  exit 2
fi
if (( (agent_port_base - game_port_base) % 10 == 0 && agent_port_base <= final_game_port && game_port_base <= final_agent_port )); then
  if (( agent_port_base > game_port_base )); then
    overlap_port="$agent_port_base"
  else
    overlap_port="$game_port_base"
  fi
  echo "agent and game port families must be disjoint: $overlap_port." >&2
  exit 2
fi

if [[ -z "${OUT_ROOT+x}" || -z "$out_root" ]]; then
  echo "RESUME=1 requires an explicit OUT_ROOT." >&2
  exit 2
fi
if [[ ! -d "$out_root" ]]; then
  echo "RESUME=1 requires OUT_ROOT to exist: $out_root" >&2
  exit 2
fi

if ! [[ "$display_id" =~ ^:[0-9]+$ ]]; then
  echo "DISPLAY_ID must be an X display like :149 when this launcher manages Xvnc." >&2
  exit 2
fi

if [[ "$workers" -gt 1 && "${NOVPHY_ALLOW_NETWORK_LISTENERS:-}" != "1" ]]; then
  echo "WORKERS>1 opens additional unauthenticated local network listeners; set NOVPHY_ALLOW_NETWORK_LISTENERS=1 only on an isolated/firewalled host." >&2
  exit 2
fi

out_root_canonical="$(realpath -m -- "$out_root")"

physics_args=()
if [[ "$physics_capture" == "1" ]]; then
  active_root_canonical="$(realpath -m -- "$repo_root/data/novphy_rollouts_dataset_20260708_171531")"
  if [[ "$out_root_canonical" == "$active_root_canonical" ]]; then
    echo "PHYSICS_CAPTURE_V1 cannot target the active cohort root: $out_root_canonical" >&2
    exit 2
  fi
  if [[ -n "$physics_player_dir" && -n "$physics_player_archive" ]]; then
    echo "PHYSICS_PLAYER_DIR and PHYSICS_PLAYER_ARCHIVE are mutually exclusive." >&2
    exit 2
  fi
  if [[ -n "$physics_player_dir" ]]; then
    [[ -f "$physics_player_dir/player.tar" ]] && physics_player_archive="$physics_player_dir/player.tar" || physics_player_archive="$physics_player_dir"
    [[ -n "$physics_smoke_marker" ]] || physics_smoke_marker="$physics_player_dir/physics_capture_v1_smoke.json"
  fi
  if [[ -z "$physics_player_archive" || -z "$physics_smoke_marker" || ! -e "$physics_player_archive" || ! -f "$physics_smoke_marker" ]]; then
    echo "PHYSICS_CAPTURE_V1 requires PHYSICS_PLAYER_DIR or PHYSICS_PLAYER_ARCHIVE plus PHYSICS_SMOKE_MARKER." >&2
    exit 2
  fi
  if ! physics_provenance_line="$(python - "$physics_player_archive" "$physics_smoke_marker" <<'PY'
import sys
from scripts.prepare_rollout_dataset import resolve_physics_capture_provenance
provenance = resolve_physics_capture_provenance(__import__("pathlib").Path(sys.argv[1]), __import__("pathlib").Path(sys.argv[2]))
print("\t".join((provenance.player_version, provenance.protocol_version, provenance.archive_path)))
PY
  )"; then
    echo "PHYSICS_CAPTURE_V1 staged player smoke validation failed." >&2
    exit 2
  fi
  IFS=$'\t' read -r physics_player_version physics_protocol_version physics_archive_path <<<"$physics_provenance_line"
  if [[ -n "$physics_player_dir" ]]; then
    physics_args+=(--physics-capture-v1 --physics-player-dir "$physics_player_dir")
  else
    physics_args+=(--physics-capture-v1 --physics-player-archive "$physics_player_archive" --physics-smoke-marker "$physics_smoke_marker")
  fi
fi

refuse_active_collector() {
  local proc_dir cmdline_path cmdline
  if [[ ! -d "$proc_root" ]]; then
    echo "Cannot inspect process state: $proc_root is unavailable." >&2
    return 1
  fi
  for proc_dir in "$proc_root"/[0-9]*; do
    [[ -e "$proc_dir" ]] || continue
    cmdline_path="$proc_dir/cmdline"
    if [[ ! -r "$cmdline_path" ]]; then
      echo "Cannot inspect process state: $cmdline_path is unreadable." >&2
      return 1
    fi
    cmdline="$(tr '\000' ' ' < "$cmdline_path")"
    if [[ "$cmdline" == *"collect_rollouts.py"* && "$cmdline" == *"$out_root_canonical/"* ]]; then
      echo "Matching collect_rollouts.py is already active for OUT_ROOT descendants: $out_root_canonical" >&2
      return 1
    fi
  done
  return 0
}

refuse_occupied_port() {
  local port="$1" label="$2" port_hex tcp_path inspection
  printf -v port_hex '%04X' "$port"
  for tcp_path in "$proc_root/net/tcp" "$proc_root/net/tcp6"; do
    if [[ ! -r "$tcp_path" ]]; then
      echo "Cannot inspect $label port $port: $tcp_path is unreadable." >&2
      return 1
    fi
    if ! inspection="$(awk -v port="$port_hex" '
      $4 == "0A" && $2 ~ (":" port "$") { found = 1 }
      END { print found ? "occupied" : "available" }
    ' "$tcp_path")"; then
      echo "Cannot inspect $label port $port: failed to parse $tcp_path." >&2
      return 1
    fi
    if [[ "$inspection" == "occupied" ]]; then
      echo "$label port $port is occupied." >&2
      return 1
    fi
    if [[ "$inspection" != "available" ]]; then
      echo "Cannot inspect $label port $port: unexpected inspection output for $tcp_path." >&2
      return 1
    fi
  done
  return 0
}

refuse_occupied_display() {
  local display="$1" display_number
  display_number="${display#:}"
  if [[ ! -d "$x11_tmp_root" || ! -d "$x11_tmp_root/.X11-unix" ]]; then
    echo "Cannot inspect X display state under $x11_tmp_root." >&2
    return 1
  fi
  if [[ -e "$x11_tmp_root/.X${display_number}-lock" || -S "$x11_tmp_root/.X11-unix/X${display_number}" ]]; then
    echo "X display $display is occupied." >&2
    return 1
  fi
  return 0
}

preflight_resources() {
  local worker_index worker_display agent_port game_port base_display_number
  refuse_active_collector
  base_display_number="${display_id#:}"
  for ((worker_index = 0; worker_index < workers; worker_index++)); do
    worker_display=":$((base_display_number + worker_index))"
    agent_port="$((agent_port_base + worker_index * 10))"
    game_port="$((game_port_base + worker_index * 10))"
    refuse_occupied_display "$worker_display"
    refuse_occupied_port "$agent_port" "agent"
    refuse_occupied_port "$game_port" "game"
  done
}

preflight_resources

lock_path="${out_root_canonical}.novphy_rollout_collection.lock"
exec {collection_lock_fd}>"$lock_path"
if ! flock -n "$collection_lock_fd"; then
  echo "Another collector holds the output-root lock: $lock_path" >&2
  exit 1
fi

plan_command=(
  python scripts/prepare_rollout_dataset.py plan
  --output-dir "$plan_dir"
  --command-output-root "$out_root_canonical"
  --train-target "$train_target"
  --dev-target "$dev_target"
  --seed "$seed"
  --count "$count"
  --fps "$fps"
  --duration "$duration"
  --display "$display_id"
  --workers "$workers"
  --agent-port-base "$agent_port_base"
  --game-port-base "$game_port_base"
  --expected-buckets "$expected_buckets"
  --level-type-prefix "$level_type_prefix"
)
if [[ "$physics_capture" == "1" ]]; then
  plan_command+=("${physics_args[@]}")
fi
selected_split_label="train/dev"
collection_script="$plan_dir/collect_train_dev.sh"
if [[ "$train_only" == "1" ]]; then
  plan_command+=(--train-only)
  selected_split_label="train"
  collection_script="$plan_dir/collect_train.sh"
elif [[ "$include_test" == "1" ]]; then
  plan_command+=(--include-test --test-target "$test_target")
  selected_split_label="train/dev/test"
  collection_script="$plan_dir/collect_train_dev_test.sh"
fi
"${plan_command[@]}"

echo "This will collect the newly scheduled capped $selected_split_label rollout cohort."
echo "Plan directory: $plan_dir"
echo "Output root:    $out_root_canonical"
echo "Display:        $display_id"
echo "Rollouts/level: $count"
echo "FPS:            $fps"
echo "Duration:       $duration"
echo "Workers:        $workers"
echo "Agent port base: $agent_port_base"
echo "Game port base:  $game_port_base"
echo "Train target/bucket: $train_target"
echo "Dev target/bucket:   $dev_target"
if [[ "$include_test" == "1" ]]; then
  echo "Test target/bucket:  $test_target"
fi
echo "Planner seed:        $seed"
echo ""
echo "Reconciled plan summary:"
python - "$plan_dir/collection_plan.json" <<'PY'
import json
import sys

plan = json.loads(open(sys.argv[1], encoding="utf-8").read())
contract = plan["contract"]
counts = plan["counts"]
print(json.dumps({
    "buckets": len(plan["summary"]) // len(plan["selected_splits"]),
    "contract": contract,
    "counts": counts,
}, indent=2, sort_keys=True))
PY
echo ""
echo "Important: worker-local sciencebirdsgames/Linux/config.xml will be rewritten per level"
echo "and left pointing at the final collected level. Selected splits: $selected_split_label."
echo "Failed levels will be logged to $plan_dir/failed_levels.tsv and later levels will continue."

if [[ "${NOVPHY_YES:-}" != "1" ]]; then
  read -r -p "Continue with full $selected_split_label collection? Type 'yes' to proceed: " answer
  if [[ "$answer" != "yes" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

preflight_resources

if ! command -v Xvnc >/dev/null 2>&1; then
  echo "Missing Xvnc on PATH. Initialize the NovPhy environment first, for example: source ~/cd_novphy" >&2
  exit 1
fi

xvnc_pids=()
cleanup() {
  for xvnc_pid in "${xvnc_pids[@]}"; do
    kill "$xvnc_pid" 2>/dev/null || true
    wait "$xvnc_pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

base_display_number="${display_id#:}"
for ((worker_index = 0; worker_index < workers; worker_index++)); do
  worker_display=":$((base_display_number + worker_index))"
  worker_log="${xvnc_log%.log}_${worker_display#:}.log"
  Xvnc "$worker_display" -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 >"$worker_log" 2>&1 &
  xvnc_pids+=("$!")
done
sleep 2
for xvnc_pid in "${xvnc_pids[@]}"; do
  if ! kill -0 "$xvnc_pid" 2>/dev/null; then
    echo "Xvnc failed to start; cleaning up owned Xvnc processes." >&2
    exit 1
  fi
done

bash "$collection_script"
