#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

plan_dir="${PLAN_DIR:-data/rollout_dataset_plan_$(date +%Y%m%d_%H%M%S)}"
out_root="${OUT_ROOT:-data/rollout_dataset_$(date +%Y%m%d_%H%M%S)}"
display_id="${DISPLAY_ID:-:149}"
count="${ROLLOUT_COUNT:-2}"
fps="${ROLLOUT_FPS:-30}"
duration="${ROLLOUT_DURATION:-5}"
display_label="${display_id#:}"
xvnc_log="${XVNC_LOG:-/tmp/novphy_rollout_xvnc_${USER:-user}_${display_label}.log}"
workers="${WORKERS:-1}"

usage() {
  cat <<'EOF'
Collect the full NovPhy train/dev rollout dataset.

This script plans deterministic train/dev/test partitions, starts Xvnc,
then runs the generated train/dev collection script. Test levels stay held out.
Failed levels are logged to PLAN_DIR/failed_levels.tsv; collection continues to
later levels, then exits nonzero if any level failed.

Environment overrides:
  PLAN_DIR          Output directory for partitions and generated commands
  OUT_ROOT          Output root for collected rollout data
  DISPLAY_ID        X display for Xvnc and desktop capture, default :149
  ROLLOUT_COUNT     Rollouts per level, default 2
  ROLLOUT_FPS       Capture FPS, default 30
  ROLLOUT_DURATION  Minimum post-shot capture duration, default 5
  WORKERS           Parallel rollout workers, default 1. Workers use isolated X displays, engine dirs, agent ports, and game ports.
  NOVPHY_ALLOW_NETWORK_LISTENERS=1
                     Required with WORKERS>1 after confirming the host is isolated/firewalled.
  XVNC_LOG          Xvnc log path, default /tmp/novphy_rollout_xvnc_${USER}_${DISPLAY_ID-without-colon}.log
  NOVPHY_YES=1      Skip the confirmation prompt

Example:
  source ~/cd_novphy && NOVPHY_YES=1 scripts/collect_full_rollout_training_dataset.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if ! [[ "$workers" =~ ^[0-9]+$ ]]; then
  echo "WORKERS must be a positive integer." >&2
  exit 2
fi

if [[ "$workers" -lt 1 ]]; then
  echo "WORKERS must be at least 1." >&2
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

if ! command -v Xvnc >/dev/null 2>&1; then
  echo "Missing Xvnc on PATH. Initialize the NovPhy environment first, for example: source ~/cd_novphy" >&2
  exit 1
fi

echo "This will collect train/dev rollouts for the installed NovPhy levels."
echo "Plan directory: $plan_dir"
echo "Output root:    $out_root"
echo "Display:        $display_id"
echo "Rollouts/level: $count"
echo "FPS:            $fps"
echo "Duration:       $duration"
echo "Workers:        $workers"
echo ""
echo "Important: sciencebirdsgames/Linux/config.xml will be rewritten per level"
echo "and left pointing at the final collected level. Test levels are not collected."
echo "Failed levels will be logged to $plan_dir/failed_levels.tsv and later levels will continue."

if [[ "${NOVPHY_YES:-}" != "1" ]]; then
  read -r -p "Continue with full train/dev collection? Type 'yes' to proceed: " answer
  if [[ "$answer" != "yes" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

python scripts/prepare_rollout_dataset.py plan \
  --output-dir "$plan_dir" \
  --command-output-root "$out_root" \
  --count "$count" \
  --fps "$fps" \
  --duration "$duration" \
  --display "$display_id" \
  --workers "$workers"

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

bash "$plan_dir/collect_train_dev.sh"
