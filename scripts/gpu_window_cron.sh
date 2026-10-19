#!/usr/bin/env bash
# cron entry point of the night-window GPU scheduler (docs/gpu-window-scheduler.md). Installed crontab:
#   TZ=Australia/Melbourne
#   */5 * * * * /mnt/array/sukaih/Project/NovPhy/scripts/gpu_window_cron.sh
# Jobs inherit this environment: the novphy env and env.sh, as in the documented manual launches.
set -eo pipefail  # no -u: conda's activate scripts read unset variables
cd "$(dirname "$0")/.."
export TZ=Australia/Melbourne
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate novphy
source env.sh
mkdir -p .local-artifacts/gpu-window
exec python scripts/gpu_window.py tick >> .local-artifacts/gpu-window/cron.log 2>&1
