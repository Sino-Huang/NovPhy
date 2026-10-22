#!/usr/bin/env bash
# cron entry point of the night-window GPU scheduler (docs/gpu-window-scheduler.md). Installed crontab:
#   TZ=Australia/Melbourne
#   */5 * * * * /mnt/array/sukaih/Project/NovPhy/scripts/gpu_window_cron.sh
# Jobs inherit this environment: the novphy env and env.sh, as in the documented manual launches.
set -eo pipefail  # no -u: conda's activate scripts read unset variables
cd "$(dirname "$0")/.."
export TZ=Australia/Melbourne
# cron starts with a 1024 open-file soft limit; interactive shells here run at the hard limit (1048576).
# PyTorch shares tensors returned by worker processes as file descriptors, so a 1024 limit breaks process pools.
# The first #123 --score attempt (2026-10-21 06:25) died that way in build_eval_cache before any score.
ulimit -Sn "$(ulimit -Hn)"
# Fatal signals (e.g. SIGSEGV in native code) then dump a Python stack into the job log. The first #125 --publish
# (2026-10-23 01:15) died with exit -11 after 25 s and left no traceback.
export PYTHONFAULTHANDLER=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate novphy
source env.sh
mkdir -p .local-artifacts/gpu-window
exec python scripts/gpu_window.py tick >> .local-artifacts/gpu-window/cron.log 2>&1
