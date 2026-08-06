#!/usr/bin/env bash
set -uo pipefail

evidence=.omo/evidence/world-model-physics-instrumentation/final-wave-exact-final-6668b12-direct
expected_head=6668b12f43f2c577c7f2446c98aedea0811f913e
release=sciencebirdsgames/physics-v1
result=0

check() {
  local name="$1"
  shift
  if "$@"; then
    printf 'PASS %s\n' "$name"
  else
    printf 'FAIL %s\n' "$name"
    result=1
  fi
}

check exact_head test "$(git rev-parse HEAD)" = "$expected_head"
check product_paths_clean git diff --quiet HEAD -- scripts src tasks/task_template_designer/Assets tests
check direct_preflight_failed test "$(cat "$evidence/direct-preflight.exit")" = 1
check guard_observable rg -q 'PackagingError: untracked product source: !! tasks/task_template_designer/Packages/manifest.json' "$evidence/direct-preflight.stderr"
check build_log_guard_observable rg -q 'PackagingError: untracked product source: !! tasks/task_template_designer/Packages/manifest.json' "$evidence/build-1.log"
check build_exit_nonzero test "$(cat "$evidence/build-1.exit")" = 1
check stage_1_empty test -z "$(find "$evidence/build-1-stage" -mindepth 1 -print -quit)"
check stage_2_empty test -z "$(find "$evidence/build-2-stage" -mindepth 1 -print -quit)"
check probe_stage_absent test ! -e "$evidence/preflight-probe-stage"
check release_archive_present test -s "$release/novphy-physics-player-2019.4.41f2.tar.gz"
check release_receipt_present test -s "$release/archive.sha256"
check claim_valid python -m json.tool "$evidence/DoneClaim.json"
check cleanup_valid python -m json.tool "$evidence/cleanup-receipt.json"
check claim_needs_fix test "$(python -c 'import json; print(json.load(open(".omo/evidence/world-model-physics-instrumentation/final-wave-exact-final-6668b12-direct/DoneClaim.json"))["status"])')" = needs-fix
check publication_not_performed test "$(python -c 'import json; print(str(json.load(open(".omo/evidence/world-model-physics-instrumentation/final-wave-exact-final-6668b12-direct/DoneClaim.json"))["publication"]["performed"]).lower())')" = false

exit "$result"
