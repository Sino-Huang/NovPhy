#!/usr/bin/env bash
# Copy a Unity Editor log to a published location with licensing identity removed.
#
# Unity writes its Licensing Client channel, the assigned serial number, and
# LICENSE SYSTEM entries into every build log. Those lines carry an account
# identifier and a partially masked serial, so they must not reach the promotion
# stage. Each matching line is replaced in place, preserving line count and
# ordering so published logs stay diffable and free of per-run license timestamps.
set -euo pipefail

source_log="${1:?usage: redact_unity_log.sh <source-log> <destination-log>}"
destination_log="${2:?usage: redact_unity_log.sh <source-log> <destination-log>}"

marker='[redacted: Unity licensing line withheld from published provenance]'

destination_temporary="$(mktemp "$(dirname "$destination_log")/.$(basename "$destination_log").XXXXXX")"
trap 'rm -f "$destination_temporary"' EXIT

sed -E \
  -e "s@^.*\[Licensing::Module\].*\$@${marker}@" \
  -e "s@^.*\[LicensingClient\].*\$@${marker}@" \
  -e "s@^LICENSE SYSTEM .*\$@${marker}@" \
  "$source_log" > "$destination_temporary"

if grep -qE '\[Licensing::Module\]|\[LicensingClient\]|^LICENSE SYSTEM ' "$destination_temporary"; then
  echo "refusing to publish Unity log: licensing lines survived redaction" >&2
  exit 3
fi

chmod 0644 "$destination_temporary"
mv "$destination_temporary" "$destination_log"
trap - EXIT
