#!/usr/bin/env bash
set -euo pipefail
root="$(dirname "$0")"
self="$root/9001.x86_64"
binary="$root/9001-player.x86_64"
wrapper="$root/.9001-wrapper"
mv "$self" "$wrapper"
mv "$binary" "$self"
restore() {
  mv "$self" "$binary"
  mv "$wrapper" "$self"
}
trap restore EXIT
physics_args=()
if [[ -n "${NOVPHY_PHYSICS_CAPTURE_PORT:-}" ]]; then
  [[ "$NOVPHY_PHYSICS_CAPTURE_PORT" =~ ^[0-9]+$ ]] \
    && (( NOVPHY_PHYSICS_CAPTURE_PORT >= 1 && NOVPHY_PHYSICS_CAPTURE_PORT <= 65535 )) \
    || { echo "NOVPHY_PHYSICS_CAPTURE_PORT must be a TCP port" >&2; exit 2; }
  physics_args=(--physics-port "$NOVPHY_PHYSICS_CAPTURE_PORT")
fi
"$self" -force-glcore -screen-fullscreen 0 -screen-width 840 -screen-height 480 "${physics_args[@]}" "$@"
