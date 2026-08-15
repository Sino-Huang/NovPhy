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
exec "$self" -force-glcore -screen-fullscreen 0 -screen-width 840 -screen-height 480 "$@"
