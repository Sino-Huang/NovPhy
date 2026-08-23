#!/bin/bash
# sudo kill -9 $(sudo lsof -t -i :8767)

physics_review_args=()
for novphy_webui_arg in "$@"; do
  if [[ "$novphy_webui_arg" == "--physics-v2-review" || "$novphy_webui_arg" == "--issue-53-review-root" ]]; then
    physics_review_args=(--speed 1)
    break
  fi
done

python3 -m src.webui.server --host 127.0.0.1 --port 8767 "${physics_review_args[@]}" "$@"
