# Problems

## 2026-07-05 Start Work Context
- Main bug: collector accepts static/menu/non-gameplay rollout artifacts as if valid; `shoot_response=1` and protocol `PLAYING` can disagree with visual evidence.
- Potential root causes to preserve: missing same-engine re-prepare, protocol state drift, capture/window/focus mismatch, and terminal/menu transition after previous shot.
