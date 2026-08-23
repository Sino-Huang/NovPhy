"""Human-guided diagnostic and confirmatory physics-v2 review sessions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from uuid import uuid4

from scripts.cohort_v2_scenarios import (
    write_immutable_cohort_v2_json,
)
from scripts.collect_rollouts import action_to_shot, normalize_action_to_game
from scripts.physics_capture_v2 import (
    bind_physics_capture_v2_engine,
    normalized_initial_engine_state_identity,
)
from scripts.physics_capture_v2_persistence import (
    persist_physics_capture_v2,
    validate_physics_capture_v2_artifact,
)
from scripts.slingshot_readiness import PreparedScreenShot


REVIEW_GOALS = ("collision", "persistent support", "support change")
_GOAL_SCENARIO_INDEX = {
    "collision": 0,
    "persistent support": 1,
    "support change": 1,
}
REVIEW_GOAL_LEVELS = {
    "collision": 1,
    "persistent support": 2,
    "support change": 2,
}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _identity(namespace: str, value: Mapping[str, Any]) -> str:
    return f"{namespace}:{_canonical_bytes(value).decode('utf-8')}"


def _support_sets(record: Mapping[str, Any]) -> list[set[tuple[str, str]]]:
    return [
        {
            (support["supporter_entity_id"], support["supported_entity_id"])
            for support in sample["supports"]
        }
        for sample in record["fixed_step_samples"]
    ]


def _goal_evidence_facts(goal: str, record: Mapping[str, Any]) -> set[tuple[str, ...]]:
    if goal == "collision":
        contacts_by_step = {
            sample["fixed_step"]: {
                frozenset((contact["entity_a_id"], contact["entity_b_id"]))
                for contact in sample["contacts"]
            }
            for sample in record["fixed_step_samples"]
        }
        return {
            ("collision", *sorted(event["participants"]))
            for event in record["events"]
            if event["event_type"] == "collision"
            and frozenset(event["participants"]) in contacts_by_step.get(event["fixed_step"], set())
        }

    supports = _support_sets(record)
    if goal == "persistent support":
        persistent = supports[0] & supports[1] if len(supports) >= 2 else set()
        return {("persistent_support", *pair) for pair in persistent}

    changes: set[tuple[str, ...]] = set()
    for previous, current in zip(supports, supports[1:]):
        changes.update(("support_added", *pair) for pair in current - previous)
        changes.update(("support_removed", *pair) for pair in previous - current)
    return changes


def coverage_verdict(goal: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate only the issue-44 observation rule for one guided goal."""
    if goal not in REVIEW_GOALS:
        raise ValueError(f"unknown physics-v2 review goal: {goal}")
    demonstrated = bool(_goal_evidence_facts(goal, record))
    if goal == "collision":
        reason = None if demonstrated else "no collision event matched a same-step raw contact pair"
    elif goal == "persistent support":
        reason = None if demonstrated else "the first two fixed-step samples share no support relation"
    else:
        reason = None if demonstrated else "no adjacent fixed-step samples change the support set"
    return {
        "goal": goal,
        "status": "demonstrated" if demonstrated else "unavailable",
        "demonstrated": demonstrated,
        "reason": reason,
    }


def _write_status(path: Path, value: Mapping[str, Any]) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


class PhysicsV2ReviewSession:
    """Own one diagnostic action and at most one confirmatory replay."""

    def __init__(self, output_root: Path, *, probe_plan_path: Path) -> None:
        self.output_root = Path(output_root)
        self.probe_plan_path = Path(probe_plan_path)
        self.session_id = f"physics-v2-review-{uuid4().hex}"
        self.root = self.output_root / self.session_id
        self.state = "idle"
        self.goal: str | None = None
        self.action: dict[str, Any] | None = None
        self.socket_command: dict[str, int] | None = None
        self.authority: dict[str, Any] | None = None
        self.exploration_verdict: dict[str, Any] | None = None
        self.replay_verdict: dict[str, Any] | None = None
        self._initial_engine_state_identity: str | None = None
        self._diagnostic_goal_evidence: set[tuple[str, ...]] | None = None
        self._plan_path: Path | None = None
        self._plan_bytes: bytes | None = None
        self._replay_attempts = 0
        self.readiness: dict[str, Any] = {}
        self.root.mkdir(parents=True, exist_ok=False)
        self._write_session_status()

    def _probe_plan(self) -> Mapping[str, Any]:
        value = json.loads(self.probe_plan_path.read_text(encoding="utf-8"))
        if value.get("identity") is None or not isinstance(value.get("scenarios"), list):
            raise ValueError("physics-v2 probe plan is malformed")
        return value

    def _authority_for(self, goal: str) -> dict[str, Any]:
        plan = self._probe_plan()
        try:
            scenario = plan["scenarios"][_GOAL_SCENARIO_INDEX[goal]]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(f"probe plan has no scenario authority for {goal}") from error
        manifest_reference = Path(scenario["scenario_manifest_reference"])
        if not manifest_reference.is_absolute():
            repository_reference = Path.cwd() / manifest_reference
            packaged_reference = (
                self.probe_plan_path.parent / "review-manifests" / manifest_reference.name
            )
            manifest_reference = (
                packaged_reference if packaged_reference.is_file() else repository_reference
            )
        manifest = json.loads(manifest_reference.read_text(encoding="utf-8"))
        return {
            "source_probe_plan_identity": plan["identity"],
            "scenario_id": scenario["scenario_id"],
            "scenario_manifest_identity": manifest["identity"],
            "scenario_template_id": scenario["scenario_template_identity"],
            "level_instance_id": scenario["level_instance_identity"],
            "scenario_lineage_id": scenario["scenario_lineage_identity"],
            "review_level": REVIEW_GOAL_LEVELS[goal],
        }

    def _source_bindings(self, *, replay: bool) -> dict[str, str]:
        assert self.authority is not None
        suffix = "confirmatory" if replay else "diagnostic"
        intervention = (
            json.loads(self._plan_bytes)["intervention_id"]
            if replay and self._plan_bytes is not None
            else f"diagnostic-intervention:{self.session_id}"
        )
        return {
            "scenario_template_id": self.authority["scenario_template_id"],
            "level_instance_id": self.authority["level_instance_id"],
            "scenario_lineage_id": self.authority["scenario_lineage_id"],
            "rollout_id": f"{self.session_id}:{suffix}",
            "intervention_id": intervention,
        }

    def stage(self, goal: str, action: Mapping[str, Any]) -> dict[str, Any]:
        if self.state != "idle":
            raise ValueError("review action can only be staged from idle")
        if goal not in REVIEW_GOALS:
            raise ValueError(f"unknown physics-v2 review goal: {goal}")
        normalized = normalize_action_to_game(dict(action))
        frame_height = int(action.get("frame_height", 480))
        shot = action_to_shot(dict(action), frame_height=frame_height)
        self.goal = goal
        self.action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": list(normalized["drag_start"]),
            "drag_release": list(normalized["drag_release"]),
            "tapTime": int(normalized["tapTime"]),
            "releaseTime": int(normalized["releaseTime"]),
            "frame_height": frame_height,
        }
        self.socket_command = {
            "x": shot["x"],
            "y": shot["y"],
            "tapTime": shot["tapTime"],
            "releaseTime": shot["releaseTime"],
        }
        self.action["socket_command"] = dict(self.socket_command)
        self.authority = self._authority_for(goal)
        self.state = "staged"
        self._write_session_status()
        return self.snapshot()

    def begin_exploration(self) -> dict[str, dict[str, int]]:
        if self.state != "staged" or self.socket_command is None:
            raise ValueError("diagnostic exploration requires one staged action")
        self.state = "exploring"
        self._write_session_status()
        return {"socket_command": dict(self.socket_command)}

    def bind_prepared_exploration(self, prepared: PreparedScreenShot) -> None:
        if self.state != "exploring":
            raise ValueError("prepared exploration can only bind while exploring")
        self.action = dict(prepared.action)
        self.socket_command = {
            key: int(prepared.socket_command[key])
            for key in ("x", "y", "tapTime", "releaseTime")
        }
        self.action["socket_command"] = dict(self.socket_command)
        self.readiness["exploration"] = dict(prepared.evidence)
        self._write_session_status()

    def bind_prepared_replay(self, prepared: PreparedScreenShot) -> None:
        if self.state != "replaying":
            raise ValueError("prepared replay can only bind while replaying")
        self.readiness["replay"] = dict(prepared.evidence)
        self._write_session_status()

    def complete_exploration(self, engine_record: Mapping[str, Any]) -> dict[str, Any]:
        if self.state != "exploring" or self.goal is None:
            raise ValueError("no diagnostic exploration is active")
        capture = bind_physics_capture_v2_engine(engine_record, self._source_bindings(replay=False))
        diagnostic = self.root / "diagnostic"
        write_immutable_cohort_v2_json(_json_value(engine_record), diagnostic / "engine-envelope.json")
        self.exploration_verdict = coverage_verdict(self.goal, capture.record)
        write_immutable_cohort_v2_json(self.exploration_verdict, diagnostic / "verdict.json")
        self._initial_engine_state_identity = normalized_initial_engine_state_identity(capture)
        self._diagnostic_goal_evidence = _goal_evidence_facts(self.goal, capture.record)
        self.state = "explored"
        self._write_session_status()
        return self.snapshot()

    def freeze_replay(self) -> dict[str, Any]:
        if self.state != "explored" or self.action is None or self.goal is None or self.authority is None:
            raise ValueError("confirmatory replay can only be frozen after diagnostic exploration")
        if self.exploration_verdict is None or not self.exploration_verdict["demonstrated"]:
            raise ValueError("confirmatory replay requires a demonstrated diagnostic pilot")
        diagnostic_path = self.root / "diagnostic" / "engine-envelope.json"
        payload = {
            "schema": "physics_v2_manual_replay_plan_v1",
            "goal": self.goal,
            "source_probe_plan_identity": self.authority["source_probe_plan_identity"],
            "scenario_id": self.authority["scenario_id"],
            "scenario_manifest_identity": self.authority["scenario_manifest_identity"],
            "scenario_template_id": self.authority["scenario_template_id"],
            "level_instance_id": self.authority["level_instance_id"],
            "scenario_lineage_id": self.authority["scenario_lineage_id"],
            "action": self.action,
            "expected_initial_engine_state_identity": self._initial_engine_state_identity,
            "selection_provenance": {
                "kind": "diagnostic_pilot",
                "capture_path": diagnostic_path.relative_to(self.root).as_posix(),
            },
            "diagnostic_capture_eligible": False,
            "max_attempts": 1,
        }
        intervention_payload = {
            "goal": self.goal,
            "scenario_lineage_id": self.authority["scenario_lineage_id"],
            "action": self.action,
        }
        payload["intervention_id"] = _identity("physics-v2-manual-intervention-v1", intervention_payload)
        plan = {**payload, "identity": _identity("physics-v2-manual-replay-plan-v1", payload)}
        self._plan_path = self.root / "replay-plan.json"
        write_immutable_cohort_v2_json(plan, self._plan_path)
        self._plan_bytes = self._plan_path.read_bytes()
        self.state = "frozen"
        self._write_session_status()
        result = self.snapshot()
        result["replay_plan_path"] = str(self._plan_path)
        return result

    def begin_replay(self) -> dict[str, dict[str, int]]:
        if self._replay_attempts >= 1:
            raise ValueError("review session permits one confirmatory replay")
        if self.state != "frozen" or self.socket_command is None or self._plan_path is None:
            raise ValueError("confirmatory replay requires a frozen plan")
        if self._plan_path.read_bytes() != self._plan_bytes:
            raise ValueError("frozen replay plan bytes changed")
        self._replay_attempts += 1
        self.state = "replaying"
        self._write_session_status()
        return {"socket_command": dict(self.socket_command)}

    def complete_replay(self, engine_record: Mapping[str, Any]) -> dict[str, Any]:
        if self.state != "replaying" or self.goal is None or self.authority is None:
            raise ValueError("no confirmatory replay is active")
        staging = self.root / ".confirmatory-staging"
        metadata = persist_physics_capture_v2(
            staging,
            engine_record,
            source_bindings=self._source_bindings(replay=True),
            scenario_manifest_identity=self.authority["scenario_manifest_identity"],
        )
        capture = validate_physics_capture_v2_artifact(staging, metadata)
        observed_initial = normalized_initial_engine_state_identity(capture)
        initial_matches = observed_initial == self._initial_engine_state_identity
        self.replay_verdict = coverage_verdict(self.goal, capture.record)
        replay_goal_evidence = _goal_evidence_facts(self.goal, capture.record)
        goal_evidence_matches = bool(
            self._diagnostic_goal_evidence
            and self._diagnostic_goal_evidence & replay_goal_evidence
        )
        self.replay_verdict["initial_engine_state_matches_diagnostic"] = initial_matches
        self.replay_verdict["goal_evidence_matches_diagnostic"] = goal_evidence_matches
        demonstrated = bool(
            self.replay_verdict["demonstrated"]
            and initial_matches
            and goal_evidence_matches
        )
        if not initial_matches:
            self.replay_verdict["status"] = "failed"
            self.replay_verdict["reason"] = "confirmatory initial engine state differs from the diagnostic pilot"
        elif self.replay_verdict["demonstrated"] and not goal_evidence_matches:
            self.replay_verdict["status"] = "failed"
            self.replay_verdict["reason"] = "confirmatory goal evidence differs from the diagnostic pilot"
        metadata["review_goal"] = self.goal
        metadata["replay_plan_identity"] = json.loads(self._plan_bytes)["identity"]
        metadata["eligible_for_issue_44_review"] = demonstrated
        write_immutable_cohort_v2_json(metadata, staging / "metadata.json")
        write_immutable_cohort_v2_json(self.replay_verdict, staging / "verdict.json")
        destination = self.root / ("accepted" if demonstrated else "quarantine")
        os.replace(staging, destination)
        replay_failed = not initial_matches or (
            self.replay_verdict["demonstrated"] and not goal_evidence_matches
        )
        self.state = "demonstrated" if demonstrated else ("failed" if replay_failed else "unavailable")
        self._write_session_status()
        return self.snapshot()

    def fail_active_capture(self, error: Exception) -> dict[str, Any]:
        if self.state not in {"exploring", "replaying"}:
            raise ValueError("no physics-v2 capture is active")
        self.state = "failed"
        write_immutable_cohort_v2_json(
            {"schema": "physics_v2_review_failure_v1", "error": str(error)},
            self.root / "failure.json",
        )
        self._write_session_status()
        return self.snapshot()

    def _write_session_status(self) -> None:
        _write_status(self.root / "session.json", self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state,
            "goal": self.goal,
            "action": self.action,
            "socket_command": self.socket_command,
            "slingshot_readiness": self.readiness,
            "authority": self.authority,
            "verdict": self.replay_verdict or self.exploration_verdict,
            "diagnostic_capture_eligible": False,
            "eligible_for_issue_44": False,
            "eligible_for_issue_44_review": self.state == "demonstrated",
            "replay_attempts": self._replay_attempts,
        }

    def fixed_steps(self, *, start: int = 0, count: int = 100) -> dict[str, Any]:
        if start < 0 or count <= 0 or count > 100:
            raise ValueError("fixed-step page requires start >= 0 and 1 <= count <= 100")
        candidates = (
            self.root / "accepted" / "physics_capture_v2.json",
            self.root / "quarantine" / "physics_capture_v2.json",
            self.root / "diagnostic" / "engine-envelope.json",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            return {"start": start, "count": 0, "total": 0, "steps": []}
        record = json.loads(path.read_text(encoding="utf-8"))
        steps = record["fixed_step_samples"]
        selected = []
        for step in steps[start : start + count]:
            value = dict(step)
            value["events"] = [
                event for event in record["events"]
                if event["fixed_step"] == step["fixed_step"]
            ]
            selected.append(value)
        return {"start": start, "count": len(selected), "total": len(steps), "steps": selected}
