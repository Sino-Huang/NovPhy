"""Read-only human review of issue #53 production termination mismatches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping

from scripts.capture_issue_53_evidence import (
    STAGE_ROOT,
    _attempt_options,
    _install_level,
)
from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest
from scripts.cohort_v2_release import (
    compare_production_replay,
    validate_issue_53_execution_report,
)
from scripts.cohort_v2_replay import semantic_identity
from scripts.cohort_v2_scenarios import (
    CohortV2ScenarioManifest,
    write_immutable_cohort_v2_bytes,
    write_immutable_cohort_v2_json,
)
from scripts.collect_rollouts import (
    action_to_shot,
    collect_fresh_engine_attempt,
    collect_fresh_engine_rollouts,
)
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    audit_final_evaluation_workflow_access,
)
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.physics_capture_v2_persistence import (
    persist_physics_capture_v2,
    validate_physics_capture_v2_artifact,
)
from scripts.smoke_physics_capture import archive_details, start_display, terminate
from scripts.verify_physics_player import verify_physics_player_archive
from src.webui.bridge import PhysicsCaptureV2Failure


EXECUTION_REPORT = "production-execution-report.json"
QUALITY_REPORT = "production-quality-report.json"
REVIEW_SCHEMA = "issue_53_human_review_v2"
DECISIONS = frozenset(("confirmed_mismatch", "suspicious_termination_export", "uncertain"))
FINAL_ROLE = "final_evaluation"
EXPECTED_MISMATCH_COUNT = 8
VIDEO_FPS = 5
SCREEN_COORDINATE_ALIGNMENT_CONTRACT = {
    "schema": "screen_coordinate_alignment_v1",
    "startup_speed": 50,
    "stable_observations_required_per_phase": 2,
    "poll_interval_seconds": 0.05,
    "timeout_seconds": 15.0,
    "retained_anchor_tolerance_pixels": 2.0,
    "frozen_socket_command_reanchored": False,
}


class Issue53ReviewError(ValueError):
    """The frozen production evidence or review transition is invalid."""


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Issue53ReviewError(f"Cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise Issue53ReviewError(f"{label} must be a JSON object")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _slug(value: str) -> str:
    return value.replace(":", "_").replace("%", "-")


def _scenario_family(benchmark_condition: str) -> str:
    parts = benchmark_condition.split(":")
    return "/".join(parts[-2:]) if len(parts) >= 2 else benchmark_condition


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _encode_browser_video(frames: Path, output: Path) -> None:
    """Encode retained RGB PNG frames as Chrome-compatible VP8/WebM."""
    with tempfile.TemporaryDirectory(prefix="novphy-issue53-video-") as temporary:
        encode_root = Path(temporary)
        for frame in sorted(Path(frames).glob("frame_*.png")):
            shutil.copyfile(frame, encode_root / frame.name)
        temporary_video = encode_root / "replay.webm"
        command = [
            "gst-launch-1.0",
            "-q",
            "multifilesrc",
            f"location={encode_root / 'frame_%06d.png'}",
            "index=0",
            f"caps=image/png,framerate={VIDEO_FPS}/1",
            "!",
            "pngdec",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=I420",
            "!",
            "vp8enc",
            "deadline=1",
            "!",
            "webmmux",
            "!",
            "filesink",
            f"location={temporary_video}",
        ]
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temporary_video, output)


@dataclass(frozen=True, slots=True)
class ReviewItem:
    index: int
    entry: Mapping[str, Any]

    @property
    def attempt_id(self) -> str:
        return str(self.entry["attempt_id"])

    @property
    def final(self) -> bool:
        return self.entry["exposure_role"] == FINAL_ROLE


ReplayRunner = Callable[[ReviewItem, Path, int], Mapping[str, Any]]


class Issue53ReviewSession:
    """Own the separate issue-53 review bundle without mutating production evidence."""

    def __init__(
        self,
        production_root: Path,
        output_root: Path,
        *,
        repository_root: Path | None = None,
        speed: int = 1,
        replay_runner: ReplayRunner | None = None,
    ) -> None:
        self.production_root = Path(production_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.repository_root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.speed = int(speed)
        self._replay_runner = replay_runner or self._run_diagnostic_replay
        self.execution_path = self.production_root / EXECUTION_REPORT
        local_quality = self.production_root / QUALITY_REPORT
        published_quality = (
            self.repository_root
            / "data/runtime_evidence/issue-53/production-quality-report.json"
        )
        self.quality_path = local_quality if local_quality.is_file() else published_quality
        self.collection_path = self.production_root / "frozen-collection-plan.json"
        self.parameter_path = (
            self.production_root / "frozen-production-parameter-plan.json"
        )
        self._source_bytes = {
            path: path.read_bytes()
            for path in (
                self.execution_path,
                self.quality_path,
                self.collection_path,
                self.parameter_path,
            )
        }
        collection = _load_object(self.collection_path, "frozen collection plan")
        execution = validate_issue_53_execution_report(
            _load_object(self.execution_path, "production execution report"),
            collection,
        )
        quality = _load_object(self.quality_path, "production quality report")
        self._validate_quality(quality, execution)
        mismatches = [
            entry
            for entry in execution["attempt_ledger"]
            if entry.get("status") == "accepted"
            and entry.get("terminal_reason") != entry.get("expected_termination")
        ]
        if len(mismatches) != EXPECTED_MISMATCH_COUNT:
            raise Issue53ReviewError(
                f"Issue-53 review requires exactly {EXPECTED_MISMATCH_COUNT} termination mismatches"
            )
        self.items = tuple(
            ReviewItem(index, dict(entry)) for index, entry in enumerate(mismatches)
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._authorized = False
        self._restore_final_access()
        self._write_session_manifest()

    def _validate_quality(
        self, quality: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> None:
        if quality.get("schema") != "cohort_v2_production_quality_report_v1":
            raise Issue53ReviewError("Production quality report schema is unsupported")
        if any(
            quality.get(field) != execution.get(field)
            for field in (
                "collection_plan_identity",
                "production_parameter_plan_identity",
            )
        ):
            raise Issue53ReviewError("Production quality report is not bound to the execution report")
        ledger_mismatches = {
            entry["attempt_id"]: (
                entry["expected_termination"],
                entry.get("terminal_reason"),
            )
            for entry in execution["attempt_ledger"]
            if entry.get("status") == "accepted"
            and entry.get("terminal_reason") != entry.get("expected_termination")
        }
        quality_mismatches = quality.get("termination_mismatches")
        if not isinstance(quality_mismatches, list):
            raise Issue53ReviewError("Production quality mismatches are malformed")
        for mismatch in quality_mismatches:
            if not isinstance(mismatch, Mapping):
                raise Issue53ReviewError("Production quality mismatch is malformed")
            attempt_id = mismatch.get("attempt_id")
            if ledger_mismatches.get(attempt_id) != (
                mismatch.get("expected"),
                mismatch.get("observed"),
            ):
                raise Issue53ReviewError("Production quality mismatch differs from execution evidence")
        scope = quality.get("outcome_scope", "all_exposure_roles")
        expected_ids = {
            attempt_id
            for attempt_id in ledger_mismatches
            if scope != "non_final_only"
            or next(
                entry
                for entry in execution["attempt_ledger"]
                if entry["attempt_id"] == attempt_id
            )["exposure_role"]
            != FINAL_ROLE
        }
        if {item.get("attempt_id") for item in quality_mismatches} != expected_ids:
            raise Issue53ReviewError("Production quality report omits a mismatch in its declared scope")

    def _assert_sources_unchanged(self) -> None:
        for path, content in self._source_bytes.items():
            if path.read_bytes() != content:
                raise Issue53ReviewError(f"Frozen production source changed during review: {path}")

    def _write_session_manifest(self) -> None:
        value = {
            "schema": REVIEW_SCHEMA,
            "production_root": str(self.production_root),
            "execution_report": str(self.execution_path),
            "quality_report": str(self.quality_path),
            "mismatch_count": len(self.items),
            "non_final_item_count": sum(not item.final for item in self.items),
            "sealed_final_item_count": sum(item.final for item in self.items),
            "production_evidence_mutable": False,
            "diagnostic_replays_release_eligible": False,
            "screen_coordinate_alignment_contract": SCREEN_COORDINATE_ALIGNMENT_CONTRACT,
        }
        write_immutable_cohort_v2_json(value, self.output_root / "review-session.json")

    def _item(self, index: int) -> ReviewItem:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.items):
            raise Issue53ReviewError("Unknown issue-53 review item")
        item = self.items[index]
        if item.final and not self._authorized:
            raise PermissionError("Final-evaluation review requires validated authorization")
        return item

    def _artifact(self, item: ReviewItem) -> Path:
        value = item.entry.get("artifact_path")
        if not isinstance(value, str) or not value:
            raise Issue53ReviewError("Mismatch has no retained artifact path")
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (self.production_root / path).resolve()
        if self.production_root != path and self.production_root not in path.parents:
            raise Issue53ReviewError("Retained artifact path escapes the production root")
        if not (path / "physics_capture_v2.json").is_file():
            raise Issue53ReviewError("Retained physics_capture_v2 trace is missing")
        return path

    def _item_output(self, item: ReviewItem) -> Path:
        base = self.output_root / ("sealed/items" if item.final else "items")
        return base / _slug(item.attempt_id)

    def _action_and_socket(self, item: ReviewItem) -> tuple[dict[str, Any], dict[str, Any]]:
        artifact = self._artifact(item)
        manifest = _load_object(artifact.parent / "manifest.json", "retained rollout manifest")
        action_log = _load_object(artifact.parent / "action_log.json", "retained action log")
        try:
            action = dict(manifest["rollouts"][0]["action"])
            socket_command = dict(action_log["accepted_trials"][0]["shot"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise Issue53ReviewError("Retained action evidence is malformed") from error
        calculated = action_to_shot(
            action, frame_height=int(action.get("frame_height", 480))
        )
        if calculated != socket_command:
            raise Issue53ReviewError("Retained anchored action and socket command disagree")
        reference = action.get("slingshot_reference")
        if (
            not isinstance(reference, Mapping)
            or not all(key in reference for key in ("canvasX", "canvasY"))
        ):
            raise Issue53ReviewError(
                "Retained production action has no screen-coordinate anchor"
            )
        return action, socket_command

    def _summary(self, item: ReviewItem) -> dict[str, Any]:
        action, socket_command = self._action_and_socket(item)
        decision = self._load_decision(item)
        replay = self._load_replay_result(item)
        return {
            "index": item.index,
            "locked": False,
            "role": item.entry["exposure_role"],
            "scenarioFamily": _scenario_family(
                str(item.entry["benchmark_condition_identity"])
            ),
            "action": action,
            "socketCommand": socket_command,
            "intendedStratum": item.entry["intended_coverage_stratum"],
            "expectedTermination": item.entry["expected_termination"],
            "observedTermination": item.entry["terminal_reason"],
            "coverageFacts": list(item.entry.get("realized_coverage_strata", [])),
            "traceOpened": self._opened_path(item).is_file(),
            "decision": decision,
            "replay": replay,
        }

    def snapshot(self) -> dict[str, Any]:
        self._assert_sources_unchanged()
        playlist = []
        for item in self.items:
            if item.final and not self._authorized:
                playlist.append({"index": item.index, "locked": True})
            else:
                playlist.append(self._summary(item))
        return {
            "schema": REVIEW_SCHEMA,
            "screenCoordinateAlignmentContract": SCREEN_COORDINATE_ALIGNMENT_CONTRACT,
            "mismatchCount": len(self.items),
            "authorizedFinalAccess": self._authorized,
            "sealedFinalItemCount": sum(item.final for item in self.items),
            "playlist": playlist,
            "reviewProgress": {
                "opened": sum(self._opened_path(item).is_file() for item in self.items),
                "decided": sum(self._decision_path(item).is_file() for item in self.items),
            },
            "followUpDecision": self._follow_up_decision(),
        }

    def authorize_final_access(self, authorization_identity: str) -> dict[str, Any]:
        self._assert_sources_unchanged()
        if self._authorized:
            return self.snapshot()
        manifest_path = (
            self.production_root / "authorities/authorized-final-access-manifest.json"
        )
        manifest = FinalEvaluationWorkflowAccessManifest.from_dict(
            _load_object(manifest_path, "authorized final-access manifest")
        )
        if manifest.authorization_state != "authorized":
            raise PermissionError("Final-evaluation workflow is not authorized")
        if authorization_identity != manifest.authorization_identity:
            raise PermissionError("Final-evaluation authorization identity does not match")
        partition = CohortV2PartitionExposureManifest.from_dict(
            _load_object(
                self.repository_root
                / "data/runtime_evidence/issue-47/partition-exposure-manifest.json",
                "partition exposure manifest",
            )
        )
        final_entry = next(item for item in self.items if item.final)
        artifact = next(
            (
                value
                for value in manifest.authorized_artifacts
                if value.artifact_identity
                == final_entry.entry["scenario_manifest_identity"]
            ),
            None,
        )
        if artifact is None:
            raise PermissionError("Authorized manifest does not cover the final review lineage")
        record = {
            "workflow_identity": manifest.workflow_identity,
            "operator_identity": manifest.operator_identity,
            "artifact_identity": artifact.artifact_identity,
            "source_scenario_lineage_identities": list(
                artifact.source_scenario_lineage_identities
            ),
            "accessed_at": _utc_now(),
            "authorization_identity": authorization_identity,
            "consumer_exposure_role": FINAL_ROLE,
        }
        audit = audit_final_evaluation_workflow_access(
            partition, manifest, observed_accesses=[record]
        )
        access_root = self.output_root / "sealed/access"
        write_immutable_cohort_v2_json(record, access_root / "review-access.json")
        write_immutable_cohort_v2_json(audit, access_root / "review-access-audit.json")
        write_immutable_cohort_v2_json(
            {
                "schema": "issue_53_human_review_access_context_v2",
                "purpose": "human_review_of_retained_production_mismatches",
                "production_evidence_modified": False,
                "review_outputs_sealed": True,
            },
            access_root / "context.json",
        )
        self._authorized = True
        return self.snapshot()

    def _restore_final_access(self) -> None:
        path = self.output_root / "sealed/access/review-access.json"
        if not path.is_file():
            return
        manifest = FinalEvaluationWorkflowAccessManifest.from_dict(
            _load_object(
                self.production_root / "authorities/authorized-final-access-manifest.json",
                "authorized final-access manifest",
            )
        )
        partition = CohortV2PartitionExposureManifest.from_dict(
            _load_object(
                self.repository_root
                / "data/runtime_evidence/issue-47/partition-exposure-manifest.json",
                "partition exposure manifest",
            )
        )
        record = _load_object(path, "review final-access record")
        audit_final_evaluation_workflow_access(
            partition, manifest, observed_accesses=[record]
        )
        self._authorized = True

    def _opened_path(self, item: ReviewItem) -> Path:
        return self._item_output(item) / "trace-opened.json"

    def open_trace(self, index: int) -> dict[str, Any]:
        self._assert_sources_unchanged()
        item = self._item(index)
        capture = load_physics_capture_v2(self._artifact(item) / "physics_capture_v2.json")
        terminal = capture.record["terminal_evidence"]
        value = {
            "schema": "issue_53_retained_trace_opened_v2",
            "attempt_id": item.attempt_id,
            "capture_id": capture.capture_id,
            "opened_at": _utc_now(),
            "source_disposition": "retained_production_evidence_read_only",
            "terminal_evidence": dict(terminal),
        }
        path = self._opened_path(item)
        if not path.exists():
            write_immutable_cohort_v2_json(value, path)
        return self.item_detail(index)

    def item_detail(self, index: int) -> dict[str, Any]:
        self._assert_sources_unchanged()
        item = self._item(index)
        if not self._opened_path(item).is_file():
            raise Issue53ReviewError("Open the retained trace before requesting its evidence")
        capture = load_physics_capture_v2(self._artifact(item) / "physics_capture_v2.json")
        record = capture.record
        events = [dict(event) for event in record["events"]]
        return {
            "item": self._summary(item),
            "captureId": capture.capture_id,
            "fixedStepCount": len(record["fixed_step_samples"]),
            "captureStride": capture.configured_fixed_step_capture_stride,
            "preInterventionFixedStep": record["pre_intervention_fixed_step"],
            "terminalEvidence": dict(record["terminal_evidence"]),
            "events": events,
            "terminationExplanation": self.termination_explanation(
                str(record["terminal_evidence"]["reason"])
            ),
        }

    @staticmethod
    def termination_explanation(reason: str) -> str:
        return {
            "stable_entered": (
                "Movement stopped enough to end the one-shot rollout; this does not mean the level was won."
            ),
            "level_clear": "All pigs were removed and the level-clear condition fired.",
            "level_fail": "The game fail condition fired before the level was cleared.",
            "rollout_ceiling": "The frozen fixed-step rollout ceiling ended the capture.",
        }.get(reason, "Unknown retained terminal event.")

    def fixed_steps(self, index: int, *, start: int, count: int) -> dict[str, Any]:
        self._assert_sources_unchanged()
        item = self._item(index)
        if not self._opened_path(item).is_file():
            raise Issue53ReviewError("Open the retained trace before fixed-step playback")
        if start < 0 or count <= 0 or count > 120:
            raise Issue53ReviewError("Fixed-step page requires start >= 0 and 1 <= count <= 120")
        capture = load_physics_capture_v2(self._artifact(item) / "physics_capture_v2.json")
        record = capture.record
        events_by_step: dict[int, list[Mapping[str, Any]]] = {}
        for event in record["events"]:
            events_by_step.setdefault(int(event["fixed_step"]), []).append(event)
        steps = record["fixed_step_samples"]
        selected = [
            self._playback_step(step, events_by_step.get(int(step["fixed_step"]), []))
            for step in steps[start : start + count]
        ]
        return {
            "start": start,
            "count": len(selected),
            "total": len(steps),
            "steps": selected,
        }

    @staticmethod
    def _playback_step(
        step: Mapping[str, Any], events: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        entities = []
        for entity in step["entities"]:
            body = entity.get("body")
            entities.append(
                {
                    "id": entity["entity_id"],
                    "kind": str(entity["entity_id"]).split(":")[1],
                    "lifecycle": entity["lifecycle"],
                    "bodyPresent": entity["body_present"],
                    "position": None if body is None else list(body["position"]),
                    "rotation": None if body is None else body["rotation_degrees"],
                }
            )
        active = [
            entity
            for entity in entities
            if entity["lifecycle"] == "active"
        ]
        return {
            "fixedStep": step["fixed_step"],
            "entities": entities,
            "contacts": [
                {
                    "a": contact["entity_a_id"],
                    "b": contact["entity_b_id"],
                    "point": list(contact["point"]),
                    "separation": contact["separation"],
                }
                for contact in step["contacts"]
            ],
            "supports": [
                {
                    "supporter": support["supporter_entity_id"],
                    "supported": support["supported_entity_id"],
                }
                for support in step["supports"]
            ],
            "events": [dict(event) for event in events],
            "pigsRemaining": sum(entity["kind"] == "pig" for entity in active),
            "birdsRemaining": sum(entity["kind"] == "bird" for entity in active),
        }

    def _decision_path(self, item: ReviewItem) -> Path:
        return self._item_output(item) / "decision.json"

    def _load_decision(self, item: ReviewItem) -> dict[str, Any] | None:
        path = self._decision_path(item)
        return _load_object(path, "review decision") if path.is_file() else None

    def record_decision(
        self, index: int, *, decision: str, notes: str, reviewer: str
    ) -> dict[str, Any]:
        self._assert_sources_unchanged()
        item = self._item(index)
        if not self._opened_path(item).is_file():
            raise Issue53ReviewError("Open the retained trace before recording a decision")
        if decision not in DECISIONS:
            raise Issue53ReviewError("Unknown reviewer decision")
        if not isinstance(notes, str) or not isinstance(reviewer, str) or not reviewer.strip():
            raise Issue53ReviewError("Reviewer identity is required and notes must be text")
        path = self._decision_path(item)
        if path.exists():
            raise Issue53ReviewError("The immutable decision for this item already exists")
        replay = self._load_replay_result(item)
        value = {
            "schema": "issue_53_human_review_decision_v2",
            "attempt_id": item.attempt_id,
            "exposure_role": item.entry["exposure_role"],
            "decision": decision,
            "notes": notes,
            "reviewer": reviewer.strip(),
            "recorded_at": _utc_now(),
            "retained_trace_capture_id": load_physics_capture_v2(
                self._artifact(item) / "physics_capture_v2.json"
            ).capture_id,
            "diagnostic_replay_considered": replay is not None,
            "production_evidence_modified": False,
        }
        write_immutable_cohort_v2_json(value, path)
        self._finalize_if_complete()
        return self.snapshot()

    def _replay_root(self, item: ReviewItem) -> Path:
        return self._item_output(item) / "diagnostic-replay"

    def _load_replay_result(self, item: ReviewItem) -> dict[str, Any] | None:
        path = self._replay_root(item) / "result.json"
        return _load_object(path, "diagnostic replay result") if path.is_file() else None

    def run_replay(self, index: int) -> dict[str, Any]:
        self._assert_sources_unchanged()
        item = self._item(index)
        if not self._opened_path(item).is_file():
            raise Issue53ReviewError("Open the retained trace before diagnostic replay")
        if self._decision_path(item).exists():
            raise Issue53ReviewError("Diagnostic replay must occur before the immutable decision")
        root = self._replay_root(item)
        attempt_path = root / "attempt.json"
        if attempt_path.exists():
            raise Issue53ReviewError("This item has already used its one diagnostic replay")
        write_immutable_cohort_v2_json(
            {
                "schema": "issue_53_diagnostic_replay_attempt_v2",
                "attempt_id": item.attempt_id,
                "started_at": _utc_now(),
                "max_attempts": 1,
                "disposition": "diagnostic_only",
            },
            attempt_path,
        )
        try:
            result = dict(self._replay_runner(item, root, self.speed))
        except Exception as error:
            write_immutable_cohort_v2_json(
                {
                    "schema": "issue_53_diagnostic_replay_failure_v2",
                    "attempt_id": item.attempt_id,
                    "error": str(error),
                    "diagnostic_only": True,
                },
                root / "failure.json",
            )
            raise
        alignment = result.get("screen_coordinate_alignment")
        comparison = result.get("comparison")
        components = comparison.get("components") if isinstance(comparison, Mapping) else None
        if (
            not isinstance(alignment, Mapping)
            or not isinstance(components, list)
            or not any(
                isinstance(component, Mapping)
                and component.get("component") == "screen_coordinate_alignment"
                and component.get("status") == "equality"
                for component in components
            )
        ):
            raise Issue53ReviewError(
                "Diagnostic replay result requires matching screen-coordinate alignment evidence"
            )
        result.update(
            {
                "schema": "issue_53_diagnostic_replay_result_v2",
                "attempt_id": item.attempt_id,
                "diagnostic_only": True,
                "cohort_quota_eligible": False,
                "production_accounting_eligible": False,
                "resampling_eligible": False,
                "release_eligible": False,
            }
        )
        write_immutable_cohort_v2_json(result, root / "result.json")
        return self.item_detail(index)

    def replay_video(self, index: int) -> Path:
        item = self._item(index)
        result = self._load_replay_result(item)
        if result is None:
            raise Issue53ReviewError("No diagnostic replay video exists for this item")
        path = Path(str(result.get("video_path", ""))).resolve()
        webm_path = path.with_suffix(".webm")
        if webm_path.is_file():
            path = webm_path
        root = self._replay_root(item).resolve()
        if root not in path.parents or not path.is_file():
            raise Issue53ReviewError("Diagnostic replay video path is invalid")
        return path

    def _authority_paths(self, item: ReviewItem) -> tuple[Path, Path]:
        role_name = str(item.entry["exposure_role"]).replace("_", "-")
        return (
            self.production_root / f"authorities/manifests/{role_name}.json",
            self.production_root / f"authorities/xml/{role_name}.xml",
        )

    def _run_diagnostic_replay(
        self, item: ReviewItem, root: Path, speed: int
    ) -> Mapping[str, Any]:
        original_artifact = self._artifact(item)
        action, socket_command = self._action_and_socket(item)
        manifest_path, xml_path = self._authority_paths(item)
        manifest_value = _load_object(manifest_path, "production scenario manifest")
        scenario = CohortV2ScenarioManifest.from_dict(manifest_value)
        production_player = _load_object(
            self.production_root / "player-provenance.json", "production player provenance"
        )
        current_player = verify_physics_player_archive(STAGE_ROOT, physics_v2=True)
        envelope_fields = (
            "capture_schema",
            "declared_file_count",
            "protocol_version",
            "source_snapshot_commit",
            "source_tree",
            "unity_version",
        )
        if any(production_player.get(key) != current_player.get(key) for key in envelope_fields):
            raise Issue53ReviewError("Available physics player differs from the production envelope")
        inputs = root / "inputs"
        write_immutable_cohort_v2_bytes(manifest_path.read_bytes(), inputs / "scenario-manifest.json")
        write_immutable_cohort_v2_bytes(xml_path.read_bytes(), inputs / "scenario.xml")
        write_immutable_cohort_v2_bytes(
            (self.production_root / "player-provenance.json").read_bytes(),
            inputs / "player-provenance.json",
        )
        write_immutable_cohort_v2_json(
            {
                "schema": "issue_53_diagnostic_replay_input_v2",
                "original_attempt_id": item.attempt_id,
                "scenario_manifest_identity": item.entry["scenario_manifest_identity"],
                "action": action,
                "anchored_socket_command": socket_command,
                "player_envelope_fields": {
                    key: production_player[key] for key in envelope_fields
                },
                "disposition": "diagnostic_only",
                "screen_coordinate_alignment_contract": SCREEN_COORDINATE_ALIGNMENT_CONTRACT,
            },
            inputs / "replay-input.json",
        )
        replay_id = semantic_identity(
            "issue-53-human-review-diagnostic-replay-v2", item.attempt_id
        )
        game = root / "game"
        archive_details(STAGE_ROOT, game)
        _install_level(game, xml_path, replay_id)
        collection = _load_object(self.collection_path, "frozen collection plan")
        assignment = next(
            value
            for value in collection["assignments"]
            if value["exposure_role"] == item.entry["exposure_role"]
        )
        intervention = next(
            value
            for value in collection["interventions"]
            if value["id"] == item.entry["intervention_id"]
        )
        authority = {"scenario": scenario, "xml_path": xml_path}
        options = _attempt_options(
            authority,
            assignment,
            intervention,
            replay_id,
            game,
            anchor_actions=False,
        )
        options["speed"] = speed
        options["observation_exposure_role"] = item.entry["exposure_role"]
        old_display = os.environ.get("DISPLAY")
        old_stride = os.environ.get("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE")
        display_process = None
        try:
            display, display_process = start_display(root / "display.log")
            os.environ["DISPLAY"] = display
            os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
            capture_rollout = self._video_capture_rollout

            def collector(output_dir: Path, actions: list[dict[str, Any]], **kwargs):
                return collect_fresh_engine_rollouts(
                    output_dir,
                    actions,
                    capture_rollout=capture_rollout,
                    **kwargs,
                )

            result = collect_fresh_engine_attempt(
                root / "collection",
                action,
                attempt_id=replay_id,
                attempt_number=1,
                expected_initial_engine_state_identity=scenario.scenario_manifest.declared_initial_engine_state.identity,
                collector=collector,
                **options,
            )
        finally:
            terminate(display_process)
            if old_display is None:
                os.environ.pop("DISPLAY", None)
            else:
                os.environ["DISPLAY"] = old_display
            if old_stride is None:
                os.environ.pop("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", None)
            else:
                os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = old_stride
        if result["status"] != "accepted":
            raise Issue53ReviewError(
                f"Diagnostic replay capture was not accepted: {result.get('reason')}"
            )
        replay_artifact = Path(result["artifact_path"])
        replay_manifest = _load_object(
            replay_artifact.parent / "manifest.json", "diagnostic replay manifest"
        )
        replay_rollout = replay_manifest["rollouts"][0]
        replay_action = dict(replay_rollout["action"])
        alignment = replay_rollout.get("slingshot_readiness")
        if not isinstance(alignment, Mapping):
            raise Issue53ReviewError(
                "Diagnostic replay is missing screen-coordinate alignment evidence"
            )
        startup_alignment = alignment.get("startup", {}).get("alignment")
        execution_alignment = alignment.get("execution", {}).get("alignment")
        alignment_matches = bool(
            alignment.get("status") == "ready"
            and alignment.get("frozen_command") is True
            and alignment.get("anchoring") == "retained_exact_socket_command"
            and isinstance(startup_alignment, Mapping)
            and startup_alignment.get("matched") is True
            and isinstance(execution_alignment, Mapping)
            and execution_alignment.get("matched") is True
            and alignment.get("execution", {}).get("unchanged_from_startup") is True
        )
        if not alignment_matches:
            raise Issue53ReviewError(
                "Diagnostic replay screen-coordinate alignment did not match production"
            )
        verdict = compare_production_replay(
            original_artifact,
            replay_artifact,
            original_attempt_id=item.attempt_id,
            replay_attempt_id=replay_id,
            original_action=action,
            replay_action=replay_action,
            exposure_role=str(item.entry["exposure_role"]),
        )
        verdict["components"].insert(
            0,
            {
                "component": "screen_coordinate_alignment",
                "status": "equality",
                "details": dict(alignment),
            },
        )
        original_coverage = sorted(item.entry.get("realized_coverage_strata", []))
        replay_coverage = sorted(result["realized_coverage_strata"])
        coverage_matches = original_coverage == replay_coverage
        verdict["components"].append(
            {
                "component": "coverage_strata",
                "status": "equality" if coverage_matches else "mismatch",
                "details": {
                    "original": original_coverage,
                    "replay": replay_coverage,
                },
            }
        )
        verdict["passed"] = verdict["passed"] and coverage_matches
        video_path = replay_artifact / "diagnostic-video/replay.webm"
        return {
            "replay_attempt_id": replay_id,
            "replay_artifact_path": str(replay_artifact),
            "video_path": str(video_path),
            "request_71_path": str(replay_artifact / "physics_capture_v2.json"),
            "realized_coverage_strata": result["realized_coverage_strata"],
            "comparison": verdict,
            "screen_coordinate_alignment": dict(alignment),
            "investigation_required": not verdict["passed"],
        }

    @staticmethod
    def _video_capture_rollout(
        bridge,
        output_dir: Path,
        *,
        shoot,
        source_bindings: Mapping[str, object],
        scenario_manifest_identity: str,
        deadline_seconds: float = 30.0,
        clock=time.monotonic,
        sleeper=time.sleep,
    ) -> dict[str, object]:
        response = shoot()
        frames = output_dir / "diagnostic-video/frames"
        frames.mkdir(parents=True, exist_ok=True)
        frame_metadata = []
        deadline = clock() + deadline_seconds
        engine_record = None
        while clock() < deadline:
            started = clock()
            observation = bridge.get_observation_capture()
            index = len(frame_metadata)
            frame_path = frames / f"frame_{index:06d}.png"
            frame_path.write_bytes(observation.canonical_png)
            frame_metadata.append(_plain_json(observation.metadata))
            try:
                engine_capture = bridge.get_physics_capture_v2()
                engine_record = engine_capture.record
                break
            except PhysicsCaptureV2Failure as error:
                if error.code != 3:
                    raise
            sleeper(max(0.0, 1.0 / VIDEO_FPS - (clock() - started)))
        if engine_record is None:
            raise TimeoutError("Diagnostic replay did not produce terminal request-71 evidence")
        if len(frame_metadata) == 1:
            shutil.copyfile(frames / "frame_000000.png", frames / "frame_000001.png")
            frame_metadata.append(dict(frame_metadata[0]))
        metadata_path = output_dir / "diagnostic-video/frame-metadata.json"
        metadata_path.write_text(
            json.dumps(frame_metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        video_path = output_dir / "diagnostic-video/replay.webm"
        _encode_browser_video(frames, video_path)
        metadata = persist_physics_capture_v2(
            output_dir,
            engine_record,
            source_bindings=source_bindings,
            scenario_manifest_identity=scenario_manifest_identity,
        )
        validate_physics_capture_v2_artifact(output_dir, metadata)
        metadata.update(
            {
                "shoot_response": response,
                "diagnostic_video_path": str(video_path),
                "diagnostic_video_frame_count": len(frame_metadata),
                "diagnostic_only": True,
            }
        )
        return metadata

    def _follow_up_decision(self) -> dict[str, Any] | None:
        decisions = [self._load_decision(item) for item in self.items]
        if any(value is None for value in decisions):
            return None
        replay_disagreement = any(
            (result := self._load_replay_result(item)) is not None
            and isinstance(result.get("comparison"), Mapping)
            and result["comparison"].get("passed") is False
            for item in self.items
        )
        values = [str(value["decision"]) for value in decisions if value is not None]
        if replay_disagreement:
            outcome = "deterministic_replay_defect"
            next_step = "Block plan revision and investigate deterministic replay/export."
        elif "suspicious_termination_export" in values:
            outcome = "exporter_defect"
            next_step = "Require a new exporter version envelope before further collection."
        elif all(value == "confirmed_mismatch" for value in values):
            outcome = "confirmed_plan_failure"
            next_step = (
                "Keep issue #53 incomplete; run a non-final per-family action pilot, then create a fresh sealed final lineage and new partition/plan version."
            )
        else:
            outcome = "uncertain_review"
            next_step = "Resolve uncertain retained-trace interpretations before revising the plan."
        return {
            "outcome": outcome,
            "nextStep": next_step,
            "productionReleaseMayPass": False,
        }

    def _finalize_if_complete(self) -> None:
        follow_up = self._follow_up_decision()
        if follow_up is None:
            return
        full = {
            "schema": "issue_53_human_review_report_v2",
            "reviewed_at": _utc_now(),
            "item_count": len(self.items),
            "decisions": [self._load_decision(item) for item in self.items],
            "diagnostic_replays": [
                self._load_replay_result(item) for item in self.items
            ],
            "follow_up": follow_up,
            "production_evidence_modified": False,
            "production_release_may_pass": False,
        }
        write_immutable_cohort_v2_json(
            full, self.output_root / "sealed/human-review-report.json"
        )
        public_summary = {
            "schema": "issue_53_human_review_summary_v2",
            "item_count": len(self.items),
            "non_final_decision_count": sum(not item.final for item in self.items),
            "sealed_final_decision_count": sum(item.final for item in self.items),
            "follow_up": follow_up,
            "production_release_may_pass": False,
            "sealed_report": "sealed/human-review-report.json",
        }
        write_immutable_cohort_v2_json(
            public_summary, self.output_root / "human-review-summary.json"
        )
        sealed_files = sorted(
            path.relative_to(self.output_root / "sealed").as_posix()
            for path in (self.output_root / "sealed").rglob("*")
            if path.is_file() and path.name != "sealed-review-manifest.json"
        )
        write_immutable_cohort_v2_json(
            {
                "schema": "issue_53_human_review_sealed_bundle_v2",
                "identity": "issue-53-human-review-sealed-bundle-v2",
                "ordinary_workflow_access": False,
                "artifacts": sealed_files,
            },
            self.output_root / "sealed/sealed-review-manifest.json",
        )
