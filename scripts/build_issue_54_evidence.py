"""Build and revalidate fail-closed downstream ingestion evidence for issue #54."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Final, Mapping

from world_model.data.cohort_v2 import (
    CohortV2IngestionError,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
    probe_cohort_v2_final_access,
    score_cohort_v2_endpoints,
)


SCHEMA: Final = "cohort_v2_downstream_ingestion_evidence_v1"
IDENTITY: Final = "cohort-v2-downstream-ingestion-evidence-v1:issue-54:release-v5"
REPORT_NAME: Final = "cohort-v2-downstream-ingestion-evidence.json"
DEFAULT_RELEASE: Final = Path("data/runtime_evidence/issue-53-mixed-termination-v5")
DEFAULT_SEALED: Final = Path(
    ".local-artifacts/issue-53-mixed-termination-final-release-v5"
)
ROLE_INFLUENCE: Final = {
    "training": "learned_parameters",
    "calibration": "threshold_values",
    "model_selection": "configuration_selection",
}


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _all_false(frame: Any) -> dict[str, Any]:
    return {
        "steady-state": False,
        "structure-unstable": False,
        "excess_penetration": False,
        "unsupported_stationary_or_floating_body": {
            item["entity_id"]: False
            for item in frame.labels["unsupported_stationary_or_floating_body"]
        },
    }


def _expect_rejection(name: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        operation()
    except CohortV2IngestionError as error:
        return {"name": name, "passed": True, "rejection": str(error)}
    raise ValueError(f"Adversarial check did not fail closed: {name}")


def _mutated_reader(
    release_root: Path,
    capability_declaration: Path,
    relative_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    with tempfile.TemporaryDirectory(dir=release_root.parent) as temporary:
        shadow = Path(temporary) / "release"
        shutil.copytree(release_root, shadow, copy_function=os.link)
        target = shadow / relative_path
        target.unlink()
        shutil.copy2(release_root / relative_path, target)
        value = _load(target)
        mutate(value)
        target.write_bytes(_canonical_json(value))
        CohortV2ReleaseReader(
            shadow,
            capability_declaration_path=capability_declaration,
            workflow_kind="training",
            influence="learned_parameters",
        )


def _adversarial_checks(
    release_root: Path,
    capability_declaration: Path,
    training_reader: CohortV2ReleaseReader,
) -> list[dict[str, Any]]:
    index = _load(release_root / "authoritative-derivation-index.json")
    first_attempt = next(
        item["attempt_id"]
        for item in index["artifacts"]
        if item["exposure_role"] == "training"
    )
    paths = {
        item["kind"]: Path(item["path"])
        for item in index["artifacts"]
        if item["attempt_id"] == first_attempt
    }

    def mutate_contact(value: dict[str, Any]) -> None:
        value["labels"][0]["predicates"]["contact"]["relations"].append(
            ["invented:a", "invented:b"]
        )

    def mutate_supports(value: dict[str, Any]) -> None:
        value["labels"][0]["predicates"]["supports"]["relations"].append(
            ["invented:a", "invented:b"]
        )

    def mutate_boolean(predicate: str) -> Callable[[dict[str, Any]], None]:
        def mutate(value: dict[str, Any]) -> None:
            label = value["labels"][0]["predicates"][predicate]
            label["availability"] = "available"
            label["value"] = False

        return mutate

    def mutate_excess(value: dict[str, Any]) -> None:
        label = value["labels"][0]["predicates"]["excess_penetration"]
        label["value"] = not label["value"]

    def mutate_unsupported(value: dict[str, Any]) -> None:
        label = value["labels"][0]["predicates"][
            "unsupported_stationary_or_floating_body"
        ][0]
        label["availability"] = "available"
        label["value"] = False

    mutations: dict[str, tuple[Path, Callable[[dict[str, Any]], None]]] = {
        "mutated_contact": (paths["micro"], mutate_contact),
        "mutated_supports": (paths["micro"], mutate_supports),
        "mutated_steady_state": (paths["macro"], mutate_boolean("steady-state")),
        "mutated_structure_unstable": (
            paths["macro"],
            mutate_boolean("structure-unstable"),
        ),
        "mutated_excess_penetration": (
            paths["physical-violations"],
            mutate_excess,
        ),
        "mutated_unsupported_body": (
            paths["physical-violations"],
            mutate_unsupported,
        ),
        "cross_release_sidecar": (
            paths["micro"],
            lambda value: value["source"].__setitem__(
                "capture_bundle_identity", "representative-cohort-v2-release-v5:other"
            ),
        ),
        "temporal_derivation_mismatch": (
            paths["macro"],
            lambda value: value["labels"][0].__setitem__(
                "fixed_step", value["labels"][0]["fixed_step"] + 1
            ),
        ),
        "malformed_publication_envelope": (
            Path("cohort-v2-publication.json"),
            lambda value: value.__setitem__("unexpected", True),
        ),
    }
    checks = [
        _expect_rejection(
            name,
            lambda path=path, mutate=mutate: _mutated_reader(
                release_root, capability_declaration, path, mutate
            ),
        )
        for name, (path, mutate) in mutations.items()
    ]
    checks.append(_expect_rejection(
        "excluded_capability_request",
        lambda: CohortV2ReleaseReader(
            release_root,
            capability_declaration_path=capability_declaration,
            workflow_kind="training",
            influence="learned_parameters",
            requested_capabilities=("physical_regime_gate",),
        ),
    ))
    checks.append(_expect_rejection(
        "canonical_model_input",
        lambda: training_reader.load_observation(
            training_reader.rollouts[0], observation_role="canonical"
        ),
    ))
    checks.append(_expect_rejection(
        "ordinary_final_evaluation",
        lambda: CohortV2ReleaseReader(
            release_root,
            capability_declaration_path=capability_declaration,
            workflow_kind="final_evaluation",
            influence="frozen_final_metrics_after_authorization",
        ),
    ))
    return checks


def build_ingestion_evidence(
    *,
    repository_root: Path,
    release_root: Path,
    sealed_root: Path,
    code_revision: str,
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    release_root = Path(release_root).resolve()
    sealed_root = Path(sealed_root).resolve()
    if not isinstance(code_revision, str) or not code_revision:
        raise ValueError("code_revision must be a nonempty string")
    declaration = repository_root / "docs/data_contracts/cohort_v2_capabilities_v1.json"
    readers = {
        role: CohortV2ReleaseReader(
            release_root,
            capability_declaration_path=declaration,
            workflow_kind=role,
            influence=influence,
        )
        for role, influence in ROLE_INFLUENCE.items()
    }
    roles = {}
    total_frames = total_windows = total_scored = total_unavailable = 0
    for role, reader in readers.items():
        windows = CohortV2OracleWindowDataset(reader)
        endpoints = score_cohort_v2_endpoints(reader, _all_false)
        frame_count = sum(len(rollout.frames) for rollout in reader.rollouts)
        termination_counts = Counter(
            str(rollout.frames[-1].terminal["reason"])
            for rollout in reader.rollouts
        )
        observation_leads = [
            rollout.frames[0].fixed_step - rollout.agent_observation_fixed_step
            for rollout in reader.rollouts
        ]
        roles[role] = {
            "influence": ROLE_INFLUENCE[role],
            "rollouts": len(reader.rollouts),
            "central_strata": sorted(
                rollout.coverage_stratum for rollout in reader.rollouts
            ),
            "frame_records": frame_count,
            "agent_observations": len(windows),
            "oracle_training_windows": len(windows),
            "observation_lead_fixed_steps": {
                "minimum": min(observation_leads),
                "maximum": max(observation_leads),
            },
            "terminal_records": len(reader.rollouts),
            "termination_counts": dict(sorted(termination_counts.items())),
            "endpoint_scoring": asdict(endpoints),
        }
        total_frames += frame_count
        total_windows += len(windows)
        total_scored += endpoints.scored_value_count
        total_unavailable += endpoints.unavailable_value_count

    adversarial = _adversarial_checks(release_root, declaration, readers["training"])
    final_receipt = probe_cohort_v2_final_access(release_root, sealed_root)
    report = {
        "schema": SCHEMA,
        "identity": IDENTITY,
        "code_revision": code_revision,
        "reader_identity": "world-model-cohort-v2-reader-v1",
        "capability_declaration_identity": "cohort-v2-capabilities-v1",
        "publication_identity": "representative-cohort-v2-publication-v5:issue-53:mixed-termination",
        "cohort_release_identity": readers["training"].release_identity,
        "partition_identity": readers["training"].partition_identity,
        "authoritative_derivation_index_identity": readers["training"].derivation_identity,
        "roles": roles,
        "counts": {
            "rollouts": sum(len(reader.rollouts) for reader in readers.values()),
            "frame_records": total_frames,
            "agent_observations": total_windows,
            "oracle_training_windows": total_windows,
            "endpoint_scored_values": total_scored,
            "endpoint_unavailable_values": total_unavailable,
        },
        "training_examples": {
            "artifact_kind": "release_bound_derived_product",
            "primary_artifacts_mutated": False,
        },
        "observation_access": {
            "model_input": "agent_only",
            "canonical": "diagnostic_only",
        },
        "authorized_final_evaluation_probe": asdict(final_receipt),
        "adversarial_checks": adversarial,
        "passed": final_receipt.passed and all(
            check["passed"] for check in adversarial
        ),
    }
    return report


def _code_revision(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED)
    parser.add_argument("--output", type=Path, default=Path("data/runtime_evidence/issue-54"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    output = (repository_root / args.output).resolve()
    existing = _load(output / REPORT_NAME) if args.validate else None
    revision = existing["code_revision"] if existing is not None else _code_revision(repository_root)
    report = build_ingestion_evidence(
        repository_root=repository_root,
        release_root=repository_root / args.release_root,
        sealed_root=repository_root / args.sealed_root,
        code_revision=revision,
    )
    if existing is not None:
        if existing != report:
            raise ValueError("Issue-54 ingestion evidence differs from exact revalidation")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / REPORT_NAME).write_bytes(_canonical_json(report))
    print(json.dumps({
        "identity": report["identity"],
        "rollouts": report["counts"]["rollouts"],
        "oracle_training_windows": report["counts"]["oracle_training_windows"],
        "adversarial_checks": len(report["adversarial_checks"]),
        "passed": report["passed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
