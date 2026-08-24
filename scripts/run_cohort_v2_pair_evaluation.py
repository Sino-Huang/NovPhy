"""Run or validate the issue-3 cohort-v2 exhaustive capability audit."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from world_model.data import CohortV2ReleaseReader
from world_model.training.cohort_v2_evaluation import (
    ENDPOINT_CAPABILITIES,
    CohortV2ExhaustiveEvaluator,
    CohortV2PairGrid,
    validate_cohort_v2_evaluation,
    write_cohort_v2_evaluation,
)
from world_model.training.grid_artifacts import canonical_json_bytes


DEFAULT_RELEASE: Final = Path("data/runtime_evidence/issue-53-mixed-termination-v5")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-3-capability-audit")
ROLE_INFLUENCE: Final = (
    ("training", "learned_parameters"),
    ("calibration", "threshold_values"),
    ("model_selection", "configuration_selection"),
)


@dataclass(frozen=True, slots=True)
class _CapabilityAuditScorer:
    """Declare that no cohort-v2 transition checkpoint exists yet."""

    checkpoint_identity: str = "checkpoint-unavailable:issue-3-capability-audit"
    objective_identity: str = "pair-objective-unavailable:issue-3-capability-audit"
    capabilities: frozenset[str] = frozenset()

    def objective(self, window, pair) -> float:
        del window, pair
        raise AssertionError("an unavailable checkpoint must never emit a score")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compact-report", type=Path)
    parser.add_argument("--implementation-commit")
    return parser


def _readers(repository_root: Path, release_root: Path):
    declaration = repository_root / "docs/data_contracts/cohort_v2_capabilities_v1.json"
    production_plan = repository_root / "data/runtime_evidence/issue-53-plan-v5"
    readers = []
    for index, (role, influence) in enumerate(ROLE_INFLUENCE, start=1):
        print(f"[{index}/4] loading and validating {role} release records", flush=True)
        reader = CohortV2ReleaseReader(
            release_root,
            capability_declaration_path=declaration,
            production_plan_root=production_plan,
            workflow_kind=role,
            influence=influence,
        )
        frame_records = sum(len(rollout.frame_records) for rollout in reader.rollouts)
        print(
            f"[{index}/4] {role}: rollouts={len(reader.rollouts)} "
            f"frame_records={frame_records}",
            flush=True,
        )
        readers.append(reader)
    return tuple(readers)


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.compact_report is not None and not args.implementation_commit:
        parser.error("--compact-report requires --implementation-commit")
    repository_root = args.repository_root.resolve()
    output = (repository_root / args.output).resolve()
    release_root = (repository_root / args.release_root).resolve()
    readers = _readers(repository_root, release_root)
    scorer = _CapabilityAuditScorer()
    if args.validate:
        print(f"[validate] reading {output}", flush=True)
        receipt = validate_cohort_v2_evaluation(
            output,
            readers=readers,
            checkpoint_identity=scorer.checkpoint_identity,
            checkpoint_capabilities=scorer.capabilities,
            objective_identity=scorer.objective_identity,
        )
        print(
            f"[validate] passed states={receipt.state_count} "
            f"outcomes={receipt.outcome_count} "
            f"available={receipt.available_count} "
            f"unavailable={receipt.unavailable_count}",
            flush=True,
        )
        return 0

    grid = CohortV2PairGrid()
    print(
        f"[4/4] enumerating horizons={grid.horizons} "
        f"pairs={len(grid.pairs)} over every nonterminal state",
        flush=True,
    )
    result = CohortV2ExhaustiveEvaluator(scorer, grid).evaluate(readers)
    print(
        f"[4/4] complete states={len(result.states)} "
        f"outcomes={result.outcome_count} available={result.available_count} "
        f"unavailable={result.unavailable_count}",
        flush=True,
    )
    if args.dry_run:
        print("[dry-run] no files written", flush=True)
        return 0
    receipt = write_cohort_v2_evaluation(output, result, readers=readers)
    print(
        f"[write] {output} records={receipt.records_identity}", flush=True
    )
    if args.compact_report is not None:
        report_path = (repository_root / args.compact_report).resolve()
        report = {
            "artifact_type": "cohort_v2_exhaustive_pair_evaluation_summary",
            "available_count": receipt.available_count,
            "capability_declaration_identity": receipt.capability_declaration_identity,
            "checkpoint_capabilities": list(receipt.checkpoint_capabilities),
            "checkpoint_identity": receipt.checkpoint_identity,
            "endpoint_capabilities": list(ENDPOINT_CAPABILITIES),
            "evaluation_identity": receipt.evaluation_identity,
            "grid_identity": receipt.grid_identity,
            "implementation_commit": args.implementation_commit,
            "model_objectives_rerun_during_validation": False,
            "objective_identity": receipt.objective_identity,
            "outcome_count": receipt.outcome_count,
            "partition_identity": receipt.partition_identity,
            "records_identity": receipt.records_identity,
            "release_identity": receipt.release_identity,
            "rerun_commands": [
                "python -u -m scripts.run_cohort_v2_pair_evaluation --dry-run",
                "python -u -m scripts.run_cohort_v2_pair_evaluation",
                "python -u -m scripts.run_cohort_v2_pair_evaluation --validate",
            ],
            "schema": "cohort_v2_exhaustive_pair_evaluation_summary_v1",
            "source_bound_validation": "passed",
            "state_count": receipt.state_count,
            "state_set_identity": receipt.state_set_identity,
            "unavailable_count": receipt.unavailable_count,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(canonical_json_bytes(report))
        print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
