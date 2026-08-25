"""Compare matched-capacity oracle symbolic interfaces (issue #13)."""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Final

import torch

from world_model.data import CohortV2OracleWindowDataset, CohortV2ReleaseReader
from world_model.model import Abstraction
from world_model.training import (
    CohortV2ParallelExhaustiveEvaluator,
    validate_cohort_v2_evaluation,
    write_cohort_v2_evaluation,
)
from world_model.training.cohort_v2_micro import (
    MICRO_CAPABILITIES,
    CohortV2MicroConfig,
    CohortV2MicroError,
    CohortV2MicroPairScorer,
    CohortV2MicroTrainingData,
)
from world_model.training.cohort_v2_symbolic_interfaces import (
    INTERFACE_ORDER,
    MATERIAL_RELATION_F1_GAIN,
    CohortV2SymbolicInterfaceTrainer,
    SymbolicInterface,
    calibrate_relation_thresholds,
    collect_relation_probabilities,
    interface_compute_macs,
    load_symbolic_interface_checkpoint,
    save_symbolic_interface_checkpoint,
    score_relations,
    select_symbolic_interface,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.loop import seed_all


DEFAULT_RELEASE: Final = Path("data/runtime_evidence/issue-53-mixed-termination-v5")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-13-symbolic-interfaces")
ROLE_INFLUENCE: Final = (
    ("training", "learned_parameters"),
    ("calibration", "threshold_values"),
    ("model_selection", "configuration_selection"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--evaluation-devices",
        default="auto",
        help="comma-separated devices; auto uses every visible GPU",
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--relation-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--symbolic-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--score-log-every", type=int, default=250)
    parser.add_argument("--implementation-commit", default="working-tree")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def _readers(repository_root: Path, release_root: Path):
    declaration = repository_root / "docs/data_contracts/cohort_v2_capabilities_v1.json"
    production_plan = repository_root / "data/runtime_evidence/issue-53-plan-v5"
    readers = []
    for index, (role, influence) in enumerate(ROLE_INFLUENCE, start=1):
        print(f"[load {index}/3] validating {role} release records", flush=True)
        reader = CohortV2ReleaseReader(
            release_root,
            capability_declaration_path=declaration,
            production_plan_root=production_plan,
            workflow_kind=role,
            influence=influence,
        )
        frame_records = sum(len(rollout.frame_records) for rollout in reader.rollouts)
        print(
            f"[load {index}/3] {role}: rollouts={len(reader.rollouts)} "
            f"frame_records={frame_records}",
            flush=True,
        )
        readers.append(reader)
    return tuple(readers)


def _config(args: argparse.Namespace) -> CohortV2MicroConfig:
    return CohortV2MicroConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        symbolic_weight=args.symbolic_weight,
        device=args.device,
    )


def _evaluation_devices(args: argparse.Namespace) -> tuple[str, ...]:
    if args.evaluation_batch_size <= 0 or args.relation_batch_size <= 0:
        raise CohortV2MicroError("evaluation batch sizes must be positive")
    if args.evaluation_devices == "auto":
        if torch.device(args.device).type == "cuda":
            count = torch.cuda.device_count()
            if count == 0:
                raise CohortV2MicroError("CUDA requested but no GPU is visible")
            return tuple(f"cuda:{index}" for index in range(count))
        return (args.device,)
    devices = tuple(value.strip() for value in args.evaluation_devices.split(","))
    if not devices or any(not value for value in devices) or len(set(devices)) != len(devices):
        raise CohortV2MicroError("evaluation devices must be unique and nonempty")
    for value in devices:
        device = torch.device(value)
        if device.type == "cuda" and (
            device.index is None or device.index >= torch.cuda.device_count()
        ):
            raise CohortV2MicroError(f"evaluation device is unavailable: {value}")
    return devices


def _relation_payload(scores) -> dict[str, object]:
    payload = {}
    for predicate, counts in scores.items():
        payload[predicate] = {
            **asdict(counts),
            "f1": counts.f1,
        }
    payload["macro_f1"] = sum(counts.f1 for counts in scores.values()) / len(scores)
    return payload


def _model_selection_objectives(evaluation) -> list[dict[str, object]]:
    rows = []
    states = tuple(
        state for state in evaluation.states if state.exposure_role == "model_selection"
    )
    for horizon in (1, 5, 15):
        values = tuple(
            float(outcome.objective)
            for state in states
            for outcome in state.outcomes
            if outcome.pair.delta == horizon
            and outcome.pair.abstraction is Abstraction.MICRO
            and outcome.available
        )
        rows.append({
            "available_state_count": len(values),
            "mean_objective": sum(values) / len(values),
            "requested_horizon": horizon,
        })
    return rows


def _mean_interface_compute(reader, interface: SymbolicInterface, hidden_dim: int) -> float:
    costs = []
    for window in CohortV2OracleWindowDataset(reader, requested_horizons=(1, 5, 15)):
        contact = window.context.labels["contact"]
        supports = window.context.labels["supports"]
        if (
            contact.get("availability") != "available"
            or supports.get("availability") != "available"
        ):
            continue
        relations = (*contact["relations"], *supports["relations"])
        entities = {entity for relation in relations for entity in relation}
        costs.append(interface_compute_macs(
            interface,
            hidden_dim,
            len(contact["relations"]),
            len(supports["relations"]),
            len(entities),
        ))
    return sum(costs) / len(costs)


def _dry_run(
    readers,
    config: CohortV2MicroConfig,
    relation_batch_size: int,
) -> int:
    dry_config = replace(config, batch_size=min(2, config.batch_size))
    parameter_counts = set()
    macro_f1 = {}
    for index, interface in enumerate(INTERFACE_ORDER, start=1):
        print(f"[dry-run variant {index}/4] {interface}", flush=True)
        seed_all(dry_config.seed)
        data = CohortV2MicroTrainingData(readers[0], dry_config)
        trainer = CohortV2SymbolicInterfaceTrainer(data, dry_config, interface)
        seen = set()
        while seen != {Abstraction.CONTINUOUS, Abstraction.MICRO}:
            result = trainer.train_step()
            seen.add(result.pair.abstraction)
            print(
                f"[dry-run {interface}] step={result.step + 1} "
                f"pair=h{result.pair.delta}/{result.pair.abstraction} "
                f"loss={result.total_loss:.6f}",
                flush=True,
            )
        parameter_counts.add(sum(p.numel() for p in trainer.predictor.parameters()))
        calibration = collect_relation_probabilities(
            trainer.predictor,
            trainer.codec,
            readers[1],
            batch_size=relation_batch_size,
            limit=12,
        )
        thresholds = calibrate_relation_thresholds(calibration)
        held_out = collect_relation_probabilities(
            trainer.predictor,
            trainer.codec,
            readers[2],
            batch_size=relation_batch_size,
            limit=12,
        )
        scores = score_relations(held_out, thresholds)
        macro_f1[interface] = sum(value.f1 for value in scores.values()) / len(scores)
        print(
            f"[dry-run {interface}] parameter_count="
            f"{sum(p.numel() for p in trainer.predictor.parameters())} "
            f"held_out_macro_f1={macro_f1[interface]:.6f}",
            flush=True,
        )
    if len(parameter_counts) != 1:
        raise CohortV2MicroError("matched-capacity variants have different sizes")
    selected, comparisons = select_symbolic_interface(macro_f1)
    print(
        f"[dry-run decision] selected={selected} comparisons={len(comparisons)}",
        flush=True,
    )
    print("[dry-run] no files written and no final-evaluation records accessed", flush=True)
    return 0


def _validate_existing(
    output: Path,
    readers,
    config: CohortV2MicroConfig,
    device: str,
) -> int:
    summary = json.loads((output / "summary.json").read_bytes())
    macro_f1 = {}
    parameter_counts = set()
    for index, interface in enumerate(INTERFACE_ORDER, start=1):
        root = output / str(interface)
        predictor, codec, checkpoint = load_symbolic_interface_checkpoint(
            root / "checkpoint.pt",
            reader=readers[0],
            config=config,
            interface=interface,
            device=device,
        )
        scorer = CohortV2MicroPairScorer(
            predictor, codec, checkpoint, config, readers
        )
        receipt = validate_cohort_v2_evaluation(
            root / "pair_evaluation",
            readers=readers,
            checkpoint_identity=checkpoint.identity,
            checkpoint_capabilities=MICRO_CAPABILITIES,
            objective_identity=scorer.objective_identity,
        )
        stored = summary["variants"][str(interface)]
        if (
            stored["checkpoint_identity"] != checkpoint.identity
            or stored["evaluation_identity"] != receipt.evaluation_identity
            or stored["trainable_parameter_count"] != checkpoint.trainable_parameter_count
        ):
            raise CohortV2MicroError(f"stored summary differs for {interface}")
        macro_f1[interface] = float(stored["model_selection_relations"]["macro_f1"])
        parameter_counts.add(checkpoint.trainable_parameter_count)
        print(
            f"[validate {index}/4] {interface}: states={receipt.state_count} "
            f"available={receipt.available_count}",
            flush=True,
        )
    selected, comparisons = select_symbolic_interface(macro_f1)
    if (
        len(parameter_counts) != 1
        or summary["decision"]["selected_interface"] != str(selected)
        or summary["decision"]["comparisons"] != list(comparisons)
    ):
        raise CohortV2MicroError("stored keep/remove decision does not recompute")
    print(f"[validate] passed selected={selected}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.log_every <= 0 or args.score_log_every <= 0:
        raise CohortV2MicroError("progress intervals must be positive")
    repository_root = args.repository_root.resolve()
    release_root = (repository_root / args.release_root).resolve()
    output = (repository_root / args.output).resolve()
    readers = _readers(repository_root, release_root)
    config = _config(args)
    evaluation_devices = _evaluation_devices(args)
    if args.dry_run:
        return _dry_run(readers, config, args.relation_batch_size)
    if args.validate:
        return _validate_existing(output, readers, config, args.device)

    variants = {}
    macro_f1 = {}
    parameter_counts = set()
    for variant_index, interface in enumerate(INTERFACE_ORDER, start=1):
        root = output / str(interface)
        seed_all(config.seed)
        data = CohortV2MicroTrainingData(readers[0], config)
        trainer = CohortV2SymbolicInterfaceTrainer(data, config, interface)
        print(
            f"[variant {variant_index}/4] {interface}: train steps={config.steps} "
            f"batch={config.batch_size} device={config.device}",
            flush=True,
        )
        first_loss = None
        latest = None
        for step in range(config.steps):
            latest = trainer.train_step()
            if first_loss is None:
                first_loss = latest.total_loss
            if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == config.steps:
                print(
                    f"[train {interface} {step + 1}/{config.steps}] "
                    f"pair=h{latest.pair.delta}/{latest.pair.abstraction} "
                    f"total={latest.total_loss:.6f} carrier={latest.carrier_loss:.6f} "
                    f"micro={latest.micro_loss:.6f} lr={latest.learning_rate:.2e}",
                    flush=True,
                )
        assert first_loss is not None and latest is not None
        checkpoint = save_symbolic_interface_checkpoint(
            root / "checkpoint.pt", trainer
        )
        parameter_counts.add(checkpoint.trainable_parameter_count)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scorers = []
        primary_predictor = None
        primary_codec = None
        for device in evaluation_devices:
            predictor, codec, loaded = load_symbolic_interface_checkpoint(
                root / "checkpoint.pt",
                reader=readers[0],
                config=config,
                interface=interface,
                device=device,
            )
            if primary_predictor is None:
                primary_predictor, primary_codec = predictor, codec
            scorers.append(CohortV2MicroPairScorer(
                predictor,
                codec,
                loaded,
                config,
                readers,
                progress_every=args.score_log_every,
                worker_name=f"{interface}:{device}",
            ))
        print(
            f"[score {interface}] exhaustive pair grid devices={evaluation_devices} "
            f"batch_size={args.evaluation_batch_size}",
            flush=True,
        )
        evaluation = CohortV2ParallelExhaustiveEvaluator(
            tuple(scorers), batch_size=args.evaluation_batch_size
        ).evaluate(readers)
        receipt = write_cohort_v2_evaluation(
            root / "pair_evaluation", evaluation, readers=readers
        )
        assert primary_predictor is not None and primary_codec is not None
        calibration_values = collect_relation_probabilities(
            primary_predictor,
            primary_codec,
            readers[1],
            batch_size=args.relation_batch_size,
            progress_label=f"{interface}:calibration",
        )
        thresholds = calibrate_relation_thresholds(calibration_values)
        model_selection_values = collect_relation_probabilities(
            primary_predictor,
            primary_codec,
            readers[2],
            batch_size=args.relation_batch_size,
            progress_label=f"{interface}:model_selection",
        )
        relation_scores = score_relations(model_selection_values, thresholds)
        relation_payload = _relation_payload(relation_scores)
        macro_f1[interface] = float(relation_payload["macro_f1"])
        variants[str(interface)] = {
            "adapter_parameter_count": checkpoint.adapter_parameter_count,
            "checkpoint_identity": checkpoint.identity,
            "evaluation_identity": receipt.evaluation_identity,
            "final_loss": latest.total_loss,
            "first_loss": first_loss,
            "mean_interface_compute_macs": _mean_interface_compute(
                readers[2], interface, config.hidden_dim
            ),
            "model_selection_micro_objectives": _model_selection_objectives(evaluation),
            "model_selection_relations": relation_payload,
            "relation_thresholds_from_calibration": thresholds,
            "trainable_parameter_count": checkpoint.trainable_parameter_count,
        }
        print(
            f"[variant complete {variant_index}/4] {interface}: "
            f"macro_f1={macro_f1[interface]:.6f} states={receipt.state_count}",
            flush=True,
        )
        del scorers, primary_predictor, evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(parameter_counts) != 1:
        raise CohortV2MicroError("matched-capacity variants have different sizes")
    selected, comparisons = select_symbolic_interface(macro_f1)
    summary = {
        "artifact_type": "cohort_v2_symbolic_interface_comparison",
        "bounded_negative_disposition": (
            "not_required_named_secondary_spsg_contrastive_loss_ablation"
        ),
        "capabilities": sorted(MICRO_CAPABILITIES),
        "decision": {
            "comparisons": list(comparisons),
            "material_relation_macro_f1_gain": MATERIAL_RELATION_F1_GAIN,
            "selected_interface": str(selected),
            "spsg_decision": "keep" if selected is SymbolicInterface.SPSG else "remove",
        },
        "endpoint_plausibility_selection_status": (
            "not_used:source-endpoint derivation labels are identical across interfaces; "
            "predicted carriers are not engine frame records"
        ),
        "final_evaluation_accessed": False,
        "implementation_commit": args.implementation_commit,
        "matched_capacity_parameter_count": next(iter(parameter_counts)),
        "micro_relation_authority": "cohort-v2-micro-relation-derivation-spec-v1:contact+supports",
        "release_identity": readers[0].release_identity,
        "rerun_commands": [
            "python -u -m scripts.run_cohort_v2_symbolic_interfaces --dry-run",
            "python -u -m scripts.run_cohort_v2_symbolic_interfaces",
            "python -u -m scripts.run_cohort_v2_symbolic_interfaces --validate",
        ],
        "role_binding_claim": "not_made",
        "schema": "cohort_v2_symbolic_interface_comparison_v1",
        "variants": variants,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_bytes(canonical_json_bytes(summary))
    print(
        f"[complete] selected={selected} spsg={summary['decision']['spsg_decision']} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CohortV2MicroError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
