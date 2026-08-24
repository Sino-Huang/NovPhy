"""Oracle macro-event training and scoring for the complete cohort-v2 pair grid."""
from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from world_model.data import (
    CohortV2CentralFrameRecord,
    CohortV2OracleWindow,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
)
from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    COHORT_V2_RELEASE_IDENTITY,
    MACRO_SPEC_IDENTITY,
)
from world_model.model import (
    Abstraction,
    DualOutputPredictor,
    MacroReadoutHead,
    PredictionPair,
    PredictorConfig,
    identity,
)
from world_model.training.cohort_v2 import build_cohort_v2_transition_request
from world_model.training.cohort_v2_evaluation import (
    COHORT_V2_HORIZONS,
    TERMINAL_EVENT_ENDPOINT_CAPABILITY,
    CohortV2EvaluationResult,
)
from world_model.training.cohort_v2_micro import (
    MICRO_RELATION_AUTHORITY,
    CohortV2StateCodec,
    cohort_v2_action,
    cohort_v2_model_state_identity,
    micro_predicate_loss,
    micro_relation_loss,
)
from world_model.training.grid_artifacts import canonical_json_bytes


MACRO_CHECKPOINT_SCHEMA: Final = "cohort_v2_macro_checkpoint_v1"
MACRO_FRONTIER_SCHEMA: Final = "cohort_v2_macro_frontier_input_v1"
MACRO_STATE_AUTHORITY: Final = MACRO_SPEC_IDENTITY
MACRO_EVENT_ENDPOINT_AUTHORITY: Final = (
    "cohort-v2-terminal-event-endpoint-v1:requested-horizon-clamped:"
    "stable_entered+level_fail"
)
MACRO_PREDICATES: Final = ("steady-state", "structure-unstable")
MACRO_EVENT_TYPES: Final = ("stable_entered", "level_fail")
MACRO_CAPABILITIES: Final = frozenset({
    "transition.continuous",
    "transition.micro",
    "transition.macro",
    TERMINAL_EVENT_ENDPOINT_CAPABILITY,
})
MACRO_PAIRS: Final = tuple(
    PredictionPair(horizon, abstraction)
    for horizon in COHORT_V2_HORIZONS
    for abstraction in Abstraction
)


class CohortV2MacroError(ValueError):
    """The oracle macro training, checkpoint, or frontier input is invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2MacroConfig:
    seed: int = 20260824
    steps: int = 1800
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    symbolic_weight: float = 1.0
    latent_dim: int = 192
    hidden_dim: int = 384
    depth: int = 3
    max_entities: int = 12
    device: str = "cuda"

    def __post_init__(self) -> None:
        for field in (
            "seed", "steps", "batch_size", "latent_dim", "hidden_dim",
            "depth", "max_entities",
        ):
            value = getattr(self, field)
            minimum = 0 if field == "seed" else 1
            if type(value) is not int or value < minimum:
                raise CohortV2MacroError(f"{field} must be an integer >= {minimum}")
        if self.latent_dim < 2 + self.max_entities * 13:
            raise CohortV2MacroError("latent_dim cannot hold the declared entity slots")
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.grad_clip <= 1e3
            or self.symbolic_weight < 0.0
        ):
            raise CohortV2MacroError("optimizer or symbolic-loss configuration is invalid")
        if type(self.device) is not str or not self.device.strip():
            raise CohortV2MacroError("device must be nonempty")

    @property
    def predictor_config(self) -> PredictorConfig:
        return PredictorConfig(
            latent_dim=self.latent_dim,
            action_dim=5,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            event_type_count=len(MACRO_EVENT_TYPES),
        )

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-macro-training-config-v1",
            self.seed,
            self.steps,
            self.batch_size,
            self.learning_rate,
            self.weight_decay,
            self.grad_clip,
            self.symbolic_weight,
            self.latent_dim,
            self.hidden_dim,
            self.depth,
            self.max_entities,
            tuple(pair.identity for pair in MACRO_PAIRS),
            MACRO_EVENT_TYPES,
        ))


def _available(frame: CohortV2CentralFrameRecord, predicates: tuple[str, ...]) -> bool:
    return all(
        frame.labels[predicate].get("availability") == "available"
        for predicate in predicates
    )


def macro_event_endpoint_available(window: CohortV2OracleWindow) -> bool:
    """Return whether this macro action reaches an accepted terminal event."""
    terminal = window.target.terminal
    return (
        terminal is not None
        and terminal.get("reason") in MACRO_EVENT_TYPES
        and _available(window.context, MACRO_PREDICATES)
        and _available(window.target, MACRO_PREDICATES)
    )


@dataclass(frozen=True, slots=True)
class MacroReadoutLoss:
    loss: torch.Tensor
    per_example: torch.Tensor
    macro_state_loss: torch.Tensor
    endpoint_delta_loss: torch.Tensor
    event_type_loss: torch.Tensor
    supervised_predicate_count: int
    endpoint_count: int


def macro_readout_loss(
    head: MacroReadoutHead,
    carrier: torch.Tensor,
    windows: tuple[CohortV2OracleWindow, ...],
    *,
    weights: torch.Tensor | None = None,
) -> MacroReadoutLoss:
    """Supervise endpoint macro state, duration, and terminal event class."""
    if not isinstance(head, MacroReadoutHead):
        raise CohortV2MacroError("macro loss requires the model macro head")
    if (
        not isinstance(carrier, torch.Tensor)
        or carrier.ndim != 2
        or type(windows) is not tuple
        or len(windows) != carrier.shape[0]
        or not windows
    ):
        raise CohortV2MacroError("macro windows must match the carrier batch")
    if not all(macro_event_endpoint_available(window) for window in windows):
        raise CohortV2MacroError("macro supervision requires an available terminal endpoint")
    if weights is None:
        weights = torch.ones(carrier.shape[0], device=carrier.device)
    if not isinstance(weights, torch.Tensor) or weights.shape != (carrier.shape[0],):
        raise CohortV2MacroError("macro loss weights must match the carrier batch")

    readout = head(carrier)
    macro_targets = torch.tensor(
        [
            [float(window.target.labels[predicate]["value"]) for predicate in MACRO_PREDICATES]
            for window in windows
        ],
        dtype=carrier.dtype,
        device=carrier.device,
    )
    macro_per_example = F.binary_cross_entropy_with_logits(
        readout.macro_logits, macro_targets, reduction="none"
    ).mean(dim=1)
    delta_targets = torch.tensor(
        [window.effective_horizon / window.requested_horizon for window in windows],
        dtype=carrier.dtype,
        device=carrier.device,
    )
    delta_per_example = F.smooth_l1_loss(
        readout.delta_hat.squeeze(-1), delta_targets, reduction="none"
    )
    event_targets = torch.tensor(
        [MACRO_EVENT_TYPES.index(window.target.terminal["reason"]) for window in windows],
        dtype=torch.long,
        device=carrier.device,
    )
    event_per_example = F.cross_entropy(
        readout.event_logits, event_targets, reduction="none"
    )
    per_example = (macro_per_example + delta_per_example + event_per_example) / 3.0

    def weighted(values: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum()

    return MacroReadoutLoss(
        loss=weighted(per_example),
        per_example=per_example,
        macro_state_loss=weighted(macro_per_example),
        endpoint_delta_loss=weighted(delta_per_example),
        event_type_loss=weighted(event_per_example),
        supervised_predicate_count=len(windows) * len(MACRO_PREDICATES),
        endpoint_count=len(windows),
    )


class CohortV2MacroTrainingData:
    """Balanced full-grid pools sourced only from the training exposure role."""

    def __init__(self, reader: CohortV2ReleaseReader, config: CohortV2MacroConfig) -> None:
        if not isinstance(reader, CohortV2ReleaseReader):
            raise CohortV2MacroError("macro training requires a validated release reader")
        if not reader.rollouts or {rollout.exposure_role for rollout in reader.rollouts} != {"training"}:
            raise CohortV2MacroError("learned parameters may use only the training exposure role")
        if reader.release_identity != COHORT_V2_RELEASE_IDENTITY:
            raise CohortV2MacroError("macro training reader targets another release")
        dataset = CohortV2OracleWindowDataset(
            reader, requested_horizons=COHORT_V2_HORIZONS
        )
        pools: dict[PredictionPair, list[CohortV2OracleWindow]] = {
            pair: [] for pair in MACRO_PAIRS
        }
        for window in dataset:
            if window.effective_horizon == window.requested_horizon:
                pools[PredictionPair(window.requested_horizon, Abstraction.CONTINUOUS)].append(window)
                if _available(window.context, ("contact", "supports")) and _available(
                    window.target, ("contact", "supports")
                ):
                    pools[PredictionPair(window.requested_horizon, Abstraction.MICRO)].append(window)
            if macro_event_endpoint_available(window):
                pools[PredictionPair(window.requested_horizon, Abstraction.MACRO)].append(window)
        empty = tuple(pair.identity for pair, values in pools.items() if not values)
        if empty:
            raise CohortV2MacroError(f"training has no eligible windows for pairs: {empty}")
        self.reader = reader
        self.config = config
        self.pools = {pair: tuple(values) for pair, values in pools.items()}
        self.frame_counts = {
            rollout.attempt_id: len(rollout.frame_records) for rollout in reader.rollouts
        }

    def schedule_at(self, step: int) -> PredictionPair:
        if type(step) is not int or step < 0:
            raise CohortV2MacroError("step must be nonnegative")
        keys = list(MACRO_PAIRS)
        cycle, offset = divmod(step, len(keys))
        generator = random.Random(self.config.seed)
        for _ in range(cycle + 1):
            generator.shuffle(keys)
        return keys[offset]

    def batch_at(
        self, pair: PredictionPair, step: int
    ) -> tuple[CohortV2OracleWindow, ...]:
        pool = self.pools[pair]
        pair_offset = MACRO_PAIRS.index(pair)
        generator = random.Random(self.config.seed + step * len(MACRO_PAIRS) + pair_offset)
        return tuple(
            pool[generator.randrange(len(pool))] for _ in range(self.config.batch_size)
        )

    def duration_weights(
        self, windows: tuple[CohortV2OracleWindow, ...]
    ) -> torch.Tensor:
        return torch.tensor(
            [
                window.effective_horizon / (self.frame_counts[window.attempt_id] - 1)
                for window in windows
            ],
            dtype=torch.float32,
        )


@dataclass(frozen=True, slots=True)
class CohortV2MacroStepResult:
    step: int
    pair: PredictionPair
    total_loss: float
    carrier_loss: float
    micro_loss: float
    macro_loss: float
    endpoint_count: int
    learning_rate: float


class CohortV2MacroTrainer:
    def __init__(self, data: CohortV2MacroTrainingData, config: CohortV2MacroConfig) -> None:
        self.data = data
        self.config = config
        self.device = torch.device(config.device)
        self.codec = CohortV2StateCodec(
            latent_dim=config.latent_dim, max_entities=config.max_entities
        )
        self.predictor = DualOutputPredictor(config.predictor_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.step_count = 0
        self.pair_counts = {pair: 0 for pair in MACRO_PAIRS}

    def _learning_rate(self, step: int) -> float:
        progress = min(step, self.config.steps - 1) / max(1, self.config.steps - 1)
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train_step(self) -> CohortV2MacroStepResult:
        step = self.step_count
        pair = self.data.schedule_at(step)
        windows = self.data.batch_at(pair, step)
        contexts = self.codec.batch(tuple(window.context for window in windows)).to(self.device)
        targets = self.codec.batch(tuple(window.target for window in windows)).to(self.device)
        actions = torch.stack(tuple(cohort_v2_action(window) for window in windows)).to(self.device)
        weights = self.data.duration_weights(windows).to(self.device)
        request = build_cohort_v2_transition_request(pair, windows)
        carrier = self.predictor.carrier(contexts, actions, request)
        carrier_per_example = (carrier - targets).pow(2).mean(dim=1)
        carrier_loss = (carrier_per_example * weights).sum() / weights.sum()
        micro_loss = carrier.sum() * 0.0
        macro_loss = carrier.sum() * 0.0
        endpoint_count = 0
        if pair.abstraction is Abstraction.MICRO:
            relation = micro_relation_loss(
                self.predictor.micro_head,
                carrier,
                tuple(window.target for window in windows),
                weights=weights,
            )
            predicate = micro_predicate_loss(
                self.predictor.micro_head,
                carrier,
                tuple(window.target for window in windows),
                weights=weights,
            )
            micro_loss = (relation.loss + predicate.loss) / 2.0
        elif pair.abstraction is Abstraction.MACRO:
            macro = macro_readout_loss(
                self.predictor.macro_head, carrier, windows, weights=weights
            )
            macro_loss = macro.loss
            endpoint_count = macro.endpoint_count
        symbolic = micro_loss + macro_loss
        total = carrier_loss + self.config.symbolic_weight * symbolic
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), self.config.grad_clip)
        learning_rate = self._learning_rate(step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.step()
        self.step_count += 1
        self.pair_counts[pair] += 1
        return CohortV2MacroStepResult(
            step=step,
            pair=pair,
            total_loss=float(total.detach()),
            carrier_loss=float(carrier_loss.detach()),
            micro_loss=float(micro_loss.detach()),
            macro_loss=float(macro_loss.detach()),
            endpoint_count=endpoint_count,
            learning_rate=learning_rate,
        )


@dataclass(frozen=True, slots=True)
class CohortV2MacroCheckpoint:
    path: Path
    identity: str
    step: int
    pair_counts: tuple[tuple[str, int], ...]


def _checkpoint_identity(
    reader: CohortV2ReleaseReader,
    config: CohortV2MacroConfig,
    codec: CohortV2StateCodec,
    model_state_identity: str,
    step: int,
    pair_counts: tuple[tuple[str, int], ...],
) -> str:
    return identity((
        "cohort-v2-macro-checkpoint-v1",
        reader.release_identity,
        reader.partition_identity,
        CAPABILITY_DECLARATION_IDENTITY,
        config.identity,
        config.predictor_config.identity,
        codec.identity,
        model_state_identity,
        step,
        pair_counts,
        tuple(sorted(MACRO_CAPABILITIES)),
        MICRO_RELATION_AUTHORITY,
        MACRO_STATE_AUTHORITY,
        MACRO_EVENT_ENDPOINT_AUTHORITY,
        "training",
    ))


def save_cohort_v2_macro_checkpoint(
    path: Path, trainer: CohortV2MacroTrainer
) -> CohortV2MacroCheckpoint:
    counts = tuple(
        (f"horizon={pair.delta},abstraction={pair.abstraction}", trainer.pair_counts[pair])
        for pair in MACRO_PAIRS
    )
    model_state = trainer.predictor.state_dict()
    model_state_identity = cohort_v2_model_state_identity(model_state)
    checkpoint_identity = _checkpoint_identity(
        trainer.data.reader,
        trainer.config,
        trainer.codec,
        model_state_identity,
        trainer.step_count,
        counts,
    )
    payload = {
        "capabilities": list(sorted(MACRO_CAPABILITIES)),
        "capability_declaration_identity": CAPABILITY_DECLARATION_IDENTITY,
        "checkpoint_identity": checkpoint_identity,
        "codec_identity": trainer.codec.identity,
        "config_identity": trainer.config.identity,
        "exposure_role": "training",
        "macro_event_endpoint_authority": MACRO_EVENT_ENDPOINT_AUTHORITY,
        "macro_state_authority": MACRO_STATE_AUTHORITY,
        "micro_relation_authority": MICRO_RELATION_AUTHORITY,
        "model_state": model_state,
        "model_state_identity": model_state_identity,
        "pair_counts": dict(counts),
        "partition_identity": trainer.data.reader.partition_identity,
        "predictor_config_identity": trainer.config.predictor_config.identity,
        "release_identity": trainer.data.reader.release_identity,
        "schema": MACRO_CHECKPOINT_SCHEMA,
        "step": trainer.step_count,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return CohortV2MacroCheckpoint(target, checkpoint_identity, trainer.step_count, counts)


def load_cohort_v2_macro_checkpoint(
    path: Path,
    *,
    reader: CohortV2ReleaseReader,
    config: CohortV2MacroConfig,
    device: str | None = None,
) -> tuple[DualOutputPredictor, CohortV2StateCodec, CohortV2MacroCheckpoint]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise CohortV2MacroError(f"cannot load macro checkpoint: {error}") from error
    required = {
        "capabilities", "capability_declaration_identity", "checkpoint_identity",
        "codec_identity", "config_identity", "exposure_role",
        "macro_event_endpoint_authority", "macro_state_authority", "model_state",
        "micro_relation_authority", "model_state_identity", "pair_counts", "partition_identity",
        "predictor_config_identity", "release_identity", "schema", "step",
    }
    if type(payload) is not dict or set(payload) != required:
        raise CohortV2MacroError("macro checkpoint envelope is malformed")
    codec = CohortV2StateCodec(
        latent_dim=config.latent_dim, max_entities=config.max_entities
    )
    pair_counts_value = payload["pair_counts"]
    if type(pair_counts_value) is not dict:
        raise CohortV2MacroError("macro checkpoint pair counts are malformed")
    pair_counts = tuple(
        (
            f"horizon={pair.delta},abstraction={pair.abstraction}",
            pair_counts_value.get(f"horizon={pair.delta},abstraction={pair.abstraction}"),
        )
        for pair in MACRO_PAIRS
    )
    step = payload["step"]
    if not isinstance(payload["model_state"], Mapping):
        raise CohortV2MacroError("macro checkpoint model state is malformed")
    model_state_identity = cohort_v2_model_state_identity(payload["model_state"])
    expected_identity = _checkpoint_identity(
        reader, config, codec, model_state_identity, step, pair_counts
    )
    if (
        payload["schema"] != MACRO_CHECKPOINT_SCHEMA
        or payload["release_identity"] != reader.release_identity
        or payload["partition_identity"] != reader.partition_identity
        or payload["capability_declaration_identity"] != CAPABILITY_DECLARATION_IDENTITY
        or payload["config_identity"] != config.identity
        or payload["predictor_config_identity"] != config.predictor_config.identity
        or payload["codec_identity"] != codec.identity
        or payload["model_state_identity"] != model_state_identity
        or payload["exposure_role"] != "training"
        or payload["capabilities"] != list(sorted(MACRO_CAPABILITIES))
        or payload["macro_state_authority"] != MACRO_STATE_AUTHORITY
        or payload["macro_event_endpoint_authority"] != MACRO_EVENT_ENDPOINT_AUTHORITY
        or payload["micro_relation_authority"] != MICRO_RELATION_AUTHORITY
        or type(step) is not int
        or step != config.steps
        or any(type(count) is not int or count < 0 for _, count in pair_counts)
        or sum(count for _, count in pair_counts) != step
        or payload["checkpoint_identity"] != expected_identity
    ):
        raise CohortV2MacroError("macro checkpoint provenance is stale or malformed")
    predictor = DualOutputPredictor(config.predictor_config)
    try:
        predictor.load_state_dict(payload["model_state"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise CohortV2MacroError(f"macro checkpoint model state is invalid: {error}") from error
    predictor.to(torch.device(device or config.device)).eval()
    return predictor, codec, CohortV2MacroCheckpoint(
        Path(path), expected_identity, step, pair_counts
    )


class CohortV2MacroPairScorer:
    """Full-grid duration-weighted carrier and selected-readout objective."""

    capabilities = MACRO_CAPABILITIES

    def __init__(
        self,
        predictor: DualOutputPredictor,
        codec: CohortV2StateCodec,
        checkpoint: CohortV2MacroCheckpoint,
        config: CohortV2MacroConfig,
        readers: tuple[CohortV2ReleaseReader, ...],
        *,
        progress_every: int = 0,
        progress_total: int | None = None,
        worker_name: str | None = None,
    ) -> None:
        self.predictor = predictor
        self.codec = codec
        self.checkpoint_identity = checkpoint.identity
        self.config = config
        self.device = next(predictor.parameters()).device
        self.frame_counts = {
            rollout.attempt_id: len(rollout.frame_records)
            for reader in readers
            for rollout in reader.rollouts
        }
        self.objective_identity = identity((
            "cohort-v2-macro-pair-objective-v1",
            "duration-weighted-carrier-mse+selected-symbolic-readout+normalized-effective-horizon",
            config.symbolic_weight,
            MICRO_RELATION_AUTHORITY,
            MACRO_STATE_AUTHORITY,
            MACRO_EVENT_ENDPOINT_AUTHORITY,
            codec.identity,
            checkpoint.identity,
        ))
        self.progress_every = progress_every
        self.progress_total = progress_total
        self.worker_name = worker_name or str(self.device)
        self.call_count = 0
        self._next_progress = progress_every

    def objective(self, window: CohortV2OracleWindow, pair: PredictionPair) -> float:
        return self.objective_batch((window,), pair)[0]

    def objective_batch(
        self,
        windows: tuple[CohortV2OracleWindow, ...],
        pair: PredictionPair,
    ) -> tuple[float, ...]:
        if type(windows) is not tuple or not windows:
            raise CohortV2MacroError("scoring batch must be nonempty")
        with torch.no_grad():
            context = self.codec.batch(tuple(window.context for window in windows)).to(self.device)
            target = self.codec.batch(tuple(window.target for window in windows)).to(self.device)
            action = torch.stack(tuple(cohort_v2_action(window) for window in windows)).to(self.device)
            request = build_cohort_v2_transition_request(pair, windows)
            carrier = self.predictor.carrier(context, action, request)
            carrier_loss = (carrier - target).pow(2).mean(dim=1)
            symbolic = torch.zeros_like(carrier_loss)
            if pair.abstraction is Abstraction.MICRO:
                relation = micro_relation_loss(
                    self.predictor.micro_head,
                    carrier,
                    tuple(window.target for window in windows),
                ).per_example
                predicate = micro_predicate_loss(
                    self.predictor.micro_head,
                    carrier,
                    tuple(window.target for window in windows),
                ).per_example
                symbolic = (relation + predicate) / 2.0
            elif pair.abstraction is Abstraction.MACRO:
                symbolic = macro_readout_loss(
                    self.predictor.macro_head, carrier, windows
                ).per_example
            duration_weight = torch.tensor(
                [
                    window.effective_horizon / (self.frame_counts[window.attempt_id] - 1)
                    for window in windows
                ],
                dtype=carrier.dtype,
                device=self.device,
            )
            values = duration_weight * (
                carrier_loss + self.config.symbolic_weight * symbolic
            )
        self.call_count += len(windows)
        if self.progress_every and self.call_count >= self._next_progress:
            suffix = f"/{self.progress_total}" if self.progress_total is not None else ""
            print(
                f"[score {self.worker_name} {self.call_count}{suffix}] latest="
                f"h{pair.delta}/{pair.abstraction} "
                f"mean_objective={float(values.mean()):.6f}",
                flush=True,
            )
            while self._next_progress <= self.call_count:
                self._next_progress += self.progress_every
        return tuple(float(value) for value in values.detach().cpu().tolist())


def _frontier_document(
    *,
    evaluation_identity: str,
    checkpoint_identity: str,
    objective_identity: str,
    grid_identity: str,
    state_set_identity: str,
    release_identity: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    fields = (
        evaluation_identity,
        checkpoint_identity,
        objective_identity,
        grid_identity,
        state_set_identity,
        tuple(
            (
                row["requested_horizon"], row["abstraction"], row["status"],
                row["available_state_count"], row["mean_objective"],
                tuple(row["unavailable_reasons"]),
            )
            for row in rows
        ),
    )
    return {
        "artifact_type": "pair_frontier_input",
        "checkpoint_identity": checkpoint_identity,
        "evaluation_identity": evaluation_identity,
        "exposure_role": "model_selection",
        "frontier_identity": identity(("cohort-v2-macro-frontier-input-v1", *fields)),
        "grid_identity": grid_identity,
        "macro_event_endpoint_authority": MACRO_EVENT_ENDPOINT_AUTHORITY,
        "macro_state_authority": MACRO_STATE_AUTHORITY,
        "micro_relation_authority": MICRO_RELATION_AUTHORITY,
        "objective_identity": objective_identity,
        "pairs": rows,
        "release_identity": release_identity,
        "schema": MACRO_FRONTIER_SCHEMA,
        "state_set_identity": state_set_identity,
    }


def _frontier_rows(
    pairs: tuple[PredictionPair, ...], records: tuple[object, ...]
) -> list[dict[str, object]]:
    rows = []
    for pair_index, pair in enumerate(pairs):
        outcomes = tuple(record.outcomes[pair_index] for record in records)
        values = tuple(float(outcome.objective) for outcome in outcomes if outcome.available)
        reasons = sorted({
            reason for outcome in outcomes for reason in outcome.unavailable_reasons
        })
        rows.append({
            "abstraction": str(pair.abstraction),
            "available_state_count": len(values),
            "mean_objective": sum(values) / len(values) if values else None,
            "requested_horizon": pair.delta,
            "status": "available" if values else "unavailable",
            "unavailable_reasons": reasons,
        })
    return rows


def _frontier_payload(result: CohortV2EvaluationResult) -> dict[str, object]:
    states = tuple(
        state for state in result.states if state.exposure_role == "model_selection"
    )
    return _frontier_document(
        evaluation_identity=result.identity,
        checkpoint_identity=result.checkpoint_identity,
        objective_identity=result.objective_identity,
        grid_identity=result.grid.identity,
        state_set_identity=result.state_set_identity,
        release_identity=result.release_identity,
        rows=_frontier_rows(result.grid.pairs, states),
    )


def write_cohort_v2_macro_frontier_input(
    path: Path, result: CohortV2EvaluationResult
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(_frontier_payload(result)))
    os.replace(temporary, target)
    return target


def validate_cohort_v2_macro_frontier_input(
    path: Path, result: CohortV2EvaluationResult
) -> None:
    try:
        actual = Path(path).read_bytes()
    except OSError as error:
        raise CohortV2MacroError(f"cannot load macro frontier input: {error}") from error
    if actual != canonical_json_bytes(_frontier_payload(result)):
        raise CohortV2MacroError("macro frontier input differs from its evaluation")


def validate_cohort_v2_macro_frontier_artifacts(
    path: Path, evaluation_root: Path
) -> None:
    try:
        actual = Path(path).read_bytes()
        manifest = json.loads((Path(evaluation_root) / "manifest.json").read_bytes())
        records = tuple(
            json.loads(line)
            for line in (Path(evaluation_root) / manifest["records"]).read_bytes().splitlines()
            if json.loads(line).get("exposure_role") == "model_selection"
        )
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise CohortV2MacroError(
            f"cannot load macro frontier source artifacts: {error}"
        ) from error
    rows = []
    for pair_index, pair in enumerate(manifest["pairs"]):
        outcomes = tuple(record["outcomes"][pair_index] for record in records)
        values = tuple(
            float(outcome["objective"])
            for outcome in outcomes
            if outcome.get("status") == "available"
        )
        reasons = sorted({
            reason
            for outcome in outcomes
            for reason in outcome.get("unavailable_reasons", ())
        })
        rows.append({
            "abstraction": pair["abstraction"],
            "available_state_count": len(values),
            "mean_objective": sum(values) / len(values) if values else None,
            "requested_horizon": pair["requested_horizon"],
            "status": "available" if values else "unavailable",
            "unavailable_reasons": reasons,
        })
    expected = canonical_json_bytes(_frontier_document(
        evaluation_identity=manifest["evaluation_identity"],
        checkpoint_identity=manifest["checkpoint_identity"],
        objective_identity=manifest["objective_identity"],
        grid_identity=manifest["grid_identity"],
        state_set_identity=manifest["state_set_identity"],
        release_identity=manifest["release_identity"],
        rows=rows,
    ))
    if actual != expected:
        raise CohortV2MacroError(
            "macro frontier input differs from persisted evaluation records"
        )


__all__ = [
    "MACRO_CAPABILITIES",
    "MACRO_EVENT_ENDPOINT_AUTHORITY",
    "MACRO_EVENT_TYPES",
    "MACRO_FRONTIER_SCHEMA",
    "MACRO_PAIRS",
    "MACRO_PREDICATES",
    "MACRO_STATE_AUTHORITY",
    "CohortV2MacroCheckpoint",
    "CohortV2MacroConfig",
    "CohortV2MacroError",
    "CohortV2MacroPairScorer",
    "CohortV2MacroStepResult",
    "CohortV2MacroTrainer",
    "CohortV2MacroTrainingData",
    "MacroReadoutLoss",
    "load_cohort_v2_macro_checkpoint",
    "macro_event_endpoint_available",
    "macro_readout_loss",
    "save_cohort_v2_macro_checkpoint",
    "validate_cohort_v2_macro_frontier_artifacts",
    "validate_cohort_v2_macro_frontier_input",
    "write_cohort_v2_macro_frontier_input",
]
