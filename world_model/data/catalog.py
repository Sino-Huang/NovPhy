"""Immutable snapshot catalog of validated rollout episodes.

EpisodeCatalog.build() enumerates only direct, non-symlink episode directories
beneath the requested split, validates each through the public rollout-artifact
validator, and returns an immutable snapshot.  Existing instances are unaffected
by later filesystem changes; call refresh() to build a new snapshot.

Design constraints (from Todo 2):
- Only direct, non-symlink subdirectories of root/split are considered.
- The public validator (scripts.rollout_artifacts.validate_rollout_episode) is
  the sole acceptance predicate; its semantics are NOT duplicated here.
- When a collection plan is supplied, each accepted episode that appears in the
  plan's ``selected`` list for the requested split receives its exact
  ``relative_path`` (source-level key) from the plan record.  Episodes absent
  from the plan are accepted but carry source_level_key=None.
- When no plan is supplied, provenance_available=False and source_level_key is
  always None.  The catalog never invents a key from directory names.
- A requested capability that is not declared in capture_contract raises
  ValueError immediately (fail-closed).
- An unknown or currently-unsupported capture contract (e.g. physics_capture_v1)
  results in a rejection, not silent fallback.
- root must not be mutated; no caches are written.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Collection, TypeVar

from world_model.data.catalog_errors import (
    DuplicateSourceKeyError,
    RequiredCapabilityError,
)
from world_model.data.catalog_plan import (
    episode_contract_from_manifest,
    load_plan_resolution,
    make_validation_contract,
)
from world_model.data.catalog_records import build_episode_record
from world_model.data.catalog_provenance import check_source_key_disjointness
from world_model.data.types import (
    CaptureContractDescriptor,
    ContractValueError,
    EpisodeRecord,
)

_AttributeValue = TypeVar("_AttributeValue")


class EpisodeCatalog:
    """Immutable snapshot of validated rollout episodes for a single split.

    Construct via EpisodeCatalog.build().  The episodes tuple is fixed at
    construction; call refresh() to re-scan and obtain a new instance.
    """

    __slots__ = (
        "_root",
        "_split",
        "_capture_contract",
        "_required_capabilities",
        "_plan_path",
        "_episodes",
        "_rejection_count",
        "_rejection_code_counts",
        "_provenance_available",
        "_cohort_identity",
        "_collection_plan_identity",
    )

    def __init__(
        self,
        *,
        root: Path,
        split: str,
        capture_contract: CaptureContractDescriptor,
        required_capabilities: tuple[str, ...],
        plan_path: Path | None,
        episodes: tuple[EpisodeRecord, ...],
        rejection_count: int,
        rejection_code_counts: dict[str, int],
        provenance_available: bool,
        cohort_identity: str | None = None,
        collection_plan_identity: str | None = None,
    ) -> None:
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_split", split)
        object.__setattr__(self, "_capture_contract", capture_contract)
        object.__setattr__(self, "_required_capabilities", required_capabilities)
        object.__setattr__(self, "_plan_path", plan_path)
        object.__setattr__(self, "_episodes", episodes)
        object.__setattr__(self, "_rejection_count", rejection_count)
        object.__setattr__(self, "_rejection_code_counts", dict(rejection_code_counts))
        object.__setattr__(self, "_provenance_available", provenance_available)
        object.__setattr__(
            self,
            "_cohort_identity",
            cohort_identity or f"rollout-cohort-v1:{root.name or 'root'}",
        )
        object.__setattr__(
            self,
            "_collection_plan_identity",
            collection_plan_identity
            or (
                "collection-plan-v1:direct-catalog-source"
                if plan_path is None
                else f"collection-plan-v1:{plan_path.name}"
            ),
        )

    def __setattr__(self, name: str, value: _AttributeValue) -> None:
        raise AttributeError("EpisodeCatalog is immutable")

    @property
    def episodes(self) -> tuple[EpisodeRecord, ...]:
        return object.__getattribute__(self, "_episodes")

    @property
    def rejection_count(self) -> int:
        return object.__getattribute__(self, "_rejection_count")

    @property
    def rejection_codes(self) -> tuple[str, ...]:
        return tuple(sorted(object.__getattribute__(self, "_rejection_code_counts")))

    @property
    def rejection_code_counts(self) -> dict[str, int]:
        """Return a copy of typed artifact-rejection counts."""
        return dict(object.__getattribute__(self, "_rejection_code_counts"))

    @property
    def provenance_available(self) -> bool:
        return object.__getattribute__(self, "_provenance_available")

    @property
    def split(self) -> str:
        return object.__getattribute__(self, "_split")

    @property
    def capture_contract(self) -> CaptureContractDescriptor:
        return object.__getattribute__(self, "_capture_contract")

    @property
    def cohort_identity(self) -> str:
        return object.__getattribute__(self, "_cohort_identity")

    @property
    def collection_plan_identity(self) -> str:
        return object.__getattribute__(self, "_collection_plan_identity")

    @property
    def dataset_root(self) -> Path:
        """Absolute/declared filesystem provenance, excluded from semantic identity."""
        return object.__getattribute__(self, "_root").resolve(strict=False)

    @property
    def collection_plan_path(self) -> Path | None:
        """Filesystem provenance for the plan artifact, excluded from identity."""
        path = object.__getattribute__(self, "_plan_path")
        return None if path is None else path.resolve(strict=False)

    def refresh(self) -> "EpisodeCatalog":
        """Return a new catalog built from the current filesystem state."""
        return EpisodeCatalog.build(
            root=object.__getattribute__(self, "_root"),
            split=object.__getattribute__(self, "_split"),
            capture_contract=object.__getattribute__(self, "_capture_contract"),
            required_capabilities=object.__getattribute__(self, "_required_capabilities"),
            collection_plan=object.__getattribute__(self, "_plan_path"),
        )

    @staticmethod
    def from_records(
        root: Path,
        split: str,
        episodes: tuple[EpisodeRecord, ...],
        capture_contract: CaptureContractDescriptor,
        required_capabilities: Collection[str] = (),
    ) -> "EpisodeCatalog":
        """Build a snapshot from validated immutable episode records."""
        if type(episodes) is not tuple or not episodes:
            raise ContractValueError("episodes", "must be a nonempty immutable tuple")
        caps = tuple(required_capabilities)
        declared = set(capture_contract.declared_capabilities)
        missing = tuple(capability for capability in caps if capability not in declared)
        if missing:
            raise RequiredCapabilityError(
                capability=missing[0],
                contract_name=capture_contract.contract_name,
            )

        resolved_root = root.resolve()
        names: set[str] = set()
        relative_paths: set[str] = set()
        for episode in episodes:
            if type(episode) is not EpisodeRecord:
                raise ContractValueError("episodes", "must contain EpisodeRecord values")
            path = (resolved_root / episode.relative_path).resolve()
            if (
                str(episode.split) != split
                or episode.capture_contract != capture_contract
                or episode.name in names
                or episode.relative_path in relative_paths
                or resolved_root not in path.parents
                or not path.is_dir()
            ):
                raise ContractValueError(
                    "episodes", "records are duplicated, mismatched, or outside the catalog"
                )
            names.add(episode.name)
            relative_paths.add(episode.relative_path)

        return EpisodeCatalog(
            root=resolved_root,
            split=split,
            capture_contract=capture_contract,
            required_capabilities=caps,
            plan_path=None,
            episodes=episodes,
            rejection_count=0,
            rejection_code_counts={},
            provenance_available=False,
            cohort_identity=f"rollout-cohort-v1:{resolved_root.name or 'root'}",
            collection_plan_identity="collection-plan-v1:direct-catalog-source",
        )

    @staticmethod
    def build(
        root: Path,
        split: str,
        capture_contract: CaptureContractDescriptor,
        required_capabilities: Collection[str] = (),
        collection_plan: Path | None = None,
    ) -> "EpisodeCatalog":
        """Build an immutable snapshot catalog.

        Parameters
        ----------
        root:
            The dataset output root (e.g. ``data/novphy_rollouts_dataset_…``).
            The split directory is ``root / split``.
        split:
            One of ``"train"``, ``"dev"``, or ``"test"``.
        capture_contract:
            The expected capture contract descriptor.  Every validated episode
            must carry this exact contract; others are rejected.  Fail-closed:
            unknown or unsupported contracts are counted as rejections.
        required_capabilities:
            Capability names that must be declared in capture_contract.  If any
            capability is absent from capture_contract.declared_capabilities,
            ValueError is raised immediately before any episode is enumerated.
        collection_plan:
            Optional path to a ``collection_plan.json`` artifact produced by
            ``scripts.prepare_rollout_dataset``.  When supplied, plan records
            for the requested split are used to attach the exact source-level
            ``relative_path`` as ``source_level_key``.  Episodes not in the
            plan are accepted but carry ``source_level_key=None``.
        """
        # Validate requested capabilities against the contract (fail-closed).
        caps = tuple(required_capabilities)
        declared = set(capture_contract.declared_capabilities)
        for cap in caps:
            if cap not in declared:
                raise RequiredCapabilityError(
                    capability=cap,
                    contract_name=capture_contract.contract_name,
                )

        split_root = root / split
        episodes: list[EpisodeRecord] = []
        rejection_count = 0
        rejection_code_counts: dict[str, int] = {}

        plan_source_keys: dict[str, str] | None = None
        plan_val_contract = None
        provenance_available = False
        collection_plan_identity = "collection-plan-v1:direct-catalog-source"

        if collection_plan is not None:
            resolution = load_plan_resolution(collection_plan, split, root)
            if resolution is not None:
                plan_source_keys = resolution.source_keys
                collection_plan_identity = resolution.identity
                plan_val_contract = make_validation_contract(
                    resolution.count,
                    resolution.fps,
                    resolution.duration_seconds,
                )
                provenance_available = True

        if not split_root.is_dir():
            return EpisodeCatalog(
                root=root,
                split=split,
                capture_contract=capture_contract,
                required_capabilities=caps,
                plan_path=collection_plan,
                episodes=tuple(episodes),
                rejection_count=rejection_count,
                rejection_code_counts=rejection_code_counts,
                provenance_available=provenance_available,
                cohort_identity=f"rollout-cohort-v1:{root.name or 'root'}",
                collection_plan_identity=collection_plan_identity,
            )

        # Enumerate direct, non-symlink subdirectories only.
        candidates: list[Path] = []
        for child in split_root.iterdir():
            if child.is_symlink():
                continue
            if not child.is_dir():
                continue
            candidates.append(child)
        candidates.sort()

        from scripts.rollout_artifacts import EpisodeAccepted, validate_rollout_episode  # noqa: PLC0415

        validation_results = ()
        if candidates:
            with ThreadPoolExecutor(max_workers=min(32, len(candidates))) as executor:
                if plan_val_contract is not None:
                    validation_contracts = (plan_val_contract,) * len(candidates)
                else:
                    validation_contracts = tuple(
                        executor.map(episode_contract_from_manifest, candidates)
                    )
                if capture_contract.contract_name == "physics_capture_v1":
                    validate = partial(
                        validate_rollout_episode,
                        capture_contract="physics_capture_v1",
                    )
                else:
                    validate = validate_rollout_episode
                validation_results = tuple(executor.map(validate, candidates, validation_contracts))

        for candidate, result in zip(candidates, validation_results):
            if not isinstance(result, EpisodeAccepted):
                rejection_count += 1
                code = str(result.code)
                rejection_code_counts[code] = rejection_code_counts.get(code, 0) + 1
                continue

            validated_ep = result.episode

            # Determine source key.
            source_level_key: str | None = None
            if plan_source_keys is not None:
                resolved_key = str(candidate.resolve(strict=False))
                source_level_key = plan_source_keys.get(resolved_key)

            # Build the immutable EpisodeRecord.
            try:
                record = build_episode_record(
                    validated_ep,
                    name=candidate.name,
                    split=split,
                    root=root,
                    source_level_key=source_level_key,
                )
            except (ContractValueError, ValueError):
                rejection_count += 1
                rejection_code_counts["invalid_episode_record"] = (
                    rejection_code_counts.get("invalid_episode_record", 0) + 1
                )
                continue

            episodes.append(record)

        return EpisodeCatalog(
            root=root,
            split=split,
            capture_contract=capture_contract,
            required_capabilities=caps,
            plan_path=collection_plan,
            episodes=tuple(episodes),
            rejection_count=rejection_count,
            rejection_code_counts=rejection_code_counts,
            provenance_available=provenance_available,
            cohort_identity=f"rollout-cohort-v1:{root.name or 'root'}",
            collection_plan_identity=collection_plan_identity,
        )
