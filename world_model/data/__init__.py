from world_model.data.types import (
    LEGACY_RGB_V1,
    PHYSICS_CAPTURE_V1,
    PHYSICS_CAPTURE_V1_CAPABILITIES,
    CaptureContractDescriptor,
    ContractValueError,
    EpisodeRecord,
    FrameRecord,
    ReproducibilityMetadata,
    ShotAction,
    ShotRecord,
    SidecarPath,
    SplitName,
    TemporalWindowRequest,
    infer_capture_contract,
)
from world_model.data.supervision import (
    DERIVED_LABEL_VECTOR_FIELDS,
    MACRO_LABEL_SIDECAR,
    DerivedFrameLabel,
    MacroFrameLabel,
    OracleGateSpec,
    PhysicsEvent,
    PhysicsFrameSupervision,
    PhysicsSupervisionRequest,
)
from world_model.data.catalog import EpisodeCatalog, check_source_key_disjointness
from world_model.data.catalog_errors import (
    DuplicateSourceKeyError,
    RequiredCapabilityError,
)
from world_model.data.curriculum import (
    CurriculumBindingMismatchError,
    CurriculumCandidate,
    CurriculumCandidateView,
    CurriculumPolicy,
    CurriculumSchedule,
    CurriculumStage,
    CurriculumState,
    catalog_digest,
)
from world_model.data.ablations import (
    AblationRunConfig,
    ComputeBudgetMismatchError,
    TemporalAblationComparison,
    TemporalAblationManifest,
    TemporalAblationPreset,
    WindowCostRule,
    build_temporal_ablation_manifest,
    compare_temporal_ablation_manifests,
    get_temporal_ablation_preset,
)


def __getattr__(name: str) -> type:
    if name in (
        "FrameReadError",
        "NoEligibleTemporalWindowError",
        "TemporalWindowDataset",
        "WindowSample",
    ):
        from world_model.data import dataset  # noqa: PLC0415

        value = getattr(dataset, name)
        if isinstance(value, type):
            return value
    if name in ("EpochSampler", "TemporalWindowBatch", "TemporalWindowCollator"):
        from world_model.data import sampling  # noqa: PLC0415

        value = getattr(sampling, name)
        if isinstance(value, type):
            return value
    raise AttributeError(f"module {__name__!r} has no type export {name!r}")

__all__ = [
    "LEGACY_RGB_V1",
    "PHYSICS_CAPTURE_V1",
    "PHYSICS_CAPTURE_V1_CAPABILITIES",
    "CaptureContractDescriptor",
    "AblationRunConfig",
    "ComputeBudgetMismatchError",
    "ContractValueError",
    "CurriculumBindingMismatchError",
    "CurriculumCandidate",
    "CurriculumCandidateView",
    "CurriculumPolicy",
    "CurriculumSchedule",
    "CurriculumStage",
    "CurriculumState",
    "EpisodeCatalog",
    "EpisodeRecord",
    "DuplicateSourceKeyError",
    "FrameReadError",
    "FrameRecord",
    "NoEligibleTemporalWindowError",
    "ReproducibilityMetadata",
    "RequiredCapabilityError",
    "ShotAction",
    "ShotRecord",
    "SidecarPath",
    "SplitName",
    "TemporalWindowDataset",
    "TemporalWindowBatch",
    "TemporalWindowCollator",
    "TemporalWindowRequest",
    "TemporalAblationComparison",
    "TemporalAblationManifest",
    "TemporalAblationPreset",
    "WindowCostRule",
    "WindowSample",
    "PhysicsEvent",
    "PhysicsFrameSupervision",
    "PhysicsSupervisionRequest",
    "DERIVED_LABEL_VECTOR_FIELDS",
    "DerivedFrameLabel",
    "MACRO_LABEL_SIDECAR",
    "MacroFrameLabel",
    "OracleGateSpec",
    "check_source_key_disjointness",
    "build_temporal_ablation_manifest",
    "compare_temporal_ablation_manifests",
    "get_temporal_ablation_preset",
    "infer_capture_contract",
    "EpochSampler",
    "catalog_digest",
]
