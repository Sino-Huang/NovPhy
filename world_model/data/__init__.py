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
    AuthoritativePhysicsSidecars,
    MACRO_LABEL_SIDECAR,
    DerivedFrameLabel,
    MacroFrameLabel,
    OracleGateSpec,
    PhysicsEvent,
    PhysicsFrameSupervision,
    PhysicsSupervisionRequest,
)
from scripts.physics_relational_supervision import (
    RELATIONAL_LABEL_SIDECAR,
    RELATIONAL_LABEL_SCHEMA_VERSION,
    RELATIONAL_SUPERVISION_SIDECAR,
    RELATIONAL_SUPERVISION_SCHEMA_VERSION,
    Availability,
    ContactCitation,
    ContactRelationLabel,
    ContactTruth,
    ModelRelativeMicroRelationUsefulness,
    ModelRelativeMicroRelationUsefulnessLabel,
    PhysicalRegimeEligibility,
    PhysicalRegimeEligibilityLabel,
    RelationalAvailability,
    RelationalFrameLabel,
    RelationalLabelError,
    RelationalLabels,
    RelationalStateIdentity,
    RelationalSupervision,
    RelationalSupervisionError,
    SupportRelationLabel,
    SupportLabel,
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
    catalog_identity,
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


def __getattr__(name: str):
    if name in (
        "AgentObservation",
        "DecisionInference",
        "DecisionTargets",
        "DecisionTransition",
        "DeploymentFrameRecordSymbols",
        "DeploymentCarrierDataset",
        "DeploymentTemporalError",
        "DeploymentTrajectory",
        "DeploymentTrajectoryReader",
        "ExecutedAction",
        "TemporalCarrier",
        "TemporalObjectSlot",
        "TemporalObservationContext",
        "TemporalVisualCarrierAdapter",
        "TransitionCarriers",
        "TrajectoryLineageBinding",
        "build_transition_carriers",
    ):
        from world_model.data import deployment_temporal  # noqa: PLC0415

        return getattr(deployment_temporal, name)
    if name in (
        "CohortV2CentralFrameRecord",
        "CohortV2FinalAccessReceipt",
        "CohortV2FinalEvaluationReader",
        "CohortV2IngestionError",
        "CohortV2OracleWindow",
        "CohortV2OracleWindowDataset",
        "CohortV2ReleaseReader",
        "CohortV2Rollout",
        "CohortV2AlignedObservationReader",
    ):
        if name == "CohortV2AlignedObservationReader":
            from world_model.data import cohort_v2_aligned  # noqa: PLC0415

            return cohort_v2_aligned.CohortV2AlignedObservationReader
        from world_model.data import cohort_v2  # noqa: PLC0415

        value = getattr(cohort_v2, name)
        if isinstance(value, type):
            return value
    if name == "probe_cohort_v2_final_access":
        from world_model.data import cohort_v2  # noqa: PLC0415

        return getattr(cohort_v2, name)
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
    "AgentObservation",
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
    "CohortV2CentralFrameRecord",
    "CohortV2AlignedObservationReader",
    "CohortV2FinalAccessReceipt",
    "CohortV2FinalEvaluationReader",
    "CohortV2IngestionError",
    "CohortV2OracleWindow",
    "CohortV2OracleWindowDataset",
    "CohortV2ReleaseReader",
    "CohortV2Rollout",
    "DecisionInference",
    "DecisionTargets",
    "DecisionTransition",
    "DeploymentFrameRecordSymbols",
    "DeploymentCarrierDataset",
    "DeploymentTemporalError",
    "DeploymentTrajectory",
    "DeploymentTrajectoryReader",
    "ExecutedAction",
    "TemporalCarrier",
    "TemporalObjectSlot",
    "TemporalObservationContext",
    "TemporalVisualCarrierAdapter",
    "TransitionCarriers",
    "TrajectoryLineageBinding",
    "build_transition_carriers",
    "probe_cohort_v2_final_access",
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
    "AuthoritativePhysicsSidecars",
    "PhysicsFrameSupervision",
    "PhysicsSupervisionRequest",
    "DERIVED_LABEL_VECTOR_FIELDS",
    "DerivedFrameLabel",
    "MACRO_LABEL_SIDECAR",
    "MacroFrameLabel",
    "OracleGateSpec",
    "RELATIONAL_LABEL_SIDECAR",
    "RELATIONAL_LABEL_SCHEMA_VERSION",
    "RELATIONAL_SUPERVISION_SIDECAR",
    "RELATIONAL_SUPERVISION_SCHEMA_VERSION",
    "Availability",
    "ContactCitation",
    "ContactRelationLabel",
    "ContactTruth",
    "ModelRelativeMicroRelationUsefulness",
    "ModelRelativeMicroRelationUsefulnessLabel",
    "PhysicalRegimeEligibility",
    "PhysicalRegimeEligibilityLabel",
    "RelationalAvailability",
    "RelationalFrameLabel",
    "RelationalLabelError",
    "RelationalLabels",
    "RelationalStateIdentity",
    "RelationalSupervision",
    "RelationalSupervisionError",
    "SupportRelationLabel",
    "SupportLabel",
    "check_source_key_disjointness",
    "build_temporal_ablation_manifest",
    "compare_temporal_ablation_manifests",
    "get_temporal_ablation_preset",
    "infer_capture_contract",
    "EpochSampler",
    "catalog_identity",
]
