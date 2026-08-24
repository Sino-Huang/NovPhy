"""Milestone 1a + 1b — the BG-NS-JEPA backbone.

The continuous latent ``z`` is the sole rollout state carrier. Symbolic content
enters only through the selected transition adapter and selected supervised
readout, so no hard symbolic decode becomes rollout state.
"""
from world_model.model.config import (
    ABSTRACTION_ORDER,
    MACRO_TRANSITION_INPUTS,
    MICRO_TRANSITION_INPUTS,
    Abstraction,
    EncoderConfig,
    JepaConfig,
    PredictionPair,
    PredictorConfig,
    abstraction_index,
    coerce_abstraction,
    identity,
)
from world_model.model.ema import EmaTargetEncoder
from world_model.model.encoder import ContextEncoder, EncoderOutput, build_encoder
from world_model.model.heads import (
    MacroReadout,
    MacroReadoutHead,
    MicroReadoutHead,
    mode_weight,
)
from world_model.model.predictor import (
    ContinuousTransitionAdapter,
    DualOutputPredictor,
    FiLMBlock,
    MacroTransitionAdapter,
    MacroTransitionBatch,
    MacroTransitionInput,
    MicroTransitionAdapter,
    MicroTransitionBatch,
    MicroTransitionInput,
    PairConditioner,
    PredictorOutput,
    TransitionRequest,
)
from world_model.model.jepa import JepaBackbone

__all__ = [
    "ABSTRACTION_ORDER",
    "Abstraction",
    "ContextEncoder",
    "ContinuousTransitionAdapter",
    "DualOutputPredictor",
    "EmaTargetEncoder",
    "EncoderConfig",
    "EncoderOutput",
    "FiLMBlock",
    "JepaBackbone",
    "JepaConfig",
    "MACRO_TRANSITION_INPUTS",
    "MICRO_TRANSITION_INPUTS",
    "MacroTransitionAdapter",
    "MacroTransitionBatch",
    "MacroTransitionInput",
    "MacroReadout",
    "MacroReadoutHead",
    "MicroReadoutHead",
    "MicroTransitionAdapter",
    "MicroTransitionBatch",
    "MicroTransitionInput",
    "PairConditioner",
    "PredictionPair",
    "PredictorConfig",
    "PredictorOutput",
    "TransitionRequest",
    "abstraction_index",
    "build_encoder",
    "coerce_abstraction",
    "identity",
    "mode_weight",
]
