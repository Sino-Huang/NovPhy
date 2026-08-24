"""Milestone 1a + 1b — the BG-NS-JEPA backbone.

The continuous latent ``z`` is the sole rollout state carrier.  Everything
symbolic lives in mode-head readouts that enter only the loss, so base training
stays teacher-forced and fully differentiable.
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
    PairConditioner,
    PredictorOutput,
    SymbolicTransitionAdapter,
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
    "MacroReadout",
    "MacroReadoutHead",
    "MicroReadoutHead",
    "PairConditioner",
    "PredictionPair",
    "PredictorConfig",
    "PredictorOutput",
    "SymbolicTransitionAdapter",
    "abstraction_index",
    "build_encoder",
    "coerce_abstraction",
    "identity",
    "mode_weight",
]
