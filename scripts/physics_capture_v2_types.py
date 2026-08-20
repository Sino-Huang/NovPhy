"""Immutable public values for the prospective cohort-v2 capture contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class PhysicsCaptureV2:
    """A validated, Unity-authored non-observation capture."""

    capture_id: str
    shot_id: str
    configured_fixed_step_capture_stride: int
    source_bindings: Mapping[str, str]
    record: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PhysicsCaptureV2CapabilityReport:
    """A validated accounting report for actual exporter probe artifacts."""

    report_id: str
    provenance: Mapping[str, str]
    record: Mapping[str, object]
