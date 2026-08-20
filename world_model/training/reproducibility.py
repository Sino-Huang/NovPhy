"""Explicit runtime policy for reproducible CUDA runs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import torch

_CUBLAS_WORKSPACE_CONFIGS: Final[frozenset[str]] = frozenset({":16:8", ":4096:8"})


class ReproducibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    """Canonical CUDA execution policy bound into Phase-A provenance."""

    cublas_workspace_config: str = ":4096:8"
    deterministic_algorithms: bool = True
    matmul_allow_tf32: bool = False
    cudnn_allow_tf32: bool = False
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    def __post_init__(self) -> None:
        if self.cublas_workspace_config not in _CUBLAS_WORKSPACE_CONFIGS:
            raise ReproducibilityError("unsupported cuBLAS workspace configuration")
        if any(
            type(value) is not bool
            for value in (
                self.deterministic_algorithms,
                self.matmul_allow_tf32,
                self.cudnn_allow_tf32,
                self.cudnn_deterministic,
                self.cudnn_benchmark,
            )
        ):
            raise ReproducibilityError("runtime policy flags must be booleans")

    @property
    def canonical(self) -> dict[str, bool | str]:
        """Return the closed, JSON-safe policy recorded in run manifests."""
        return {
            "cublas_workspace_config": self.cublas_workspace_config,
            "cudnn_allow_tf32": self.cudnn_allow_tf32,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cudnn_deterministic": self.cudnn_deterministic,
            "deterministic_algorithms": self.deterministic_algorithms,
            "matmul_allow_tf32": self.matmul_allow_tf32,
        }

    @property
    def identity_fields(self) -> tuple[str | bool, ...]:
        """Return stable declared fields for the reproducibility identity."""
        return tuple(self.canonical.values())


def apply_reproducibility(config: ReproducibilityConfig) -> None:
    """Apply policy before Torch seeding or any other CUDA-context-producing work."""
    current_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and current_workspace != config.cublas_workspace_config:
        raise ReproducibilityError(
            "CUBLAS_WORKSPACE_CONFIG must be applied before CUDA initialization"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.cublas_workspace_config
    torch.use_deterministic_algorithms(config.deterministic_algorithms)
    torch.backends.cuda.matmul.allow_tf32 = config.matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = config.cudnn_allow_tf32
    torch.backends.cudnn.deterministic = config.cudnn_deterministic
    torch.backends.cudnn.benchmark = config.cudnn_benchmark


__all__ = [
    "ReproducibilityConfig",
    "ReproducibilityError",
    "apply_reproducibility",
]
