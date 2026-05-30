from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt


class NoveltyEvaluationError(ValueError):
    """Raised when novelty evaluation inputs are invalid."""


def _ensure_real_model_supported(checkpoint_path: Path) -> None:
    checkpoint_note = (
        f" Checkpoint '{checkpoint_path}' exists but no real model-backed novelty evaluator is implemented in this MVP."
        if checkpoint_path.exists()
        else f" Checkpoint '{checkpoint_path}' does not exist."
    )
    raise NoveltyEvaluationError(
        "Real model loading is not implemented yet; pass --allow-synthetic-model for smoke evaluation or add a real evaluator."
        + checkpoint_note
    )


def _score_summary(scores: npt.NDArray[np.float32]) -> dict[str, float]:
    return {
        "min": float(np.min(scores)),
        "p10": float(np.quantile(scores, 0.1)),
        "p50": float(np.quantile(scores, 0.5)),
        "p90": float(np.quantile(scores, 0.9)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
    }


def _load_scores(path: Path) -> npt.NDArray[np.float32]:
    if not path.exists():
        raise NoveltyEvaluationError(f"Dataset '{path}' does not exist.")
    with h5py.File(path, "r") as handle:
        pixels_dataset = handle["pixels"]
        action_dataset = handle["action"]
        if not isinstance(pixels_dataset, h5py.Dataset) or not isinstance(action_dataset, h5py.Dataset):
            raise NoveltyEvaluationError("Expected 'pixels' and 'action' to be HDF5 datasets.")
        pixels = np.asarray(pixels_dataset[...], dtype=np.float32)
        action = np.asarray(action_dataset[...], dtype=np.float32)
    action_score = np.nanmean(np.square(np.nan_to_num(action, nan=0.0)), axis=1)
    pixel_score = pixels.mean(axis=(1, 2, 3)) / 255.0
    return np.asarray(action_score + pixel_score, dtype=np.float32)


def _compute_auroc(labels: npt.NDArray[np.int64], scores: npt.NDArray[np.float32]) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise NoveltyEvaluationError("Both normal and novel datasets must contain at least one score.")
    wins = 0.0
    for score in pos:
        wins += float(np.sum(score > neg))
        wins += 0.5 * float(np.sum(score == neg))
    return wins / float(len(pos) * len(neg))


def _compute_auprc(labels: npt.NDArray[np.int64], scores: npt.NDArray[np.float32]) -> float:
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / max(int(np.sum(labels == 1)), 1)
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapz(precision, recall))


def evaluate_novelty(
    *,
    normal_path: Path,
    novel_path: Path,
    checkpoint_path: Path,
    allow_synthetic_model: bool,
) -> dict[str, Any]:
    if not allow_synthetic_model:
        _ensure_real_model_supported(checkpoint_path)

    normal_scores = _load_scores(normal_path)
    novel_scores = _load_scores(novel_path)
    if len(normal_scores) == 0:
        raise NoveltyEvaluationError("Normal dataset is empty.")
    if len(novel_scores) == 0:
        raise NoveltyEvaluationError("Novel dataset is empty.")

    scores = np.concatenate([normal_scores, novel_scores])
    labels = np.concatenate([np.zeros_like(normal_scores, dtype=np.int64), np.ones_like(novel_scores, dtype=np.int64)])

    threshold = float(np.quantile(normal_scores, 0.95))
    normal_summary = _score_summary(normal_scores)
    novel_summary = _score_summary(novel_scores)
    false_positives = int(np.sum(normal_scores > threshold))
    true_positives = int(np.sum(novel_scores > threshold))
    return {
        "normal_count": int(len(normal_scores)),
        "novel_count": int(len(novel_scores)),
        "auroc": float(_compute_auroc(labels, scores)),
        "auprc": float(_compute_auprc(labels, scores)),
        "mean_normal_score": normal_summary["mean"],
        "mean_novel_score": novel_summary["mean"],
        "normal_score_min": normal_summary["min"],
        "normal_score_p10": normal_summary["p10"],
        "normal_score_p50": normal_summary["p50"],
        "normal_score_p90": normal_summary["p90"],
        "normal_score_max": normal_summary["max"],
        "novel_score_min": novel_summary["min"],
        "novel_score_p10": novel_summary["p10"],
        "novel_score_p50": novel_summary["p50"],
        "novel_score_p90": novel_summary["p90"],
        "novel_score_max": novel_summary["max"],
        "score_gap": float(novel_summary["mean"] - normal_summary["mean"]),
        "threshold": threshold,
        "false_positive_count": false_positives,
        "true_positive_count": true_positives,
        "false_positive_rate": float(false_positives / len(normal_scores)),
        "true_positive_rate": float(true_positives / len(novel_scores)),
        "checkpoint_path": str(checkpoint_path),
        "mode": "synthetic" if allow_synthetic_model else "real-model",
    }


def write_novelty_report(output_path: Path, metrics: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_novelty_monitoring_csv(output_path: Path, metrics: dict[str, Any]) -> None:
    csv_path = output_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
