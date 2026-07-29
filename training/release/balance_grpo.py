"""Framework-independent Balance-GRPO advantage reweighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BalanceGRPOConfig:
    target_gap: float = 0.0
    lambda_: float = 2.0
    gap_clip: float = 0.5
    deadband: float = 0.02
    smoothing_alpha: float = 1.0
    class_weight_min: float = 0.5
    class_weight_max: float = 2.0
    gap_weight_min: float = 0.5
    gap_weight_max: float = 2.0
    effective_weight_min: float = 0.75
    effective_weight_max: float = 1.5


@dataclass(frozen=True)
class BalanceGRPOStats:
    safe_accuracy: float
    unsafe_accuracy: float
    gap: float
    class_weights: tuple[float, ...]
    gap_weights: tuple[float, ...]
    effective_weights: tuple[float, ...]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def reweight_advantages(
    advantages: Sequence[float],
    gold_labels: Sequence[str],
    predicted_labels: Sequence[str],
    config: BalanceGRPOConfig = BalanceGRPOConfig(),
) -> tuple[list[float], BalanceGRPOStats]:
    """Return reweighted advantages and rollout-batch statistics.

    `advantages` must already be normalized within each GRPO prompt group.
    Labels must be the strings ``safe`` or ``unsafe``. The class-count term
    handles label imbalance, while the gap term upweights the class with lower
    smoothed accuracy relative to `config.target_gap`.
    """

    if not (len(advantages) == len(gold_labels) == len(predicted_labels)):
        raise ValueError("advantages, gold_labels, and predicted_labels must have equal length")
    if not advantages:
        raise ValueError("at least one rollout sample is required")
    invalid = {label for label in [*gold_labels, *predicted_labels] if label not in {"safe", "unsafe"}}
    if invalid:
        raise ValueError(f"labels must be safe/unsafe, found {sorted(invalid)}")

    safe_total = sum(label == "safe" for label in gold_labels)
    unsafe_total = len(gold_labels) - safe_total
    safe_correct = sum(gold == pred == "safe" for gold, pred in zip(gold_labels, predicted_labels))
    unsafe_correct = sum(gold == pred == "unsafe" for gold, pred in zip(gold_labels, predicted_labels))
    alpha = config.smoothing_alpha
    safe_accuracy = (safe_correct + alpha) / (safe_total + 2.0 * alpha)
    unsafe_accuracy = (unsafe_correct + alpha) / (unsafe_total + 2.0 * alpha)
    raw_gap = unsafe_accuracy - safe_accuracy - config.target_gap
    gap = 0.0 if abs(raw_gap) < config.deadband else _clip(raw_gap, -config.gap_clip, config.gap_clip)

    class_weights: list[float] = []
    gap_weights: list[float] = []
    effective_weights: list[float] = []
    total = len(gold_labels)
    for label in gold_labels:
        count = safe_total if label == "safe" else unsafe_total
        class_weight = _clip(total / (2.0 * max(count, 1)), config.class_weight_min, config.class_weight_max)
        gap_weight = 1.0 + config.lambda_ * gap if label == "safe" else 1.0 - config.lambda_ * gap
        gap_weight = _clip(gap_weight, config.gap_weight_min, config.gap_weight_max)
        effective_weight = _clip(
            class_weight * gap_weight,
            config.effective_weight_min,
            config.effective_weight_max,
        )
        class_weights.append(class_weight)
        gap_weights.append(gap_weight)
        effective_weights.append(effective_weight)

    reweighted = [advantage * weight for advantage, weight in zip(advantages, effective_weights)]
    stats = BalanceGRPOStats(
        safe_accuracy=safe_accuracy,
        unsafe_accuracy=unsafe_accuracy,
        gap=gap,
        class_weights=tuple(class_weights),
        gap_weights=tuple(gap_weights),
        effective_weights=tuple(effective_weights),
    )
    return reweighted, stats
