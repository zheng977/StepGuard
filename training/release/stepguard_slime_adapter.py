"""SLIME hooks for the released StepGuard GRPO recipe.

This module is loaded by SLIME's ``--custom-rm-path`` and
``--custom-reward-post-process-path`` options. It intentionally contains no
SLIME imports so it can be tested as ordinary Python code.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from typing import Any

from balance_grpo import BalanceGRPOConfig, reweight_advantages


_JUDGMENT = re.compile(r"<Judgment>\s*(safe|unsafe)\s*</Judgment>", re.IGNORECASE)
_RISK_SOURCE = re.compile(r"<RiskSource>\s*([^<]+?)\s*</RiskSource>", re.IGNORECASE)
_RISK_PRESENT = re.compile(r"<RiskSourcePresent>\s*(yes|no)\s*</RiskSourcePresent>", re.IGNORECASE)
_ANALYSIS = re.compile(r"<Analysis>.*?</Analysis>", re.IGNORECASE | re.DOTALL)
_UNSAFE_STEP = re.compile(r"<UnsafeStep>\s*[^<]+?\s*</UnsafeStep>", re.IGNORECASE)


def _field(value: Any, names: Sequence[str]) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
    return None


def _record(sample: Any) -> dict[str, Any]:
    """Collect label fields from SLIME's label and metadata containers."""
    record: dict[str, Any] = {}
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        record.update(metadata)

    label = getattr(sample, "label", None)
    if isinstance(label, dict):
        record.update(label)
    elif isinstance(label, str):
        try:
            decoded = json.loads(label)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            record.update(decoded)
        elif label.strip().lower() in {"safe", "unsafe"}:
            record.setdefault("label", label)
    return record


def _gold_label(sample: Any) -> str:
    record = _record(sample)
    value = _field(record, ("label", "gold_label", "judgment", "safety_label"))
    if isinstance(value, str) and value.strip().lower() in {"safe", "unsafe"}:
        return value.strip().lower()
    raise ValueError("RL sample is missing a safe/unsafe label in label or metadata")


def _gold_risk_source(sample: Any) -> str:
    record = _record(sample)
    value = _field(record, ("risk_source", "gold_risk_source"))
    return str(value or "none").strip().lower()


def _parse_response(text: str, requires_unsafe_step: bool = False) -> tuple[str | None, str | None, bool]:
    judgments = _JUDGMENT.findall(text)
    risk_sources = _RISK_SOURCE.findall(text)
    present = _RISK_PRESENT.findall(text)
    valid = len(judgments) == len(risk_sources) == len(present) == 1 and bool(_ANALYSIS.search(text))
    if requires_unsafe_step:
        valid = valid and len(_UNSAFE_STEP.findall(text)) == 1
    judgment = judgments[0].lower() if len(judgments) == 1 else None
    risk_source = risk_sources[0].strip().lower() if len(risk_sources) == 1 else None
    return judgment, risk_source, valid


async def reward_func(args: Any, sample: Any, **_: Any) -> float:
    """Return the paper reward for one completed SLIME rollout.

    The dataset must expose ``label`` and ``risk_source`` either directly in
    SLIME's label field or inside its JSON ``metadata`` field.
    """
    gold_label = _gold_label(sample)
    gold_risk_source = _gold_risk_source(sample)
    level = str(_record(sample).get("level", "")).strip().lower()
    prediction, predicted_risk_source, valid_format = _parse_response(
        sample.response,
        requires_unsafe_step=level == "trajectory",
    )
    judgment_correct = prediction == gold_label
    risk_correct = predicted_risk_source == gold_risk_source
    return float(valid_format) * (0.7 * float(judgment_correct) + 0.3 * float(judgment_correct and risk_correct))


def _group_normalize(raw_rewards: Sequence[float], group_size: int, use_std: bool) -> list[float]:
    if group_size < 2:
        raise ValueError("Balance-GRPO requires at least two rollouts per prompt")
    if len(raw_rewards) % group_size:
        raise ValueError("rollout samples must be contiguous complete prompt groups")

    normalized: list[float] = []
    for start in range(0, len(raw_rewards), group_size):
        group = raw_rewards[start : start + group_size]
        mean = sum(group) / group_size
        centered = [reward - mean for reward in group]
        if use_std:
            # torch.std's default correction is one, matching the pinned SLIME commit.
            variance = sum(value * value for value in centered) / (group_size - 1)
            scale = math.sqrt(variance) + 1e-6
            centered = [value / scale for value in centered]
        normalized.extend(centered)
    return normalized


def post_process_rewards(args: Any, samples: Sequence[Any], **_: Any) -> tuple[list[float], list[float]]:
    """Apply GRPO normalization followed by released Balance-GRPO weighting.

    This is the function passed to SLIME's
    ``--custom-reward-post-process-path``. SLIME invokes it before converting
    rollouts to train data, so its second return value becomes the scalar
    reward/advantage signal consumed by GRPO.
    """
    if not samples:
        raise ValueError("received an empty rollout batch")
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    group_size = int(args.n_samples_per_prompt)
    normalized = _group_normalize(
        raw_rewards,
        group_size=group_size,
        use_std=bool(getattr(args, "grpo_std_normalization", True)),
    )
    labels = [_gold_label(sample) for sample in samples]
    predictions = [
        _parse_response(sample.response, requires_unsafe_step=str(_record(sample).get("level", "")).lower() == "trajectory")[0]
        or "safe"
        for sample in samples
    ]
    config = BalanceGRPOConfig(
        target_gap=float(os.environ.get("STEPGUARD_TARGET_GAP", "0.0")),
        lambda_=float(os.environ.get("STEPGUARD_GAP_LAMBDA", "2.0")),
    )
    reweighted, stats = reweight_advantages(normalized, labels, predictions, config=config)
    # These are serializable and appear in SLIME's runtime args/log dumps.
    args.stepguard_safe_accuracy = stats.safe_accuracy
    args.stepguard_unsafe_accuracy = stats.unsafe_accuracy
    args.stepguard_accuracy_gap = stats.gap
    return raw_rewards, reweighted
