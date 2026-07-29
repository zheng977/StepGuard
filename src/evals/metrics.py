"""Shared metrics utilities for benchmark adapters."""
from __future__ import annotations

from typing import Any

from evals.base import EvalRecord


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_binary_metrics(records: list[EvalRecord]) -> dict[str, Any]:
    total = len(records)
    tp = sum(1 for record in records if record.gold_label == 1 and record.pred_label == 1)
    tn = sum(1 for record in records if record.gold_label == 0 and record.pred_label == 0)
    fp = sum(1 for record in records if record.gold_label == 0 and record.pred_label == 1)
    fn = sum(1 for record in records if record.gold_label == 1 and record.pred_label == 0)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "specificity": specificity,
    }


def _paper_traj_row(metrics: dict[str, Any]) -> dict[str, Any]:
    """ACC / F1 / Recall in [0,1] plus *_pct for table copy (e.g. 75.23)."""
    acc = float(metrics["accuracy"])
    f1 = float(metrics["f1"])
    rec = float(metrics["recall"])
    return {
        "n": int(metrics["total"]),
        "accuracy": acc,
        "f1": f1,
        "recall": rec,
        "accuracy_pct": round(acc * 100, 2),
        "f1_pct": round(f1 * 100, 2),
        "recall_pct": round(rec * 100, 2),
    }
