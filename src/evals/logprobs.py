"""Logprobs extraction and analysis utilities."""
from __future__ import annotations

import math
from typing import Any


def softmax_pair(lp0: float, lp1: float) -> tuple[float, float]:
    """Softmax over two logprobs, returns (p0, p1)."""
    max_lp = max(lp0, lp1)
    e0 = math.exp(lp0 - max_lp)
    e1 = math.exp(lp1 - max_lp)
    total = e0 + e1
    return e0 / total, e1 / total


def extract_binary_probs(top_logprobs: list[Any]) -> tuple[float, float, str]:
    """Extract p(safe) and p(unsafe) from top_logprobs of the first token.

    Auto-detects format by comparing probability mass:
    - "safe"/"unsafe" tokens (SFT models like AgentDoG)
    - "0"/"1" tokens (general models)

    Returns (p_safe, p_unsafe, format_used).
    """
    # Build token -> logprob map (lowercase, stripped).
    # Keep the highest logprob for each normalized key.
    lp_map: dict[str, float] = {}
    for entry in top_logprobs:
        key = entry.token.strip().lower()
        if key not in lp_map or entry.logprob > lp_map[key]:
            lp_map[key] = entry.logprob

    # Compare probability mass of both token pairs, use the dominant one.
    mass_01 = math.exp(lp_map.get("0", -100.0)) + math.exp(lp_map.get("1", -100.0))
    mass_su = math.exp(lp_map.get("safe", -100.0)) + math.exp(lp_map.get("unsafe", -100.0))

    if mass_su > mass_01:
        p_unsafe = math.exp(lp_map.get("unsafe", -100.0))
        p_safe = 1.0 - p_unsafe
        return p_safe, p_unsafe, "safe/unsafe"

    if mass_01 > 0:
        p_safe, p_unsafe = softmax_pair(lp_map.get("0", -100.0), lp_map.get("1", -100.0))
        return p_safe, p_unsafe, "0/1"

    return 0.0, 0.0, "unknown"


def binary_entropy(p_safe: float, p_unsafe: float) -> float:
    """Binary entropy in bits (max = 1.0)."""
    h = 0.0
    if p_safe > 0:
        h -= p_safe * math.log2(p_safe)
    if p_unsafe > 0:
        h -= p_unsafe * math.log2(p_unsafe)
    return h


def kl_divergence(p: tuple[float, float], q: tuple[float, float], *, eps: float = 1e-10) -> float:
    """KL(p || q) for binary distributions. p, q = (p_safe, p_unsafe)."""
    kl = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        kl += pi * math.log(pi / qi)
    return kl
