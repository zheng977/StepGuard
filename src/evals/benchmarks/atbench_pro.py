"""Adapter for ATbench-Pro benchmark.

ATbench-Pro has the same structure as AT-Bench but with slightly different
field names:
  - "contents" instead of "content"
  - "harm_type" instead of "real_world_harm"
  - "id" instead of "conv_id"
  - "source" field (safe/unsafe/false_refusal/benign/long_html/long_context)

We normalize to AT-Bench format and reuse ATBenchAdapter logic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .at_bench import ATBenchAdapter


class ATBenchProAdapter(ATBenchAdapter):
    """Adapter for ATbench-Pro (1000 samples, 497 unsafe + 503 safe)."""

    name = "atbench_pro"

    def _load_raw_samples(self) -> list[dict[str, Any]]:
        raw = self.input_path.read_text(encoding="utf-8")
        fixed = re.sub(r",(\s*[}\]])", r"\1", raw)
        payload = json.loads(fixed)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list: {self.input_path}")

        # Normalize field names to AT-Bench format
        normalized = []
        for item in payload:
            item["content"] = item.pop("contents", item.get("content"))
            item["conv_id"] = item.get("id", item.get("conv_id"))
            if "harm_type" in item and "real_world_harm" not in item:
                item["real_world_harm"] = item["harm_type"]
            normalized.append(item)

        if self.limit is not None:
            return normalized[: self.limit]
        return normalized
