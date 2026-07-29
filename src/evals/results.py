from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.base import EvalCase, EvalRecord


def _normalize_name(value: str) -> str:
    return str(value).strip().replace("/", "--").replace(" ", "_")


class ResultWriter:
    def __init__(
        self,
        *,
        output_root: str | Path,
        benchmark_name: str,
        model_name: str,
        run_tag: str | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        self.run_tag = run_tag
        run_name = f"{_normalize_name(benchmark_name)}_{_normalize_name(model_name)}"
        if run_tag:
            run_name = f"{run_name}_{_normalize_name(run_tag)}"
        self.run_dir = self.output_root / run_name
        self.cases_dir = self.run_dir / "cases"
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.records_path = self.run_dir / "records.jsonl"
        self.events_path.touch(exist_ok=True)
        self.records_path.touch(exist_ok=True)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_resolved_config(self, config: dict[str, Any]) -> None:
        self._write_json(self.run_dir / "resolved_config.json", config)

    def append_event(self, event: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_case(self, case: EvalCase, record: EvalRecord) -> Path:
        path = self.cases_dir / f"{case.case_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_id": case.case_id,
            "gold_label": case.gold_label,
            "pred_label": record.pred_label,
            "status": record.status,
            "error": record.error,
            "counted_in_metrics": record.counted_in_metrics,
            "history": case.history.model_dump(),
            "action": case.action.model_dump(),
            "context": case.context.model_dump() if case.context is not None else None,
            "judgment": record.judgment.model_dump(),
            "metadata": dict(record.metadata),
        }
        self._write_json(path, payload)
        return path

    def append_record(self, record: EvalRecord) -> None:
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self.run_dir / "results_summary.json", summary)
