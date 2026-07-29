from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ag_types import Judgment
from evals.artifacts import finalize_static_summary
from evals.base import EvalRecord
from evals.reporting import collect_result_summary_rows, write_result_index


class EvalArtifactTests(unittest.TestCase):
    def test_finalize_static_summary_adds_failure_counters(self) -> None:
        records = [
            EvalRecord(
                case_id="ok",
                gold_label=1,
                pred_label=1,
                judgment=Judgment(label=1),
            ),
            EvalRecord(
                case_id="parse_failed",
                gold_label=0,
                pred_label=0,
                judgment=Judgment(label=0),
                status="parse_failed",
                counted_in_metrics=False,
                error="missing judgment",
            ),
        ]

        summary = finalize_static_summary({"accuracy": 1.0}, records, attempted_cases=3)

        self.assertEqual(summary["attempted_cases"], 3)
        self.assertEqual(summary["evaluated_cases"], 1)
        self.assertEqual(summary["completed_cases"], 1)
        self.assertEqual(summary["failed_cases"], 1)
        self.assertTrue(summary["excluded_failed_cases_from_metrics"])

    def test_collect_result_summary_rows_and_write_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "ts_bench_demo-model_prompt-agentguard_abcd1234"
            run_dir.mkdir()
            (run_dir / "resolved_config.json").write_text(
                json.dumps({"model": "demo-model", "prompt_name": "agentguard"}),
                encoding="utf-8",
            )
            (run_dir / "results_summary.json").write_text(
                json.dumps(
                    {
                        "accuracy": 0.8,
                        "precision": 0.7,
                        "recall": 0.6,
                        "f1": 0.65,
                        "total": 10,
                        "failed_cases": 1,
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_result_summary_rows(root)
            index_paths = write_result_index(root, rows)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].model, "demo-model")
            self.assertEqual(rows[0].prompt, "agentguard")
            self.assertIsNotNone(index_paths)
            csv_path, md_path = index_paths or (None, None)
            self.assertTrue(csv_path and csv_path.exists())
            self.assertTrue(md_path and md_path.exists())
            self.assertIn("demo-model", md_path.read_text(encoding="utf-8"))

    def test_collect_result_summary_rows_falls_back_to_run_dir_convention(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "ts_bench_demo-model_prompt-agentguard_abcd1234"
            run_dir.mkdir()
            (run_dir / "results_summary.json").write_text(
                json.dumps({"accuracy": 0.8}),
                encoding="utf-8",
            )

            rows = collect_result_summary_rows(root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].model, "demo-model")
            self.assertEqual(rows[0].prompt, "agentguard")

    def test_collect_result_summary_rows_supports_recursive_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "outer" / "ts_bench_demo-model_prompt-agentguard_abcd1234"
            run_dir.mkdir(parents=True)
            (run_dir / "results_summary.json").write_text(
                json.dumps({"accuracy": 0.8}),
                encoding="utf-8",
            )

            self.assertEqual(collect_result_summary_rows(root), [])
            self.assertEqual(len(collect_result_summary_rows(root, recursive=True)), 1)


if __name__ == "__main__":
    unittest.main()
