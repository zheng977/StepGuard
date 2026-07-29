from __future__ import annotations

import json
import shutil
import sys
import tempfile
from argparse import Namespace
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_ROOT = TESTS_DIR.parent / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ag_types import Judgment
from evals.base import EvalRecord
from evals.benchmarks import ATBenchProAdapter
from evals.evaluator import Evaluator
from evals.results import ResultWriter
from evals.artifacts import build_failure_record
from scripts.eval.run_eval import resolved_config_from_args


BENCHMARK_PATH = TESTS_DIR.parent / "benchmarks" / "atbench_pro" / "ATbench-Pro.json"


class _FakeGuardrail:
    def judge(self, action, history, context=None) -> Judgment:  # noqa: ANN001
        reason = f"fake_guardrail judged action={action.tool_name or 'unknown'}"
        return Judgment(
            label=1,
            confidence=1.0,
            reason=reason,
            metadata={"source": "fake_guardrail"},
        )


class _ParseFailedGuardrail:
    def judge(self, action, history, context=None) -> Judgment:  # noqa: ANN001
        return Judgment(
            label=0,
            confidence=0.0,
            reason="missing judgment tag",
            metadata={"judgment_parse_status": "parse_failed"},
        )


class _SafeUnsafeGuardrail:
    def judge(self, action, history, context=None) -> Judgment:  # noqa: ANN001
        return Judgment(
            label=1,
            confidence=1.0,
            reason="unsafe",
            metadata={"judgment_parse_status": "parsed_safe_unsafe"},
        )


class EvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_root = Path(self.temp_dir.name) / "results_test"

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root, ignore_errors=True)

    def test_atbench_pro_adapter_loads_eval_case(self) -> None:
        adapter = ATBenchProAdapter(BENCHMARK_PATH, limit=1, mode="trajectory")
        cases = adapter.load_cases()

        self.assertGreaterEqual(len(cases), 1)
        case = cases[0]
        self.assertTrue(case.case_id)
        self.assertIn(case.gold_label, (0, 1))
        self.assertTrue(case.history.user_request)

    def test_evaluator_turns_case_into_eval_record(self) -> None:
        adapter = ATBenchProAdapter(BENCHMARK_PATH, limit=1, mode="trajectory")
        cases = adapter.load_cases()
        self.assertGreater(len(cases), 0)
        case = cases[0]
        evaluator = Evaluator(_FakeGuardrail())

        record = evaluator.evaluate_case(case)

        self.assertEqual(record.case_id, case.case_id)
        self.assertEqual(record.gold_label, case.gold_label)
        self.assertEqual(record.pred_label, 1)
        self.assertEqual(record.judgment.label, 1)

    def test_evaluator_supports_concurrent_iteration(self) -> None:
        adapter = ATBenchProAdapter(BENCHMARK_PATH, limit=3, mode="trajectory")
        cases = adapter.load_cases()
        evaluator = Evaluator(_FakeGuardrail())

        results = list(evaluator.iter_evaluations(cases, concurrency=2))

        self.assertEqual(len(results), len(cases))
        self.assertTrue(all(case.case_id for case, _, _ in results))
        self.assertTrue(all(record is not None for _, record, _ in results))
        self.assertTrue(all(error is None for _, _, error in results))

    def test_evaluator_marks_parse_failures_as_non_metric_records(self) -> None:
        adapter = ATBenchProAdapter(BENCHMARK_PATH, limit=1, mode="trajectory")
        cases = adapter.load_cases()
        self.assertGreater(len(cases), 0)
        case = cases[0]
        evaluator = Evaluator(_ParseFailedGuardrail())

        record = evaluator.evaluate_case(case)

        self.assertEqual(record.status, "parse_failed")
        self.assertFalse(record.counted_in_metrics)
        self.assertEqual(record.error, "Failed to parse a valid <Judgment> block from model output.")

    def test_evaluator_counts_safe_unsafe_parses_as_success(self) -> None:
        adapter = ATBenchProAdapter(BENCHMARK_PATH, limit=1, mode="trajectory")
        cases = adapter.load_cases()
        self.assertGreater(len(cases), 0)
        case = cases[0]
        evaluator = Evaluator(_SafeUnsafeGuardrail())

        record = evaluator.evaluate_case(case)

        self.assertEqual(record.status, "success")
        self.assertTrue(record.counted_in_metrics)
        self.assertIsNone(record.error)
        self.assertEqual(record.pred_label, 1)

    def test_result_writer_creates_run_artifacts(self) -> None:
        adapter = ATBenchProAdapter(BENCHMARK_PATH, limit=1, mode="trajectory")
        cases = adapter.load_cases()
        self.assertGreater(len(cases), 0)
        case = cases[0]
        evaluator = Evaluator(_FakeGuardrail())
        record = evaluator.evaluate_case(case)

        writer = ResultWriter(
            output_root=self.output_root,
            benchmark_name=adapter.name,
            model_name="test-model",
            run_tag="unittest",
        )
        writer.write_resolved_config({"benchmark_name": adapter.name, "model_name": "test-model"})
        writer.append_event({"event_type": "run_started", "benchmark_name": adapter.name})
        writer.append_record(record)
        case_path = writer.write_case(case, record)
        summary = adapter.summarize([record])
        writer.write_summary(summary)

        run_dir = self.output_root / f"{adapter.name}_test-model_unittest"
        self.assertTrue((run_dir / "resolved_config.json").exists())
        self.assertTrue((run_dir / "results_summary.json").exists())
        self.assertTrue((run_dir / "records.jsonl").exists())
        self.assertTrue((run_dir / "events.jsonl").exists())
        self.assertTrue(case_path.exists())

        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        self.assertEqual(case_payload["case_id"], case.case_id)
        self.assertEqual(case_payload["pred_label"], 1)
        self.assertEqual(case_payload["gold_label"], case.gold_label)

    def test_result_writer_persists_failed_case_artifacts(self) -> None:
        adapter = ATBenchProAdapter(BENCHMARK_PATH, limit=1, mode="trajectory")
        cases = adapter.load_cases()
        self.assertGreater(len(cases), 0)
        case = cases[0]
        record = build_failure_record(case, status="execution_failed", error="backend boom")

        writer = ResultWriter(
            output_root=self.output_root,
            benchmark_name=adapter.name,
            model_name="test-model",
            run_tag="unittest",
        )
        case_path = writer.write_case(case, record)
        writer.append_record(record)

        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        self.assertEqual(case_payload["status"], "execution_failed")
        self.assertEqual(case_payload["error"], "backend boom")
        self.assertFalse(case_payload["counted_in_metrics"])

    def test_atbench_pro_summary_excludes_non_metric_records(self) -> None:
        adapter = ATBenchProAdapter(BENCHMARK_PATH, limit=1, mode="trajectory")
        cases = adapter.load_cases()
        self.assertGreater(len(cases), 0)
        case = cases[0]
        success_record = EvalRecord(
            case_id=case.case_id,
            gold_label=case.gold_label,
            pred_label=1,
            judgment=Judgment(label=1, confidence=1.0, reason="parsed"),
            metadata=dict(case.metadata),
        )
        failed_record = EvalRecord(
            case_id=case.case_id + "_dup",
            gold_label=case.gold_label,
            pred_label=0,
            judgment=Judgment(label=0, confidence=0.0, reason=""),
            metadata=dict(case.metadata),
            status="parse_failed",
            error="missing judgment",
            counted_in_metrics=False,
        )

        summary = adapter.summarize([success_record, failed_record])

        self.assertEqual(summary["evaluated_cases"], 1)
        self.assertEqual(summary["excluded_from_metrics_cases"], 1)

    def test_result_writer_clears_existing_run_dir(self) -> None:
        run_dir = self.output_root / "atbench_pro_test-model_unittest"
        stale_file = run_dir / "stale.txt"
        stale_file.parent.mkdir(parents=True, exist_ok=True)
        stale_file.write_text("stale", encoding="utf-8")

        writer = ResultWriter(
            output_root=self.output_root,
            benchmark_name="atbench_pro",
            model_name="test-model",
            run_tag="unittest",
        )

        self.assertTrue(writer.run_dir.exists())
        self.assertFalse(stale_file.exists())
        self.assertTrue((writer.run_dir / "events.jsonl").exists())

    def test_resolved_config_tracks_prompt_settings(self) -> None:
        args = Namespace(
            benchmark="atbench_pro_traj",
            input=str(BENCHMARK_PATH),
            output_root=str(self.output_root),
            backend="api",
            model="test-model",
            base_url="http://example.com/v1",
            api_key="test-key",
            limit=5,
            concurrency=2,
            temperature=0.0,
            max_tokens=1024,
            timeout=30,
            prompt_name="default",
            prompt_file=str(TESTS_DIR / "fixtures" / "guardrail_prompt.txt"),
            response_parser="safe_unsafe",
        )

        config = resolved_config_from_args(
            args,
            prompt_metadata={
                "prompt_name": "default",
                "prompt_file": str((TESTS_DIR / "fixtures" / "guardrail_prompt.txt").resolve()),
                "prompt_hash": "abc12345",
                "prompt_tag": "file-guardrail_prompt_abc12345",
            },
        )

        self.assertEqual(config["prompt_name"], "default")
        self.assertEqual(
            config["prompt_file"],
            str((TESTS_DIR / "fixtures" / "guardrail_prompt.txt").resolve()),
        )
        self.assertEqual(config["prompt_hash"], "abc12345")
        self.assertEqual(config["prompt_tag"], "file-guardrail_prompt_abc12345")
        self.assertEqual(config["response_parser"], "safe_unsafe")
        self.assertTrue(config["api_key_set"])


if __name__ == "__main__":
    unittest.main()
