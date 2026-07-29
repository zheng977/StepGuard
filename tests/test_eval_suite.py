from __future__ import annotations

import tempfile
import unittest
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.suite_config import load_eval_suite_config
from scripts.eval.run_eval_suite import run_eval_suite


class EvalSuiteConfigTests(unittest.TestCase):
    def test_load_eval_suite_defaults_to_stepguard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "suite.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: default_prompt_suite",
                        "benchmarks:",
                        "  - benchmark: ts_bench",
                        "    input: benchmarks/ts_bench",
                        "api_models:",
                        "  - name: api-one",
                        "    model: test-model",
                        "    base_url: http://example.test/v1",
                        "    api_key: EMPTY",
                    ]
                ),
                encoding="utf-8",
            )
            suite = load_eval_suite_config(config_path)

        self.assertEqual(suite.benchmarks[0].prompt_name, "stepguard")
        self.assertEqual(suite.benchmarks[0].response_parser, "stepguard")

    def test_load_eval_suite_config_applies_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            config_path = root / "suite.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: smoke_suite",
                        f"output_root: {root / 'results'}",
                        "defaults:",
                        "  prompt_name: stepguard",
                        "  response_parser: stepguard",
                        "  concurrency: 12",
                        "  max_tokens: 2048",
                        "benchmarks:",
                        "  - name: ts_all",
                        "    benchmark: ts_bench",
                        "    input: benchmarks/ts_bench",
                        "    ts_subset: all",
                        "  - benchmark: rjudge",
                        "    input: benchmarks/rjudge/test.json",
                        "    prompt_name: stepguard_traj",
                        "    response_parser: stepguard_traj",
                        "api_models:",
                        "  - name: api-one",
                        "    model: gpt-test",
                        "    base_url: http://example.test/v1",
                        "    api_key: EMPTY",
                        "vllm_models:",
                        "  - name: local-one",
                        "    model: local-test",
                        f"    model_path: {model_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            suite = load_eval_suite_config(config_path)

        self.assertEqual(suite.name, "smoke_suite")
        self.assertEqual([spec.name for spec in suite.benchmarks], ["ts_all", "rjudge"])
        self.assertEqual(suite.benchmarks[0].prompt_name, "stepguard")
        self.assertEqual(suite.benchmarks[0].concurrency, 12)
        self.assertEqual(suite.benchmarks[0].max_tokens, 2048)
        self.assertEqual(suite.benchmarks[1].prompt_name, "stepguard_traj")
        self.assertEqual(suite.benchmarks[1].response_parser, "stepguard_traj")

    def test_run_eval_suite_dry_run_expands_models_and_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            config_path = root / "suite.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: smoke_suite",
                        f"output_root: {root / 'results'}",
                        "defaults:",
                        "  prompt_name: stepguard",
                        "  response_parser: stepguard",
                        "  concurrency: 8",
                        "benchmarks:",
                        "  - name: ts_all",
                        "    benchmark: ts_bench",
                        "    input: benchmarks/ts_bench",
                        "  - benchmark: rjudge",
                        "    input: benchmarks/rjudge/test.json",
                        "vllm_models:",
                        "  - name: local-one",
                        "    model: local-test",
                        f"    model_path: {model_dir}",
                    ]
                ),
                encoding="utf-8",
            )
            suite = load_eval_suite_config(config_path)

            results = run_eval_suite(suite, dry_run=True)

        self.assertEqual(len(results), 2)
        self.assertEqual([result.name for result in results], ["ts_all/local-one", "rjudge/local-one"])
        self.assertTrue(all(result.status == "success" for result in results))

    def test_suite_benchmark_prompt_wins_over_global_model_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            config_path = root / "suite.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: smoke_suite",
                        f"output_root: {root / 'results'}",
                        "benchmarks:",
                        "  - name: ts_all",
                        "    benchmark: ts_bench",
                        "    input: benchmarks/ts_bench",
                        "    prompt_name: stepguard",
                        "    response_parser: stepguard",
                        "  - benchmark: rjudge",
                        "    input: benchmarks/rjudge/test.json",
                        "    prompt_name: stepguard_traj",
                        "    response_parser: stepguard_traj",
                        "vllm_models:",
                        "  - name: local-one",
                        "    model: local-test",
                        f"    model_path: {model_dir}",
                        "    prompt_name: wrong_global_prompt",
                        "    response_parser: wrong_global_parser",
                        "    prompt_name_by_benchmark:",
                        "      rjudge: model_specific_rjudge_prompt",
                    ]
                ),
                encoding="utf-8",
            )
            suite = load_eval_suite_config(config_path)
            stdout = StringIO()

            with redirect_stdout(stdout):
                run_eval_suite(suite, dry_run=True)

        output = stdout.getvalue()
        self.assertIn("--prompt-name stepguard ", output)
        self.assertIn("--response-parser stepguard ", output)
        self.assertIn("--prompt-name model_specific_rjudge_prompt", output)
        self.assertIn("--response-parser stepguard_traj ", output)
        self.assertNotIn("wrong_global_prompt", output)
        self.assertNotIn("wrong_global_parser", output)


if __name__ == "__main__":
    unittest.main()
