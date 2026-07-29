from __future__ import annotations

import unittest

from scripts.eval.run_eval import build_parser


class RunEvalCliTests(unittest.TestCase):
    def test_build_parser_accepts_model_specific_response_parsers(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--benchmark",
                "ts_bench",
                "--model",
                "TS-Guard",
                "--backend",
                "vllm",
                "--base-url",
                "http://127.0.0.1:18000/v1",
                "--response-parser",
                "tsguard",
            ]
        )

        self.assertEqual(args.response_parser, "tsguard")

    def test_build_parser_defaults_to_stepguard_prompt_name(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["--benchmark", "atbench_pro_traj"])

        self.assertEqual(args.prompt_name, "stepguard")

    def test_build_parser_rejects_removed_benchmark(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--benchmark", "at_bench"])


if __name__ == "__main__":
    unittest.main()
