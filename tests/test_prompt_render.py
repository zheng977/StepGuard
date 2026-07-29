"""Released prompt rendering tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_ROOT = TESTS_DIR.parent / "src"
BENCH_ROOT = TESTS_DIR.parent / "benchmarks"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from guardrail.prompts import PROMPT_REGISTRY, build_predictive_prompt
from evals.benchmarks.ts_bench import TSBenchAdapter


class PromptRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ts_bench_path = BENCH_ROOT / "ts_bench"
        if not ts_bench_path.exists():
            raise unittest.SkipTest("TS-Bench data not found")
        cls.case = TSBenchAdapter(ts_bench_path, subset="banking", limit=1).load_cases()[0]

    def _render(self, name: str) -> str:
        return build_predictive_prompt(
            history=self.case.history,
            action=self.case.action,
            context=self.case.context,
            prompt_name=name,
        )

    def test_stepguard_action_contract(self) -> None:
        prompt = self._render("stepguard")
        self.assertIn("<<Action Under Review>>", prompt)
        self.assertIn("<RiskSourcePresent>", prompt)
        self.assertNotIn("<UnsafeStep>", prompt)

    def test_stepguard_trajectory_contract(self) -> None:
        prompt = self._render("stepguard_traj")
        self.assertIn("<BEGIN TRAJECTORY>", prompt)
        self.assertIn("<UnsafeStep>", prompt)

    def test_released_profiles_render(self) -> None:
        for name in PROMPT_REGISTRY:
            with self.subTest(prompt=name):
                prompt = self._render(name)
                self.assertGreater(len(prompt), 50)

    def test_unknown_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._render("agentguard_lite")


if __name__ == "__main__":
    unittest.main()
