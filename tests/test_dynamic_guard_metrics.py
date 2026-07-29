from __future__ import annotations

import unittest

from evals.dynamic.guard_trace import aggregate_guard_judgments


class DynamicGuardMetricsTest(unittest.TestCase):
    def test_aggregates_openai_and_input_output_usage(self) -> None:
        metrics = aggregate_guard_judgments([
            {
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                "inference_latency_ms": 12.5,
            },
            {
                "usage": {"input_tokens": 7, "output_tokens": 2},
                "inference_latency_ms": 4.0,
            },
        ])

        self.assertEqual(metrics["guard_calls"], 2)
        self.assertEqual(metrics["guard_prompt_tokens"], 17)
        self.assertEqual(metrics["guard_completion_tokens"], 5)
        self.assertEqual(metrics["guard_total_tokens"], 22)
        self.assertEqual(metrics["guard_inference_latency_ms_total"], 16.5)


if __name__ == "__main__":
    unittest.main()
