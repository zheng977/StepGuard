from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.benchmarks.nemotron_aiq import NemotronAIQAdapter


class NemotronAIQAdapterTest(unittest.TestCase):
    def test_loads_normalized_trace_without_label_in_context(self) -> None:
        row = {
            "case_id": "nemotron_case_001",
            "user_request": "Summarize this research topic.",
            "events": [
                {"kind": "TOOL", "name": "search", "input": "query", "output": "result"},
                {"kind": "LLM", "name": "assistant", "input": "context", "output": "final answer"},
            ],
            "gold_label": 1,
            "metadata": {"source_split": "safety_data/with_defense", "sample_bucket": "successful_attack_unsafe"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            case = NemotronAIQAdapter(path).load_cases()[0]

        self.assertEqual(case.gold_label, 1)
        self.assertEqual(case.history.user_request, row["user_request"])
        self.assertEqual(len(case.history.steps), 2)
        self.assertEqual(case.history.steps[0].tool_name, "search")
        self.assertIn("[LLM] assistant", case.action.raw_text)
        self.assertEqual(case.context.available_tools[0]["name"], "search")
        self.assertEqual(case.context.memory, None)
        self.assertNotIn("sample_bucket", case.history.model_dump_json())


if __name__ == "__main__":
    unittest.main()
