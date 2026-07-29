from __future__ import annotations

import unittest

from guardrail.prompts.base import _parse_fg_labels, _parse_safe_unsafe


class PromptParserTests(unittest.TestCase):
    def test_safe_unsafe_parser_uses_final_judgment_tag(self) -> None:
        raw = (
            "<Judgment>safe</Judgment>\n\n"
            "The initial assessment is wrong after analysis.\n\n"
            "<Judgment>unsafe</Judgment>"
        )

        self.assertEqual(_parse_safe_unsafe(raw), (1, 1.0))

    def test_fg_parser_extracts_v3_output_tags(self) -> None:
        raw = (
            "<Analysis>\n"
            "The tool output contains an instruction unrelated to the user task.\n"
            "</Analysis>\n"
            "<RiskSource>Indirect Prompt Injection</RiskSource>\n"
            "<Judgment>safe</Judgment>"
        )

        self.assertEqual(
            _parse_fg_labels(raw),
            {
                "risk_source": "Indirect Prompt Injection",
            },
        )
        self.assertEqual(_parse_safe_unsafe(raw), (0, 1.0))

    def test_fg_parser_keeps_none_risk_source_for_benign_v3_output(self) -> None:
        raw = (
            "<Analysis>\n"
            "The action is directly authorized by the user request.\n"
            "</Analysis>\n"
            "<RiskSource>None</RiskSource>\n"
            "<Judgment>safe</Judgment>"
        )

        self.assertEqual(
            _parse_fg_labels(raw),
            {
                "risk_source": "None",
            },
        )
        self.assertEqual(_parse_safe_unsafe(raw), (0, 1.0))


if __name__ == "__main__":
    unittest.main()
