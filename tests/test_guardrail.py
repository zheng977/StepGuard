from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_ROOT = TESTS_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ag_types import Action, GuardrailContext, InteractionHistory
from guardrail.guardrail import PredictiveGuardrail
from infer.base import BaseInferBackend, InferResponse


class StubBackend(BaseInferBackend):
    def __init__(self, response_text: str) -> None:
        super().__init__(model="stub")
        self.response_text = response_text
        self.last_messages: list[dict[str, str]] | None = None

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.last_messages = messages
        return InferResponse(text=self.response_text, model=self.model)


class GuardrailSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = InteractionHistory(user_request="Complete the requested task safely.")
        self.action = Action(tool_name="send_email", arguments={"to": "user@example.com"}, step=1)
        self.context = GuardrailContext(available_tools=[{"name": "send_email", "description": "Send email."}])

    def test_stepguard_parses_tagged_judgment(self) -> None:
        backend = StubBackend("<Judgment>unsafe</Judgment>")
        guardrail = PredictiveGuardrail(backend, prompt_name="stepguard", response_parser="stepguard")
        judgment = guardrail.judge(self.action, self.history, self.context)
        self.assertEqual(judgment.label, 1)
        self.assertEqual(judgment.metadata["prompt_name"], "stepguard")
        self.assertIn("<RiskSourcePresent>", backend.last_messages[0]["content"])

    def test_native_baseline_parser_is_selectable(self) -> None:
        backend = StubBackend("[Answer] safe")
        guardrail = PredictiveGuardrail(backend, prompt_name="shieldagent", response_parser="shieldagent")
        judgment = guardrail.judge(self.action, self.history, self.context)
        self.assertEqual(judgment.label, 0)

    def test_unparseable_output_is_fail_closed(self) -> None:
        backend = StubBackend("I cannot determine the result.")
        guardrail = PredictiveGuardrail(backend, prompt_name="stepguard", response_parser="stepguard")

        judgment = guardrail.judge(self.action, self.history, self.context)

        self.assertEqual(judgment.label, 1)
        self.assertEqual(judgment.confidence, 1.0)
        self.assertEqual(judgment.metadata["judgment_parse_status"], "parse_failed_closed")

    def test_custom_prompt_uses_stepguard_serializer(self) -> None:
        backend = StubBackend("<Judgment>safe</Judgment>")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
            handle.write("Request: {user_request}\nAction: {current_action_text}")
            prompt_file = Path(handle.name)
        self.addCleanup(lambda: prompt_file.unlink(missing_ok=True))
        guardrail = PredictiveGuardrail(backend, prompt_file=prompt_file, response_parser="stepguard")
        judgment = guardrail.judge(self.action, self.history, self.context)
        self.assertEqual(judgment.label, 0)
        self.assertIn("Request:", backend.last_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
