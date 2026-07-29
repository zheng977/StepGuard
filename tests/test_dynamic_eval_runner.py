from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.dynamic.config import DynamicEvalConfig
from evals.dynamic.agentdyn import AgentDynBenchmark
from evals.dynamic.agentharm import AgentHarmBenchmark
from evals.dynamic.base import DynamicEvalResult
from evals.dynamic.factory import build_benchmark_kwargs, create_guardrail
from evals.dynamic.feedback import format_replan_feedback
from evals.dynamic.guard_trace import compact_guard_trace
from evals.dynamic.runner import run_dynamic_eval
from ag_types import Action, GuardrailContext, InteractionHistory, Judgment
from scripts.eval.run_dynamic_eval import build_parser


class DynamicEvalConfigTests(unittest.TestCase):
    def test_from_namespace_preserves_agentharm_fields(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--benchmark",
                "agentharm",
                "--no-guard",
                "--subset",
                "benign",
                "--limit",
                "5",
                "--feedback-mode",
                "self_reflect",
                "--agent-base-url",
                "http://127.0.0.1:8000/v1",
                "--judge-server-json",
                "logs/vllm_qwen3_35b/server.json",
                "--record-full-guard-context",
            ]
        )

        config = DynamicEvalConfig.from_namespace(args)

        self.assertEqual(config.benchmark, "agentharm")
        self.assertTrue(config.no_guard)
        self.assertEqual(config.subset, "benign")
        self.assertEqual(config.limit, 5)
        self.assertEqual(config.feedback_mode, "self_reflect")
        self.assertEqual(config.agent_base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(config.judge_server_json, "logs/vllm_qwen3_35b/server.json")
        self.assertTrue(config.record_full_guard_context)

    def test_run_tag_splits_agentharm_subsets(self) -> None:
        config = DynamicEvalConfig(benchmark="agentharm", subset="benign")

        self.assertEqual(config.run_tag(guard_prompt_tag="agentguard"), "agentguard_benign")

    def test_default_dynamic_protocol_is_self_reflect(self) -> None:
        config = DynamicEvalConfig(benchmark="agentdyn")

        self.assertEqual(config.feedback_mode, "self_reflect")
        self.assertEqual(config.blocked_history_mode, "clean")
        self.assertEqual(config.guard_reconsideration, "off")

    def test_resolved_config_does_not_write_api_secrets(self) -> None:
        config = DynamicEvalConfig(
            benchmark="agentdyn",
            api_key="secret-api-key",
            agent_api_key="secret-agent-key",
            judge_api_key="secret-judge-key",
        )

        payload = config.resolved_run_config(guard_name="guard")
        serialized = json.dumps(payload)

        self.assertTrue(payload["api_key_set"])
        self.assertTrue(payload["agent_api_key_set"])
        self.assertTrue(payload["judge_api_key_set"])
        self.assertNotIn("secret-api-key", serialized)
        self.assertNotIn("secret-agent-key", serialized)
        self.assertNotIn("secret-judge-key", serialized)


class DynamicEvalFactoryTests(unittest.TestCase):
    def test_create_guardrail_returns_no_defense_for_baseline(self) -> None:
        guardrail, guard_name = create_guardrail(DynamicEvalConfig(benchmark="agentdyn", no_guard=True))

        self.assertIsNone(guardrail)
        self.assertEqual(guard_name, "no_defense")

    def test_agentdyn_kwargs_preserve_vllm_port_key_fallback(self) -> None:
        config = DynamicEvalConfig(
            benchmark="agentdyn",
            api_key="guard-key",
            agent_port=18000,
            suites=["shopping"],
            attack="important_instructions",
        )

        kwargs = build_benchmark_kwargs(config)

        self.assertEqual(kwargs["agent_api_key"], "EMPTY")
        self.assertEqual(kwargs["agent_port"], 18000)
        self.assertEqual(kwargs["suites"], ["shopping"])
        self.assertEqual(kwargs["attack"], "important_instructions")

    def test_agentharm_kwargs_map_official_source_root(self) -> None:
        config = DynamicEvalConfig(
            benchmark="agentharm",
            subset="benign",
            agentharm_source_root="benchmark-repos/inspect_evals/src",
            n_irrelevant_tools=3,
        )

        kwargs = build_benchmark_kwargs(config)

        self.assertEqual(kwargs["subset"], "benign")
        self.assertEqual(kwargs["official_source_root"], "benchmark-repos/inspect_evals/src")
        self.assertEqual(kwargs["n_irrelevant_tools"], 3)

    def test_server_json_overrides_agent_and_judge_urls(self) -> None:
        config = DynamicEvalConfig(
            benchmark="agentharm",
            agent_server_json="logs/vllm_qwen3_35b/server.json",
            judge_server_json="logs/vllm_qwen3_35b/server.json",
        )

        class Resolved:
            base_url = "http://10.103.13.80:8000/v1"

        with patch("evals.dynamic.factory.resolve_vllm_base_url", return_value=Resolved()):
            kwargs = build_benchmark_kwargs(config)

        self.assertEqual(kwargs["agent_base_url"], "http://10.103.13.80:8000/v1")
        self.assertEqual(kwargs["judge_base_url"], "http://10.103.13.80:8000/v1")
        self.assertTrue(kwargs["agent_no_proxy"])
        self.assertEqual(kwargs["agent_api_key"], "EMPTY")


class AgentHarmPromptParityTests(unittest.TestCase):
    def test_agentharm_uses_official_default_system_prompt(self) -> None:
        benchmark = AgentHarmBenchmark(
            subset="benign",
            agent_model="Qwen3.6-35B-A3B",
        )
        benchmark._prepare_official_source()

        prompt = benchmark._official_agent_system_prompt()

        self.assertIn("You are a helpful LLM agent", prompt)
        self.assertIn("calls all necessary tools", prompt)

    def test_agentharm_uses_official_empty_user_template(self) -> None:
        benchmark = AgentHarmBenchmark(
            subset="benign",
            agent_model="Qwen3.6-35B-A3B",
        )
        benchmark._prepare_official_source()

        rendered = benchmark._official_agent_user_prompt("do the task")

        self.assertIn("do the task", rendered)


class DynamicEvalRunnerTests(unittest.TestCase):
    def test_runner_writes_summary_records_and_sanitized_config(self) -> None:
        class FakeBenchmark:
            name = "agentharm"

            def run(self, guardrail, **kwargs):
                self.run_kwargs = kwargs
                return []

            def to_eval_records(self, results):
                return []

            def summarize(self, results):
                return {
                    "subset": "harmful",
                    "avg_score": 0.0,
                    "avg_full_score": 0.0,
                    "harm_score": 0.0,
                    "harm_full_score": 0.0,
                    "security_rate_full": 1.0,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            config = DynamicEvalConfig(
                benchmark="agentharm",
                no_guard=True,
                output_root=temp_dir,
                api_key="secret-key",
                subset="harmful",
                feedback_mode="detailed",
            )
            fake_benchmark = FakeBenchmark()

            with (
                patch("evals.dynamic.runner.create_guardrail", return_value=(None, "no_defense")),
                patch("evals.dynamic.runner.create_dynamic_benchmark", return_value=fake_benchmark),
                patch("evals.dynamic.runner.print_dynamic_summary"),
            ):
                result = run_dynamic_eval(config)

            run_dir = Path(result.run_dir)
            resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "results_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(result.record_count, 0)
            self.assertEqual(summary["subset"], "harmful")
            self.assertEqual(fake_benchmark.run_kwargs["feedback_mode"], "detailed")
            self.assertTrue(resolved["api_key_set"])
            self.assertFalse(resolved["record_full_guard_context"])
            self.assertNotIn("secret-key", json.dumps(resolved))


class AgentDynSummaryTests(unittest.TestCase):
    def test_summary_treats_security_as_attack_resistance(self) -> None:
        benchmark = AgentDynBenchmark(
            agent_model="agent-model",
            suites=["shopping"],
            attack="important_instructions",
        )
        results = [
            DynamicEvalResult(
                case_id="shopping/user_task_0/injection_task_0",
                utility=True,
                security=True,
                has_attack=True,
                metadata={"suite": "shopping"},
            ),
            DynamicEvalResult(
                case_id="shopping/user_task_1/injection_task_0",
                utility=True,
                security=True,
                has_attack=True,
                metadata={"suite": "shopping"},
            ),
            DynamicEvalResult(
                case_id="shopping/user_task_2/injection_task_0",
                utility=True,
                security=False,
                has_attack=True,
                metadata={"suite": "shopping"},
            ),
        ]

        summary = benchmark.summarize(results)

        self.assertAlmostEqual(summary["security_rate"], 2 / 3)
        self.assertAlmostEqual(summary["asr"], 1 / 3)
        self.assertAlmostEqual(summary["by_suite"]["shopping"]["security_rate"], 2 / 3)
        self.assertAlmostEqual(summary["by_suite"]["shopping"]["asr"], 1 / 3)


class DynamicFeedbackTests(unittest.TestCase):
    def test_detailed_feedback_is_minimal_and_actionable(self) -> None:
        feedback = format_replan_feedback("detailed", confidence=0.9)

        self.assertIn("failed a safety check", feedback)
        self.assertIn("original user task", feedback)
        self.assertIn("real execution history", feedback)
        self.assertIn("different next action", feedback)
        self.assertNotIn("internal reasoning", feedback)
        self.assertNotIn("blocked operation", feedback)


class GuardTraceTests(unittest.TestCase):
    def test_full_context_trace_contains_guard_inputs_when_enabled(self) -> None:
        history = InteractionHistory(user_request="Send the invoice.")
        history.add_action(Action(tool_name="read_inbox", arguments={}, raw_text="read_inbox({})", step=0))
        context = GuardrailContext(tool_schemas=[{"name": "send_email", "description": "Send email"}])
        action = Action(
            tool_name="send_email",
            arguments={"to": "a@example.com"},
            raw_text='send_email({"to":"a@example.com"})',
            step=1,
        )
        judgment = Judgment(
            label=1,
            confidence=1.0,
            reason="<Analysis>risk</Analysis><Judgment>unsafe</Judgment>",
            metadata={
                "prompt": "FULL PROMPT",
                "messages": [{"role": "user", "content": "FULL PROMPT"}],
                "chat_kwargs": {"top_p": 1},
            },
        )

        with patch.dict(
            "os.environ",
            {"AGENTGUARD_RECORD_FULL_PROMPT": "1", "AGENTGUARD_RECORD_FULL_CONTEXT": "1"},
        ):
            trace = compact_guard_trace(
                action=action,
                judgment=judgment,
                blocked=True,
                history=history,
                context=context,
            )

        self.assertEqual(trace["user_request"], "Send the invoice.")
        self.assertIn("read_inbox", trace["history_text"])
        self.assertIn("send_email", trace["tool_list_text"])
        self.assertEqual(trace["current_action_text"], 'send_email({"to":"a@example.com"})')
        self.assertEqual(trace["guard_prompt"], "FULL PROMPT")
        self.assertEqual(trace["full_context"]["guard_raw_response"], judgment.reason)

    def test_generic_feedback_keeps_only_risk_rating(self) -> None:
        feedback = format_replan_feedback("generic", confidence=0.87)

        self.assertIn("risk rating", feedback)
        self.assertIn("0.9", feedback)
        self.assertNotIn("internal reasoning", feedback)


if __name__ == "__main__":
    unittest.main()
