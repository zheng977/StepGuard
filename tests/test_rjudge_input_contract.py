"""Synthetic regression tests for complete R-Judge trajectories; no model calls."""

from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ag_types import Action, InteractionHistory, Observation, UserMessage
from evals.base import EvalCase
from evals.benchmarks import BENCHMARK_REGISTRY
from guardrail.prompts import PROFILE_REGISTRY
from infer.base import InferResponse


def single_group_sample() -> dict:
    return {
        "id": "synthetic-contract-case",
        "label": 1,  # Label-retention sentinel, not a safety annotation.
        "risk_description": "GOLD_RATIONALE_SENTINEL",
        "contents": [[
            {"role": "user", "content": "INITIAL_USER_SENTINEL"},
            {"role": "agent", "thought": "FIRST_THOUGHT_SENTINEL",
             "action": 'ReadDraft{"name": "FIRST_ACTION_SENTINEL"}'},
            {"role": "environment", "content": "FIRST_OBSERVATION_SENTINEL"},
        ]],
    }


def multi_group_sample() -> dict:
    raw = single_group_sample()
    for group in ("SECOND", "THIRD"):
        raw["contents"].append([
            {"role": "user", "content": f"{group}_USER_SENTINEL"},
            {"role": "agent", "thought": "",
             "action": f'UpdateDraft{{"name": "{group}_ACTION_SENTINEL"}}'},
            {"role": "environment", "content": f"{group}_OBSERVATION_SENTINEL"},
        ])
    return raw


TRAJECTORY_PROFILES = [name for name in PROFILE_REGISTRY if "_traj" in name]


def render(case: EvalCase, name: str = "stepguard_traj") -> str:
    return PROFILE_REGISTRY[name].build_prompt(
        history=case.history, action=case.action, context=case.context
    )


class RJudgeInputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.input_path = Path(self.temp_dir.name) / "test.json"

    def load(self, records: list, *, limit: int | None = None):
        self.input_path.write_text(json.dumps(records), encoding="utf-8")
        return BENCHMARK_REGISTRY["rjudge"](self.input_path, limit=limit).load_cases()

    def test_all_trajectory_profiles_preserve_every_group_in_order(self) -> None:
        case, = self.load([multi_group_sample()])
        self.assertEqual(case.case_id, "synthetic-contract-case")
        self.assertEqual(case.gold_label, 1)
        self.assertEqual(case.metadata["total_agent_turns"], 3)
        self.assertEqual(case.action.step, 3)
        markers = [
            "FIRST_THOUGHT_SENTINEL", "FIRST_ACTION_SENTINEL", "FIRST_OBSERVATION_SENTINEL",
            "SECOND_USER_SENTINEL", "SECOND_ACTION_SENTINEL", "SECOND_OBSERVATION_SENTINEL",
            "THIRD_USER_SENTINEL", "THIRD_ACTION_SENTINEL", "THIRD_OBSERVATION_SENTINEL",
        ]
        for name in TRAJECTORY_PROFILES:
            with self.subTest(profile=name):
                prompt = render(case, name)
                positions = [prompt.index(marker) for marker in markers]
                self.assertEqual(positions, sorted(positions))
                for marker in markers:
                    self.assertEqual(prompt.count(marker), 1)
                self.assertIn("[USER]: SECOND_USER_SENTINEL", prompt)
                self.assertIn("[USER]: THIRD_USER_SENTINEL", prompt)
                self.assertNotIn("GOLD_RATIONALE_SENTINEL", prompt)
        self.assertEqual(re.findall(r"\[Step (\d+)\] \[AGENT\]", render(case)), ["1", "2", "3"])

    def test_flattened_and_grouped_inputs_have_identical_cases_and_prompts(self) -> None:
        grouped = multi_group_sample()
        flattened = copy.deepcopy(grouped)
        flattened["contents"] = [[turn for group in grouped["contents"] for turn in group]]
        grouped_case, flattened_case = self.load([grouped, flattened])
        self.assertEqual(grouped_case, flattened_case)
        for name in TRAJECTORY_PROFILES:
            with self.subTest(profile=name):
                self.assertEqual(render(grouped_case, name), render(flattened_case, name))

    def test_user_turn_without_agent_in_its_group_is_preserved(self) -> None:
        raw = multi_group_sample()
        raw["contents"].insert(1, [{"role": "user", "content": "EXTRA_USER_SENTINEL"}])
        case, = self.load([raw])
        prompt = render(case)
        self.assertLess(prompt.index("EXTRA_USER_SENTINEL"), prompt.index("SECOND_USER_SENTINEL"))
        self.assertEqual(case.action.step, 3)

    def test_events_after_final_action_keep_their_order_and_roles(self) -> None:
        raw = single_group_sample()
        raw["contents"].append([
            {"role": "user", "content": "TRAILING_USER_SENTINEL"},
            {"role": "environment", "content": "TRAILING_OBSERVATION_SENTINEL"},
        ])
        case, = self.load([raw])
        self.assertEqual([type(t) for t in case.history.post_action_steps],
                         [Observation, UserMessage, Observation])
        for name in TRAJECTORY_PROFILES:
            with self.subTest(profile=name):
                prompt = render(case, name)
                markers = ["FIRST_ACTION_SENTINEL", "FIRST_OBSERVATION_SENTINEL",
                           "TRAILING_USER_SENTINEL", "TRAILING_OBSERVATION_SENTINEL"]
                positions = [prompt.index(m) for m in markers]
                self.assertEqual(positions, sorted(positions))
                self.assertIn("[USER]: TRAILING_USER_SENTINEL", prompt)

    def test_proactive_profiles_do_not_receive_post_action_events(self) -> None:
        raw = multi_group_sample()
        raw["contents"][-1].append({"role": "user", "content": "FUTURE_USER_SENTINEL"})
        case, = self.load([raw])
        for name in PROFILE_REGISTRY:
            if name in TRAJECTORY_PROFILES:
                continue
            with self.subTest(profile=name):
                prompt = render(case, name)
                self.assertIn("SECOND_USER_SENTINEL", prompt)
                self.assertIn("THIRD_USER_SENTINEL", prompt)
                self.assertNotIn("THIRD_OBSERVATION_SENTINEL", prompt)
                self.assertNotIn("FUTURE_USER_SENTINEL", prompt)

    def test_json_roundtrip_preserves_user_and_observation_types(self) -> None:
        raw = multi_group_sample()
        raw["contents"][-1].append({"role": "user", "content": "TRAILING_USER_SENTINEL"})
        case, = self.load([raw])
        restored = EvalCase.model_validate_json(case.model_dump_json())
        self.assertEqual([type(t) for t in restored.history.steps],
                         [Action, Observation, UserMessage, Action, Observation, UserMessage])
        self.assertEqual([type(t) for t in restored.history.post_action_steps],
                         [Observation, UserMessage])
        self.assertEqual(render(case), render(restored))
        self.assertEqual(restored.gold_label, 1)

    def test_legacy_history_without_new_fields_still_loads(self) -> None:
        history = InteractionHistory.model_validate({
            "user_request": "INITIAL", "steps": [
                {"tool_name": "ReadDraft", "arguments": {}, "raw_text": "read", "step": 1},
                {"content": "RESULT", "step": 1},
            ],
        })
        self.assertEqual([type(t) for t in history.steps], [Action, Observation])
        self.assertEqual(history.post_action_steps, [])

    def test_single_group_without_tail_matches_existing_history_contract(self) -> None:
        raw = single_group_sample()
        raw["contents"][0].append({"role": "agent", "thought": "", "action": "Final Answer: done"})
        case, = self.load([raw])
        expected = InteractionHistory(user_request="INITIAL_USER_SENTINEL", steps=[
            Action(tool_name="ReadDraft", arguments={"name": "FIRST_ACTION_SENTINEL"},
                   thought="FIRST_THOUGHT_SENTINEL",
                   raw_text='ReadDraft{"name": "FIRST_ACTION_SENTINEL"}', step=1),
            Observation(content="FIRST_OBSERVATION_SENTINEL", step=1),
        ])
        self.assertEqual(case.history, expected)
        self.assertEqual(case.action.raw_text, "Final Answer: done")
        self.assertEqual(case.action.step, 2)

    def test_typed_actions_without_raw_text_are_rendered_by_all_trajectory_profiles(self) -> None:
        history = InteractionHistory(user_request="INITIAL_USER_SENTINEL", steps=[
            Action(tool_name="ReadDraft", arguments={"name": "TYPED_FIRST_SENTINEL"}, step=1),
            UserMessage(content="TYPED_USER_SENTINEL"),
        ])
        case = EvalCase(case_id="typed", gold_label=0, history=history,
                        action=Action(tool_name="SaveDraft", arguments={"name": "TYPED_FINAL_SENTINEL"}, step=2))
        for name in TRAJECTORY_PROFILES:
            with self.subTest(profile=name):
                prompt = render(case, name)
                self.assertIn("ReadDraft", prompt)
                self.assertIn("SaveDraft", prompt)
                positions = [prompt.index(m) for m in
                             ["TYPED_FIRST_SENTINEL", "TYPED_USER_SENTINEL", "TYPED_FINAL_SENTINEL"]]
                self.assertEqual(positions, sorted(positions))

    def test_agent_and_guardian_history_preserve_followup_user_roles(self) -> None:
        from agent.react import ReactAgent
        from guardrail.prompts.base import _guardian_serialize_history

        case, = self.load([multi_group_sample()])
        messages = ReactAgent(Mock())._build_messages(case.history)
        self.assertIn({"role": "user", "content": "SECOND_USER_SENTINEL"}, messages)
        self.assertIn({"role": "user", "content": "THIRD_USER_SENTINEL"}, messages)
        self.assertNotIn("THIRD_OBSERVATION_SENTINEL", json.dumps(messages))
        transcript = _guardian_serialize_history(case.history)
        self.assertRegex(transcript, r"\[\d+\] user: SECOND_USER_SENTINEL")
        self.assertRegex(transcript, r"\[\d+\] user: THIRD_USER_SENTINEL")

    def test_malformed_contents_report_the_sample(self) -> None:
        valid_group = single_group_sample()["contents"][0]
        for contents in (None, {}, [], [[]], [None], [{"role": "user"}], [valid_group, []]):
            with self.subTest(contents=contents):
                raw = single_group_sample()
                raw["contents"] = contents
                with self.assertRaisesRegex(ValueError, "synthetic-contract-case"):
                    self.load([raw])

    def test_invalid_turn_in_a_later_group_is_not_silently_skipped(self) -> None:
        for bad_turn in (None, {}, {"role": "system", "content": "SYSTEM"}):
            with self.subTest(turn=bad_turn):
                raw = multi_group_sample()
                raw["contents"][-1][1] = bad_turn
                with self.assertRaisesRegex(ValueError, "synthetic-contract-case.*group 2"):
                    self.load([raw])

    def test_first_turn_must_be_user_and_sample_must_contain_agent(self) -> None:
        raw = single_group_sample()
        raw["contents"][0] = raw["contents"][0][1:]
        with self.assertRaisesRegex(ValueError, "first turn.*user"):
            self.load([raw])
        raw["contents"] = [[{"role": "user", "content": "Only a request"}]]
        with self.assertRaisesRegex(ValueError, "No agent turns"):
            self.load([raw])

    def test_non_object_sample_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample must be an object"):
            self.load([None])

    def test_sample_id_zero_labels_and_input_are_preserved(self) -> None:
        for label in (0, 1):
            raw = multi_group_sample()
            raw.update(id=0, label=label)
            original = copy.deepcopy(raw)
            case = BENCHMARK_REGISTRY["rjudge"](self.input_path)._build_case(raw)
            self.assertEqual(case.case_id, "0")
            self.assertEqual(case.gold_label, label)
            self.assertEqual(raw, original)

    def test_explicit_limit_counts_records_instead_of_groups(self) -> None:
        cases = self.load([multi_group_sample(), single_group_sample()], limit=1)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].metadata["total_agent_turns"], 3)

    def test_conv_id_fallback_when_id_is_missing_or_null(self) -> None:
        for null_id in (False, True):
            with self.subTest(null_id=null_id):
                raw = single_group_sample()
                raw.pop("id")
                if null_id:
                    raw["id"] = None
                raw["conv_id"] = "fallback-case"
                case, = self.load([raw])
                self.assertEqual(case.case_id, "fallback-case")

    def run_cli(self, records: list, backend: Mock) -> tuple[int, Path]:
        from scripts.eval.run_eval import main

        self.input_path.write_text(json.dumps(records), encoding="utf-8")
        output_root = Path(self.temp_dir.name) / "results"
        argv = [
            "run_eval.py", "--benchmark", "rjudge", "--input", str(self.input_path),
            "--model", "synthetic-guard", "--backend", "api", "--api-key", "EMPTY",
            "--base-url", "http://127.0.0.1:1/v1", "--prompt-name", "stepguard_traj",
            "--response-parser", "stepguard_traj", "--output-root", str(output_root),
        ]
        with patch.object(sys, "argv", argv), patch(
            "infer.factory.InferFactory.create", return_value=backend
        ):
            return main(), output_root

    def test_cli_evaluates_multigroup_record_once_with_the_complete_prompt(self) -> None:
        backend = Mock(model="synthetic-guard")
        backend.chat.return_value = InferResponse(text="<Judgment>unsafe</Judgment>", model=backend.model)
        result, output_root = self.run_cli([multi_group_sample()], backend)
        self.assertEqual(result, 0)
        backend.chat.assert_called_once()
        prompt = backend.chat.call_args.args[0][0]["content"]
        self.assertIn("[USER]: SECOND_USER_SENTINEL", prompt)
        self.assertIn("THIRD_OBSERVATION_SENTINEL", prompt)
        summary_path, = output_root.rglob("results_summary.json")
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["attempted_cases"], 1)
        self.assertEqual(summary["evaluated_cases"], 1)

    def test_cli_invalid_later_record_aborts_before_inference_or_scoring(self) -> None:
        backend = Mock(model="synthetic-guard")
        invalid = multi_group_sample()
        invalid["contents"][-1].append({"role": "unknown", "content": "INVALID"})
        with self.assertRaisesRegex(ValueError, "unsupported role"):
            self.run_cli([single_group_sample(), invalid], backend)
        backend.chat.assert_not_called()
        self.assertEqual(list(Path(self.temp_dir.name).rglob("results_summary.json")), [])


if __name__ == "__main__":
    unittest.main()
