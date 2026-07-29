from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from ag_types import Action, GuardrailContext, InteractionHistory, Observation
from evals.base import BaseBenchmarkAdapter, EvalCase, EvalRecord
from evals.metrics import _compute_binary_metrics, _paper_traj_row, _safe_div


def _bench_family(dataset_family: str) -> str:
    if dataset_family == "agentdojo-traj":
        return "agentdojo"
    if dataset_family == "agentharm-traj":
        return "agentharm"
    if dataset_family == "asb-traj":
        return "asb"
    return "unknown"


# Paper / ToolSafe-style trajectory splits (matches by_bench_family internal keys).
TRAJ_BENCHMARK_LABELS: tuple[tuple[str, str], ...] = (
    ("agentharm", "AgentHarm-Traj"),
    ("asb", "ASB-Traj"),
    ("agentdojo", "AgentDojo-Traj"),
)


def _build_traj_metrics_table(
    overall: dict[str, Any],
    by_bench_family_summary: dict[str, dict[str, Any]],
) -> dict[str, Any | None]:
    """Three traj subsets + Overall, for side-by-side reporting with published tables."""
    rows: dict[str, Any | None] = {}
    for internal, paper_name in TRAJ_BENCHMARK_LABELS:
        block = by_bench_family_summary.get(internal)
        if block and int(block.get("total", 0)) > 0:
            rows[paper_name] = _paper_traj_row(block)
        else:
            rows[paper_name] = None
    rows["Overall"] = _paper_traj_row(overall)
    return rows


class TSBenchAdapter(BaseBenchmarkAdapter):
    name = "ts_bench"

    DATASETS: dict[str, tuple[str, str]] = {
        "banking": ("agentdojo-traj", "banking.json"),
        "slack": ("agentdojo-traj", "slack.json"),
        "travel": ("agentdojo-traj", "travel.json"),
        "workspace": ("agentdojo-traj", "workspace.json"),
        "harmful_steps": ("agentharm-traj", "harmful_steps.json"),
        "benign_steps": ("agentharm-traj", "benign_steps.json"),
        "asb_opi_success": ("asb-traj", "test/OPI_attack_success.json"),
        "asb_dpi_success": ("asb-traj", "test/DPI_attack_success.json"),
        "asb_attack_failure": ("asb-traj", "test/atttack_failure.json"),
    }
    SUBSET_MAP: dict[str, tuple[str, ...]] = {
        "all": tuple(DATASETS.keys()),
        "agentdojo": ("banking", "slack", "travel", "workspace"),
        "agentharm": ("harmful_steps", "benign_steps"),
        "asb": ("asb_opi_success", "asb_dpi_success", "asb_attack_failure"),
        # 不含 ASB (只 AgentDojo + AgentHarm) — 后面评测不做 ASB
        "no_asb": ("banking", "slack", "travel", "workspace", "harmful_steps", "benign_steps"),
        **{name: (name,) for name in DATASETS},
    }
    _HISTORY_STEP_RE = re.compile(r"(?ms)^\(1\)\s*Thought:.*?(?=^\(1\)\s*Thought:|\Z)")
    _OBSERVATION_SPLIT_RE = re.compile(r"(?m)^Observation:\s*")

    def __init__(
        self,
        input_path: str | Path,
        *,
        subset: str = "all",
        limit: int | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.subset = str(subset).strip() or "all"
        self.limit = limit
        try:
            self.dataset_names = self.SUBSET_MAP[self.subset]
        except KeyError as exc:
            raise ValueError(
                f'Unknown TS-Bench subset "{self.subset}". '
                f'Expected one of: {", ".join(sorted(self.SUBSET_MAP))}.'
            ) from exc
        self.excluded_gold_half_cases = 0
        self.loaded_case_count = 0

    @staticmethod
    def _parse_action_text(action_text: str, *, step: int) -> Action:
        normalized = str(action_text).strip()
        thought_match = re.search(r"\(1\)\s*Thought:\s*(.*?)(?=\n\s*\(2\)\s*Action:|$)", normalized, flags=re.DOTALL)
        tool_match = re.search(r"\(2\)\s*Action:\s*([^\n]+)", normalized)
        args_match = re.search(r"\(3\)\s*Action Input:\s*(.*)$", normalized, flags=re.DOTALL)

        thought = str(thought_match.group(1)).strip() if thought_match else ""
        tool_name = str(tool_match.group(1)).strip() if tool_match else None
        arguments: dict[str, Any] = {}

        if args_match:
            args_text = str(args_match.group(1)).strip()
            fenced_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", args_text, flags=re.DOTALL | re.IGNORECASE)
            if fenced_match:
                args_text = str(fenced_match.group(1)).strip()
            try:
                parsed = json.loads(args_text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(args_text)
                except (ValueError, SyntaxError):
                    parsed = {}
            if isinstance(parsed, dict):
                arguments = parsed

        return Action(
            tool_name=tool_name,
            arguments=arguments,
            thought=thought,
            raw_text=normalized,
            step=step,
        )

    @classmethod
    def _parse_history_steps(cls, history_text: str) -> list[Action | Observation]:
        """Parse TS-Bench raw history into alternating agent/environment turns.

        TS-Bench stores prior turns as plain text blocks:
        ``(1) Thought ... (2) Action ... (3) Action Input ... Observation ...``.
        Keeping that full blob as a single environment observation makes the
        rendered AgentGuard prompt say the environment produced agent thoughts
        and actions. Preserve the original transcript structure instead.
        """
        normalized = str(history_text).strip()
        if not normalized:
            return []

        steps: list[Action | Observation] = []
        matched_step_block = False
        for action_index, match in enumerate(cls._HISTORY_STEP_RE.finditer(normalized), start=1):
            matched_step_block = True
            block = match.group(0).strip()
            if not block:
                continue

            parts = cls._OBSERVATION_SPLIT_RE.split(block, maxsplit=1)
            action_text = parts[0].strip()
            observation_text = parts[1].strip() if len(parts) > 1 else ""

            if action_text:
                steps.append(cls._parse_action_text(action_text, step=action_index))
            if observation_text:
                steps.append(Observation(content=observation_text, step=action_index))

        if matched_step_block and steps:
            return steps

        # ASB has a small number of history fields that are observation-only.
        return [Observation(content=normalized, step=0)]

    @staticmethod
    def _parse_env_info(env_info: str) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        current_tool: dict[str, Any] | None = None
        in_parameters = False

        for line in str(env_info).splitlines():
            if not line.strip():
                in_parameters = False
                continue

            stripped = line.strip()
            tool_match = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(.*)$", stripped)
            if not line.startswith(" ") and tool_match:
                current_tool = {
                    "name": str(tool_match.group(1)).strip(),
                    "description": str(tool_match.group(2)).strip(),
                    "parameters": {},
                }
                tools.append(current_tool)
                in_parameters = False
                continue

            if current_tool is None:
                continue

            if stripped == "parameters:":
                in_parameters = True
                continue

            if in_parameters and ":" in stripped:
                param_name, _, raw_spec = stripped.partition(":")
                raw_spec = raw_spec.strip()
                try:
                    spec = ast.literal_eval(raw_spec)
                except (ValueError, SyntaxError):
                    spec = {"raw": raw_spec}
                current_tool["parameters"][param_name.strip()] = spec
                continue

            if current_tool["description"]:
                current_tool["description"] = f'{current_tool["description"]} {stripped}'.strip()
            else:
                current_tool["description"] = stripped

        return tools

    def _iter_selected_paths(self) -> list[tuple[str, str, Path]]:
        selected_paths: list[tuple[str, str, Path]] = []
        for dataset_name in self.dataset_names:
            dataset_family, filename = self.DATASETS[dataset_name]
            selected_paths.append((dataset_family, dataset_name, self.input_path / dataset_family / filename))
        return selected_paths

    def _load_raw_samples(self) -> list[dict[str, Any]]:
        raw_samples: list[dict[str, Any]] = []
        excluded_gold_half_cases = 0

        for dataset_family, dataset_name, path in self._iter_selected_paths():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"Expected list benchmark payload: {path}")
            for sample in payload:
                score = sample.get("score")
                if score == 0.5:
                    excluded_gold_half_cases += 1
                    continue
                enriched = dict(sample)
                enriched["dataset_family"] = dataset_family
                enriched["dataset_name"] = dataset_name
                enriched["_sample_index"] = len(raw_samples)
                raw_samples.append(enriched)

        if self.limit is not None:
            raw_samples = raw_samples[: self.limit]

        self.excluded_gold_half_cases = excluded_gold_half_cases
        self.loaded_case_count = len(raw_samples)
        return raw_samples

    def _build_case(self, raw_sample: dict[str, Any]) -> EvalCase:
        history = InteractionHistory(user_request=str(raw_sample["instruction"]))
        history_text = str(raw_sample.get("history", "")).strip()
        if history_text:
            for step in self._parse_history_steps(history_text):
                if isinstance(step, Action):
                    history.add_action(step)
                else:
                    history.add_observation(step)

        action_step = sum(1 for step in history.steps if isinstance(step, Action)) + 1
        action = self._parse_action_text(str(raw_sample.get("current_action", "")), step=action_step)
        env_info = str(raw_sample.get("env_info", ""))
        context = GuardrailContext(
            available_tools=self._parse_env_info(env_info),
            memory={
                "dataset_family": raw_sample["dataset_family"],
                "dataset_name": raw_sample["dataset_name"],
                "env_info_text": env_info,
            },
        )

        dataset_family = str(raw_sample["dataset_family"])
        dataset_name = str(raw_sample["dataset_name"])
        interaction_id = int(raw_sample.get("id-interaction", 0))
        segment_id = int(raw_sample.get("id-segment", 0))
        sample_index = int(raw_sample.get("_sample_index", 0))
        metadata = {
            "benchmark_name": self.name,
            "bench_family": _bench_family(dataset_family),
            "dataset_family": dataset_family,
            "dataset_name": dataset_name,
            "id-interaction": interaction_id,
            "id-segment": segment_id,
            "sample_index": sample_index,
            "gold_score": raw_sample.get("score"),
            "raw_instruction": raw_sample.get("instruction", ""),
            "raw_history": raw_sample.get("history", ""),
            "raw_current_action": raw_sample.get("current_action", ""),
            "raw_env_info": env_info,
        }
        for optional_key in ("attacker_tool", "aggressive", "attack_success"):
            if optional_key in raw_sample:
                metadata[optional_key] = raw_sample.get(optional_key)

        return EvalCase(
            case_id=f"{dataset_family}_{dataset_name}_{interaction_id}_{segment_id}_{sample_index}",
            history=history,
            action=action,
            context=context,
            gold_label=int(raw_sample["score"]),
            metadata=metadata,
        )

    def load_cases(self) -> list[EvalCase]:
        return [self._build_case(raw_sample) for raw_sample in self._load_raw_samples()]

    def summarize(self, records: list[EvalRecord]) -> dict[str, Any]:
        valid_records: list[EvalRecord] = []
        excluded_unparsed_cases = 0
        excluded_failed_cases = 0
        by_dataset: dict[str, list[EvalRecord]] = {}
        by_bench_family: dict[str, list[EvalRecord]] = {}

        for record in records:
            if not record.counted_in_metrics:
                if record.status == "parse_failed":
                    excluded_unparsed_cases += 1
                else:
                    excluded_failed_cases += 1
                continue
            if float(record.judgment.confidence) <= 0.0:
                excluded_unparsed_cases += 1
                continue
            valid_records.append(record)
            dataset_name = str(record.metadata.get("dataset_name", "unknown"))
            by_dataset.setdefault(dataset_name, []).append(record)
            family = str(record.metadata.get("bench_family", "unknown"))
            by_bench_family.setdefault(family, []).append(record)

        summary = _compute_binary_metrics(valid_records)
        by_dataset_summary = {
            dataset_name: _compute_binary_metrics(dataset_records)
            for dataset_name, dataset_records in sorted(by_dataset.items())
        }
        by_bench_family_summary = {
            family: _compute_binary_metrics(family_records)
            for family, family_records in sorted(by_bench_family.items())
        }
        traj_metrics = _build_traj_metrics_table(summary, by_bench_family_summary)

        return {
            "benchmark_name": self.name,
            **summary,
            "attempted_cases": self.loaded_case_count,
            "evaluated_cases": len(valid_records),
            "excluded_gold_half_cases": self.excluded_gold_half_cases,
            "excluded_unparsed_cases": excluded_unparsed_cases,
            "excluded_failed_cases": excluded_failed_cases,
            "excluded_from_metrics_cases": len(records) - len(valid_records),
            "by_bench_family": by_bench_family_summary,
            "traj_metrics": traj_metrics,
            "by_dataset": by_dataset_summary,
        }
