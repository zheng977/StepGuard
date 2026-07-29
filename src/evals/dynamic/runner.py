from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from evals.dynamic.config import DynamicEvalConfig
from evals.dynamic.factory import create_dynamic_benchmark, create_guardrail
from evals.reporting import print_dynamic_summary
from evals.results import ResultWriter
from src.utils.logger import get_logger


def _classify_guard_event(record_metadata: dict[str, Any], gold_label: int, event: dict[str, Any]) -> dict[str, Any]:
    """Add case-level error semantics to one guard event.

    In clean/utility trajectories, a blocked action is a false positive. In
    risk/attack trajectories, a block is only a candidate true positive until
    a human or benchmark-specific oracle checks whether that sequence step is
    actually the unsafe branch point.
    """

    blocked = bool(event.get("blocked"))
    has_attack = gold_label == 1
    if not has_attack:
        event_error_label = "false_positive" if blocked else "true_negative"
        requires_sequence_review = False
    elif blocked:
        event_error_label = "candidate_risk_block"
        requires_sequence_review = True
    else:
        event_error_label = "risk_step_not_blocked"
        requires_sequence_review = True

    return {
        "case_has_attack": has_attack,
        "case_utility": record_metadata.get("utility"),
        "case_security": record_metadata.get("security"),
        "case_suite": record_metadata.get("suite"),
        "case_subset": record_metadata.get("subset"),
        "case_category": record_metadata.get("category"),
        "case_behavior_id": record_metadata.get("behavior_id"),
        "case_attack_type": record_metadata.get("attack_type"),
        "event_error_label": event_error_label,
        "requires_sequence_review": requires_sequence_review,
    }


@dataclass(frozen=True)
class DynamicRunResult:
    run_dir: Path
    summary: dict[str, Any]
    guard_name: str
    record_count: int


def run_dynamic_eval(config: DynamicEvalConfig) -> DynamicRunResult:
    logger = get_logger("agentguard.dynamic_eval")

    guardrail, guard_name = create_guardrail(config)
    if guardrail is None:
        logger.info("Running WITHOUT guard (pure agent baseline)")

    benchmark = create_dynamic_benchmark(config)
    writer = ResultWriter(
        output_root=config.output_root,
        benchmark_name=benchmark.name,
        model_name=guard_name,
        run_tag=config.run_tag(guard_prompt_tag=getattr(guardrail, "prompt_tag", None)),
    )
    writer.write_resolved_config(config.resolved_run_config(guard_name=guard_name))

    logger.info(
        "Starting dynamic eval: benchmark=%s, guard=%s, agent=%s, suites=%s, attack=%s",
        config.benchmark,
        config.model,
        config.agent_model,
        config.suites,
        config.attack,
    )

    old_full_prompt_env = os.environ.get("AGENTGUARD_RECORD_FULL_PROMPT")
    old_full_context_env = os.environ.get("AGENTGUARD_RECORD_FULL_CONTEXT")
    if config.record_full_guard_context:
        os.environ["AGENTGUARD_RECORD_FULL_PROMPT"] = "1"
        os.environ["AGENTGUARD_RECORD_FULL_CONTEXT"] = "1"
    try:
        results = benchmark.run(
            guardrail,
            blocking_mode=config.blocking_mode,
            confidence_threshold=config.confidence_threshold,
            generic_feedback=config.generic_feedback,
            feedback_mode=config.feedback_mode,
            blocked_history_mode=config.blocked_history_mode,
            max_replans=config.max_replans,
            guard_reconsideration=config.guard_reconsideration,
        )
    finally:
        _restore_env("AGENTGUARD_RECORD_FULL_PROMPT", old_full_prompt_env)
        _restore_env("AGENTGUARD_RECORD_FULL_CONTEXT", old_full_context_env)

    records = benchmark.to_eval_records(results)
    for record in records:
        writer.append_record(record)
        for idx, guard_event in enumerate(record.metadata.get("guard_judgments") or []):
            writer.append_event(
                {
                    "event_type": "guard_judgment",
                    "case_id": record.case_id,
                    "gold_label": record.gold_label,
                    "pred_label": record.pred_label,
                    "event_index": idx,
                    **_classify_guard_event(record.metadata, record.gold_label, guard_event),
                    **guard_event,
                }
            )

    summary = benchmark.summarize(results)
    writer.write_summary(summary)

    print_dynamic_summary(
        benchmark_name=benchmark.name,
        guard_name=guard_name,
        agent_model=config.agent_model,
        attack=config.attack,
        summary=summary,
        result_dir=writer.run_dir,
        prompt_name=config.prompt_name,
        has_guard=guardrail is not None,
    )

    logger.info("Finished: %s", writer.run_dir)
    return DynamicRunResult(
        run_dir=writer.run_dir,
        summary=summary,
        guard_name=guard_name,
        record_count=len(records),
    )


def _restore_env(key: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = old_value
