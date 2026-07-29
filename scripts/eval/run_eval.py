from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if os.getenv("AGENTGUARD_EVAL_PACKAGE_FIRST") != "1":
    src_root_str = str(SRC_ROOT)
    sys.path = [path for path in sys.path if path != src_root_str]
    sys.path.insert(0, src_root_str)

from tqdm import tqdm

from evals.artifacts import build_failure_record, finalize_static_summary
from evals.base import EvalRecord
from evals.benchmarks import BENCHMARK_REGISTRY
from guardrail.prompts import PROFILE_REGISTRY
from src.utils.logger import get_logger


def _default_input_path(benchmark: str) -> Path:
    if benchmark == "ts_bench":
        return Path("benchmarks") / "ts_bench"
    if benchmark == "atbench_pro_traj":
        return Path("benchmarks") / "atbench_pro" / "ATbench-Pro.json"
    if benchmark == "rjudge":
        return Path("benchmarks") / "rjudge" / "test.json"
    if benchmark == "agentsafety":
        return Path("benchmarks") / "agentsafety" / "test.json"
    raise ValueError(f"No default input path for benchmark: {benchmark}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentGuard benchmark evaluation.")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARK_REGISTRY))
    parser.add_argument(
        "--input",
        default=None,
        help="Benchmark input path.",
    )
    parser.add_argument("--output-root", default="results", help="Output root directory.")
    parser.add_argument("--backend", default="api", choices=["api", "vllm"])
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--prompt-name", default="stepguard")
    parser.add_argument("--prompt-file", default=None, help="Optional external prompt template path.")
    parser.add_argument(
        "--response-parser",
        default="stepguard",
        choices=["auto", "strict", "safe_unsafe"] + list(PROFILE_REGISTRY),
        help="Response parser for model output.",
    )
    parser.add_argument(
        "--ts-subset",
        default="all",
        help=(
            "TS-Bench subset: all, agentdojo, agentharm, asb, or a single shard name "
            "(banking, slack, travel, workspace, harmful_steps, benign_steps, "
            "asb_opi_success, asb_dpi_success, asb_attack_failure)."
        ),
    )
    return parser


def resolved_config_from_args(
    args: argparse.Namespace,
    *,
    benchmark_name: str | None = None,
    prompt_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_input_path = Path(args.input) if args.input else _default_input_path(args.benchmark)
    config = {
        "benchmark_name": benchmark_name or args.benchmark,
        "input_path": str(resolved_input_path.resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "backend": args.backend,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_set": bool(args.api_key),
        "limit": args.limit,
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "prompt_name": args.prompt_name,
        "prompt_file": str(Path(args.prompt_file).resolve()) if args.prompt_file else None,
        "response_parser": args.response_parser,
        "run_started_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.benchmark == "ts_bench":
        config["ts_subset"] = args.ts_subset
    if prompt_metadata:
        config.update(prompt_metadata)
    return config


def main() -> int:
    from evals.evaluator import Evaluator
    from evals.results import ResultWriter
    from guardrail.guardrail import PredictiveGuardrail
    from infer.factory import InferFactory

    parser = build_parser()
    args = parser.parse_args()
    logger = get_logger("agentguard.eval")
    input_path = Path(args.input) if args.input else _default_input_path(args.benchmark)

    if not args.model:
        raise ValueError("Missing model. Pass --model or export OPENAI_MODEL.")
    if args.backend == "api" and not args.api_key:
        raise ValueError("Missing API key. Pass --api-key or export OPENAI_API_KEY.")
    if args.backend == "vllm" and not args.base_url:
        raise ValueError("Missing base URL. Pass --base-url or export OPENAI_BASE_URL.")

    adapter_cls = BENCHMARK_REGISTRY[args.benchmark]
    adapter_kwargs: dict[str, Any] = {"limit": args.limit}
    if args.benchmark == "ts_bench":
        adapter_kwargs["subset"] = args.ts_subset
    adapter = adapter_cls(input_path, **adapter_kwargs)
    infer_backend = InferFactory.create(
        args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or "EMPTY",
    )
    guardrail = PredictiveGuardrail(
        infer_backend,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        prompt_name=args.prompt_name,
        prompt_file=args.prompt_file,
        response_parser=args.response_parser,
    )
    evaluator = Evaluator(guardrail)
    writer = ResultWriter(
        output_root=args.output_root,
        benchmark_name=adapter.name,
        model_name=args.model,
        run_tag=guardrail.prompt_tag,
    )

    prompt_metadata = {
        "prompt_name": guardrail.prompt_name,
        "prompt_file": str(Path(guardrail.prompt_file).resolve()) if guardrail.prompt_file else None,
        "prompt_hash": guardrail.prompt_hash,
        "prompt_tag": guardrail.prompt_tag,
        "response_parser": guardrail.response_parser,
    }
    resolved_config = resolved_config_from_args(
        args,
        benchmark_name=adapter.name,
        prompt_metadata=prompt_metadata,
    )
    writer.write_resolved_config(resolved_config)
    writer.append_event(
        {
            "event_type": "run_started",
            "benchmark_name": adapter.name,
            "model_name": args.model,
            "input_path": str(input_path.resolve()),
            "run_dir": str(writer.run_dir.resolve()),
            "limit": args.limit,
            "concurrency": args.concurrency,
            "ts_subset": args.ts_subset if args.benchmark == "ts_bench" else None,
            "response_parser": args.response_parser,
            **prompt_metadata,
        }
    )

    cases = adapter.load_cases()
    logger.info(
        "Loaded %s cases from %s with concurrency=%s",
        len(cases),
        input_path,
        args.concurrency,
    )
    records: list[EvalRecord] = []

    for case in cases:
        writer.append_event(
            {
                "event_type": "case_started",
                "benchmark_name": adapter.name,
                "model_name": args.model,
                "case_id": case.case_id,
                "status": "running",
                "prompt_tag": guardrail.prompt_tag,
            }
        )
    progress = tqdm(
        total=len(cases),
        desc=f"Evaluating {adapter.name}",
        unit="case",
    )
    for case, record, error in evaluator.iter_evaluations(cases, concurrency=args.concurrency):
        try:
            if error is not None or record is None:
                record = build_failure_record(
                    case,
                    status="execution_failed",
                    error=str(error if error is not None else RuntimeError("Missing evaluation record")),
                )
            records.append(record)
            writer.append_record(record)
            writer.write_case(case, record)
            event_type = "case_finished" if record.is_success else "case_failed"
            writer.append_event(
                {
                    "event_type": event_type,
                    "benchmark_name": adapter.name,
                    "model_name": args.model,
                    "case_id": case.case_id,
                    "status": record.status,
                    "gold_label": record.gold_label,
                    "pred_label": record.pred_label,
                    "prompt_tag": guardrail.prompt_tag,
                    "case_path": str((writer.cases_dir / f"{case.case_id}.json").resolve()),
                    "error": record.error,
                }
            )
        except Exception as exc:  # noqa: BLE001
            fallback_record = build_failure_record(case, status="execution_failed", error=str(exc))
            records.append(fallback_record)
            writer.append_record(fallback_record)
            writer.write_case(case, fallback_record)
            writer.append_event(
                {
                    "event_type": "case_failed",
                    "benchmark_name": adapter.name,
                    "model_name": args.model,
                    "case_id": case.case_id,
                    "status": fallback_record.status,
                    "prompt_tag": guardrail.prompt_tag,
                    "case_path": str((writer.cases_dir / f"{case.case_id}.json").resolve()),
                    "error": str(exc),
                }
            )
            logger.exception("Failed case %s", case.case_id)
        finally:
            progress.update(1)
    progress.close()

    attempted_cases = len(cases)
    summary = finalize_static_summary(
        adapter.summarize(records),
        records,
        attempted_cases=attempted_cases,
    )
    completed_cases = summary["completed_cases"]
    failed_cases = summary["failed_cases"]
    writer.write_summary(summary)
    writer.append_event(
        {
            "event_type": "run_finished",
            "benchmark_name": adapter.name,
            "model_name": args.model,
            "status": "success" if failed_cases == 0 else "partial_success",
            "total_cases": attempted_cases,
            "completed_cases": completed_cases,
            "failed_cases": failed_cases,
            "prompt_tag": guardrail.prompt_tag,
            "summary_path": str((writer.run_dir / "results_summary.json").resolve()),
        }
    )
    logger.info("Finished run: %s", writer.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
