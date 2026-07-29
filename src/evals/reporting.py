from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


class BatchRunLike(Protocol):
    name: str
    backend: str
    status: str
    output_root: str
    error: str | None


@dataclass(frozen=True)
class ResultSummaryRow:
    model: str
    prompt: str
    run_dir: str
    summary_path: Path
    summary: dict[str, Any]


def format_pct(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "n/a"


def print_dynamic_summary(
    *,
    benchmark_name: str,
    guard_name: str,
    agent_model: str,
    attack: str | None,
    summary: dict[str, Any],
    result_dir: Path,
    prompt_name: str | None = None,
    has_guard: bool = True,
) -> None:
    print()
    print("=" * 80)
    print(f"  Dynamic Benchmark: {benchmark_name}")
    prompt_suffix = f" (prompt={prompt_name})" if has_guard and prompt_name else ""
    print(f"  Guard: {guard_name}{prompt_suffix}")
    print(f"  Agent: {agent_model}")
    print(f"  Attack: {attack or 'none'}")
    print("-" * 80)

    if benchmark_name == "agentharm":
        print(f"  Subset:               {summary.get('subset')}")
        print(f"  Avg Score:            {format_pct(summary.get('avg_score'))}")
        print(f"  Avg Full Score:       {format_pct(summary.get('avg_full_score'))}")
        if summary.get("subset") == "harmful":
            print(f"  Harm Score:           {format_pct(summary.get('harm_score'))}")
            print(f"  Harm Full Score:      {format_pct(summary.get('harm_full_score'))}")
            print(f"  Security Rate Full:   {format_pct(summary.get('security_rate_full'))}")
        else:
            print(f"  Benign Success Score: {format_pct(summary.get('benign_success_score'))}")
            print(f"  Benign Full Success:  {format_pct(summary.get('benign_full_success_rate'))}")
        print(f"  Action Block Rate:    {format_pct(summary.get('block_rate_action'))}")
        print(f"  Traj Block Rate:      {format_pct(summary.get('block_rate_traj'))}")
    else:
        print(f"  Benign Utility:       {format_pct(summary.get('benign_utility'))}")
        print(f"  Utility Under Attack: {format_pct(summary.get('utility_under_attack'))}")
        print(f"  Security Rate:        {format_pct(summary.get('security_rate'))}")
        print(f"  ASR:                  {format_pct(summary.get('asr'))}")
        print("-" * 80)
        for suite, suite_summary in summary.get("by_suite", {}).items():
            print(
                f"  {suite:<15} "
                f"utility={format_pct(suite_summary.get('benign_utility'))}  "
                f"utility_atk={format_pct(suite_summary.get('utility_under_attack'))}  "
                f"ASR={format_pct(suite_summary.get('asr'))}"
            )
    print("=" * 80)
    print(f"  Results: {result_dir}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_model_prompt_from_run_dir(run_dir_name: str) -> tuple[str, str]:
    parts = run_dir_name.split("_prompt-")
    if len(parts) == 2:
        prefix = parts[0]
        model = prefix
        for benchmark_prefix in (
            "overdefense_traj_",
            "overdefense_",
            "at_bench_traj_",
            "atbench_pro_",
            "agentsafety_",
            "agentharm_",
            "agentdyn_",
            "ts_bench_",
            "at_bench_",
            "rjudge_",
        ):
            if prefix.startswith(benchmark_prefix):
                model = prefix[len(benchmark_prefix):]
                break
        prompt = parts[1].split("_", 1)[0]
        return model, prompt
    return run_dir_name, "?"


def collect_result_summary_rows(output_root: str | Path, *, recursive: bool = False) -> list[ResultSummaryRow]:
    root = Path(output_root)
    rows: list[ResultSummaryRow] = []
    pattern = "**/results_summary.json" if recursive else "*/results_summary.json"
    for summary_path in sorted(root.glob(pattern)):
        summary = _read_json(summary_path)
        if not summary:
            continue
        run_dir = summary_path.parent
        resolved_config = _read_json(run_dir / "resolved_config.json")
        parsed_model, parsed_prompt = _parse_model_prompt_from_run_dir(run_dir.name)
        model = str(
            resolved_config.get("model")
            or resolved_config.get("guard_model")
            or resolved_config.get("model_name")
            or parsed_model
        )
        prompt = str(
            resolved_config.get("prompt_name")
            or summary.get("prompt_name")
            or parsed_prompt
        )
        rows.append(
            ResultSummaryRow(
                model=model,
                prompt=prompt,
                run_dir=run_dir.name,
                summary_path=summary_path,
                summary=summary,
            )
        )
    return rows


def print_batch_run_summary(results: Iterable[BatchRunLike]) -> None:
    results = list(results)
    succeeded = [result for result in results if result.status == "success"]
    failed = [result for result in results if result.status != "success"]
    print()
    print("Batch run summary")
    print(f"  succeeded: {len(succeeded)}")
    for result in succeeded:
        print(f"  - {result.name} [{result.backend}] -> {result.output_root}")
    print(f"  failed: {len(failed)}")
    for result in failed:
        print(f"  - {result.name} [{result.backend}] -> {result.error}")


def print_static_result_table(rows: list[ResultSummaryRow]) -> None:
    if not rows:
        return

    is_overdefense = any("overdefense_rate" in row.summary for row in rows)
    print()
    print("=" * 100)
    if is_overdefense:
        print(f"  {'Model':<28} {'Prompt':<18} {'Safe':>5} {'TN':>5} {'FP':>5} {'OverDef':>8} {'fail':>5}")
        print("-" * 100)
        for row in rows:
            summary = row.summary
            print(
                f"  {row.model:<28} {row.prompt:<18} "
                f"{summary.get('total_safe', 0):>5} {summary.get('tn', 0):>5} "
                f"{summary.get('fp', 0):>5} {summary.get('overdefense_rate', 0):>7.1%} "
                f"{summary.get('excluded_from_metrics_cases', 0):>5}"
            )
    else:
        print(f"  {'Model':<28} {'Prompt':<18} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'n':>5} {'fail':>5}")
        print("-" * 100)
        for row in rows:
            summary = row.summary
            print(
                f"  {row.model:<28} {row.prompt:<18} "
                f"{summary.get('accuracy', 0):6.1%} {summary.get('precision', 0):6.1%} "
                f"{summary.get('recall', 0):6.1%} {summary.get('f1', 0):6.1%} "
                f"{summary.get('total', 0):>5} {summary.get('failed_cases', 0):>5}"
            )
    print("=" * 100)


def write_result_index(output_root: str | Path, rows: list[ResultSummaryRow]) -> tuple[Path, Path] | None:
    if not rows:
        return None
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "eval_results_index.csv"
    md_path = root / "eval_results_index.md"
    fields = [
        "model",
        "prompt",
        "run_dir",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "total",
        "failed_cases",
        "summary_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            summary = row.summary
            writer.writerow(
                {
                    "model": row.model,
                    "prompt": row.prompt,
                    "run_dir": row.run_dir,
                    "accuracy": summary.get("accuracy"),
                    "precision": summary.get("precision"),
                    "recall": summary.get("recall"),
                    "f1": summary.get("f1"),
                    "total": summary.get("total"),
                    "failed_cases": summary.get("failed_cases"),
                    "summary_path": str(row.summary_path),
                }
            )

    lines = [
        "# Eval Results Index",
        "",
        "| Model | Prompt | Acc | Prec | Rec | F1 | n | fail | Run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        summary = row.summary
        lines.append(
            "| "
            f"{row.model} | {row.prompt} | "
            f"{format_pct(summary.get('accuracy'))} | {format_pct(summary.get('precision'))} | "
            f"{format_pct(summary.get('recall'))} | {format_pct(summary.get('f1'))} | "
            f"{summary.get('total', '')} | {summary.get('failed_cases', '')} | {row.run_dir} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path
