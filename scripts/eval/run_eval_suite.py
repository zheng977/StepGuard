"""Run one model matrix across multiple static benchmarks.

This is the higher-level suite runner for configs that share the same model
set across several benchmarks. Existing single-benchmark batch configs still
use scripts/eval/run_batch_eval.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.reporting import (  # noqa: E402
    collect_result_summary_rows,
    print_batch_run_summary,
    print_static_result_table,
    write_result_index,
)
from evals.suite_config import EvalSuiteConfig, StaticBenchmarkSpec, load_eval_suite_config  # noqa: E402


def _batch_helpers() -> Any:
    # Import lazily so `run_eval_suite.py --help` works in lightweight
    # environments that do not have the full eval dependency stack installed.
    from scripts.eval import run_batch_eval

    return run_batch_eval


def _batch_config_for_spec(
    spec: StaticBenchmarkSpec,
    *,
    api_models: list[Any] | None = None,
    vllm_models: list[Any] | None = None,
) -> Any:
    batch = _batch_helpers()
    return batch.BatchEvalConfig(
        benchmark=spec.benchmark,
        input_path=spec.input_path,
        output_root=spec.output_root,
        prompt_name=spec.prompt_name,
        prompt_file=spec.prompt_file,
        response_parser=spec.response_parser,
        limit=spec.limit,
        concurrency=spec.concurrency,
        temperature=spec.temperature,
        top_p=spec.top_p,
        presence_penalty=spec.presence_penalty,
        max_tokens=spec.max_tokens,
        timeout=spec.timeout,
        ts_subset=spec.ts_subset,
        api_models=api_models or [],
        vllm_models=vllm_models or [],
    )


def _selected_benchmarks(
    suite: EvalSuiteConfig,
    selected: set[str] | None,
) -> list[StaticBenchmarkSpec]:
    if not selected:
        return list(suite.benchmarks)
    benchmarks = [
        spec
        for spec in suite.benchmarks
        if spec.name in selected or spec.benchmark in selected
    ]
    missing = selected - {spec.name for spec in benchmarks} - {spec.benchmark for spec in benchmarks}
    if missing:
        raise ValueError(f"Unknown benchmark selector(s): {', '.join(sorted(missing))}")
    return benchmarks


def _validate_suite(
    specs: Iterable[StaticBenchmarkSpec],
    *,
    api_models: list[Any],
    vllm_models: list[Any],
) -> None:
    batch = _batch_helpers()
    for spec in specs:
        batch.validate_batch_eval_config(
            _batch_config_for_spec(
                spec,
                api_models=api_models,
                vllm_models=vllm_models,
            )
        )


def _raw_models_by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("name")): item for item in items if item.get("name") is not None}


def _model_benchmark_override(
    raw_model: dict[str, Any],
    spec: StaticBenchmarkSpec,
    field: str,
    *,
    coerce_file: bool = False,
) -> str | None:
    mapping = raw_model.get(f"{field}_by_benchmark")
    if not isinstance(mapping, dict):
        return None
    for key in (spec.name, spec.benchmark):
        if key not in mapping or mapping[key] is None:
            continue
        value = mapping[key]
        if coerce_file:
            batch = _batch_helpers()
            return batch._coerce_path(value, base_dir=REPO_ROOT)
        return str(value)
    return None


def _write_suite_indexes(output_roots: Iterable[str], suite_output_root: str) -> None:
    seen_roots: set[Path] = set()
    for raw_root in output_roots:
        root = Path(raw_root)
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        rows = collect_result_summary_rows(root)
        if rows:
            write_result_index(root, rows)
            print_static_result_table(rows)

    suite_root = Path(suite_output_root)
    if suite_root.exists():
        rows = collect_result_summary_rows(suite_root, recursive=True)
        if rows:
            write_result_index(suite_root, rows)


def run_eval_suite(
    suite: EvalSuiteConfig,
    *,
    selected_benchmarks: set[str] | None = None,
    limit_override: int | None = None,
    concurrency_override: int | None = None,
    dry_run: bool = False,
) -> list[Any]:
    specs = _selected_benchmarks(suite, selected_benchmarks)
    batch = _batch_helpers()
    api_models = batch._load_api_models(suite.api_models)
    vllm_models = batch._load_vllm_models(suite.vllm_models, defaults=suite.vllm_defaults)
    api_raw_by_name = _raw_models_by_name(suite.api_models)
    vllm_raw_by_name = _raw_models_by_name(suite.vllm_models)
    if not dry_run:
        _validate_suite(specs, api_models=api_models, vllm_models=vllm_models)

    results: list[Any] = []
    print(
        f"[eval_suite] suite={suite.name} benchmarks={len(specs)} "
        f"api_models={len(api_models)} vllm_models={len(vllm_models)}"
    )

    for model in api_models:
        for spec in specs:
            config = _batch_config_for_spec(spec)
            raw_model = api_raw_by_name.get(model.name, {})
            command = batch.build_eval_command(
                config,
                backend="api",
                model=model.model,
                base_url=model.base_url,
                api_key=model.api_key,
                prompt_name=_model_benchmark_override(raw_model, spec, "prompt_name"),
                prompt_file=_model_benchmark_override(raw_model, spec, "prompt_file", coerce_file=True),
                response_parser=_model_benchmark_override(raw_model, spec, "response_parser"),
                limit_override=limit_override,
                concurrency_override=concurrency_override,
            )
            run_name = f"{spec.name}/{model.name}"
            print(f"\n[eval_suite] run {run_name} (api)")
            if dry_run:
                print(" ".join(command))
                results.append(batch.BatchRunResult(run_name, "api", "success", 0, str(spec.output_root)))
                continue
            completed = batch.run_subprocess(command)
            results.append(
                batch.BatchRunResult(
                    name=run_name,
                    backend="api",
                    status="success" if completed.returncode == 0 else "failed",
                    returncode=completed.returncode,
                    output_root=str(spec.output_root),
                    error=None
                    if completed.returncode == 0
                    else f"run_eval.py exited with code {completed.returncode}",
                )
            )

    for model in vllm_models:
        running: Any | None = None
        log_path: Path | None = None
        base_url = f"http://{model.host}:{model.port}/v1"
        finished_specs: set[str] = set()
        try:
            raw_model = vllm_raw_by_name.get(model.name, {})
            if dry_run:
                print(f"\n[eval_suite] would start vLLM model={model.name} base_url={base_url}")
            else:
                running = batch.start_vllm_process(model)
                log_path = running.log_path
                batch.wait_for_http_ready(
                    base_url,
                    timeout_seconds=model.startup_timeout,
                    expected_model=model.model,
                    api_key=model.api_key,
                    process=running.process,
                )

            for spec in specs:
                config = _batch_config_for_spec(spec)
                command = batch.build_eval_command(
                    config,
                    backend="vllm",
                    model=model.model,
                    base_url=base_url,
                    api_key=model.api_key,
                    prompt_name=_model_benchmark_override(raw_model, spec, "prompt_name"),
                    prompt_file=_model_benchmark_override(raw_model, spec, "prompt_file", coerce_file=True),
                    response_parser=_model_benchmark_override(raw_model, spec, "response_parser"),
                    limit_override=limit_override,
                    concurrency_override=concurrency_override,
                )
                run_name = f"{spec.name}/{model.name}"
                print(f"\n[eval_suite] run {run_name} (vllm)")
                if dry_run:
                    print(" ".join(command))
                    results.append(batch.BatchRunResult(run_name, "vllm", "success", 0, str(spec.output_root)))
                    finished_specs.add(spec.name)
                    continue
                completed = batch.run_subprocess(command)
                results.append(
                    batch.BatchRunResult(
                        name=run_name,
                        backend="vllm",
                        status="success" if completed.returncode == 0 else "failed",
                        returncode=completed.returncode,
                        output_root=str(spec.output_root),
                        error=None
                        if completed.returncode == 0
                        else f"run_eval.py exited with code {completed.returncode}; vllm_log={log_path}",
                    )
                )
                finished_specs.add(spec.name)
        except Exception as exc:  # noqa: BLE001
            for spec in specs:
                if spec.name in finished_specs:
                    continue
                results.append(
                    batch.BatchRunResult(
                        name=f"{spec.name}/{model.name}",
                        backend="vllm",
                        status="failed",
                        returncode=None,
                        output_root=str(spec.output_root),
                        error=str(exc) if log_path is None else f"{exc}; vllm_log={log_path}",
                    )
                )
        finally:
            batch.stop_vllm_process(running)

    if not dry_run:
        _write_suite_indexes((spec.output_root for spec in specs), suite.output_root)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-benchmark static eval suite.")
    parser.add_argument("--config", required=True, help="Eval suite YAML config path.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optional benchmark suite names or registry keys to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Override per-run limit.")
    parser.add_argument("--concurrency", type=int, default=None, help="Override per-run concurrency.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running inference.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    suite = load_eval_suite_config(args.config)
    results = run_eval_suite(
        suite,
        selected_benchmarks=set(args.benchmarks) if args.benchmarks else None,
        limit_override=args.limit,
        concurrency_override=args.concurrency,
        dry_run=args.dry_run,
    )
    print_batch_run_summary(results)
    return 0 if all(result.status == "success" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
