from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def require_mapping(payload: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {name} to be a mapping.")
    return payload


def require_list(payload: Any, *, name: str) -> list[Any]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"Expected {name} to be a list.")
    return payload


def expand_env(value: Any) -> str:
    return os.path.expandvars(str(value))


def coerce_path(value: Any, *, base_dir: Path = REPO_ROOT) -> str | None:
    if value is None:
        return None
    path = Path(expand_env(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class StaticBenchmarkSpec:
    name: str
    benchmark: str
    input_path: str | None = None
    output_root: str | None = None
    prompt_name: str = "stepguard"
    prompt_file: str | None = None
    response_parser: str = "stepguard"
    limit: int | None = None
    concurrency: int = 1
    temperature: float = 0.0
    top_p: float | None = None
    presence_penalty: float | None = None
    max_tokens: int = 1024
    timeout: int = 120
    ts_subset: str = "all"


@dataclass(frozen=True)
class EvalSuiteConfig:
    name: str
    output_root: str
    benchmarks: list[StaticBenchmarkSpec]
    api_models: list[dict[str, Any]] = field(default_factory=list)
    vllm_models: list[dict[str, Any]] = field(default_factory=list)
    vllm_defaults: dict[str, Any] = field(default_factory=dict)


def _benchmark_name(item: dict[str, Any]) -> str:
    explicit_name = item.get("name")
    if explicit_name:
        return str(explicit_name)
    benchmark = str(item["benchmark"])
    ts_subset = item.get("ts_subset")
    if benchmark == "ts_bench" and ts_subset and str(ts_subset) != "all":
        return f"{benchmark}_{ts_subset}"
    return benchmark


def _load_static_benchmark_spec(
    item: dict[str, Any],
    *,
    defaults: dict[str, Any],
    suite_output_root: Path,
) -> StaticBenchmarkSpec:
    payload = deep_merge(defaults, item)
    benchmark = str(payload["benchmark"])
    name = _benchmark_name(payload)
    output_root = payload.get("output_root")
    if output_root is None:
        output_root = suite_output_root / name
    else:
        output_root = coerce_path(output_root)
    return StaticBenchmarkSpec(
        name=name,
        benchmark=benchmark,
        input_path=coerce_path(payload.get("input")),
        output_root=str(output_root),
        prompt_name=expand_env(payload.get("prompt_name", "stepguard")),
        prompt_file=coerce_path(payload.get("prompt_file")),
        response_parser=expand_env(payload.get("response_parser", "stepguard")),
        limit=int(payload["limit"]) if payload.get("limit") is not None else None,
        concurrency=int(payload.get("concurrency", 1)),
        temperature=float(payload.get("temperature", 0.0)),
        top_p=float(payload["top_p"]) if payload.get("top_p") is not None else None,
        presence_penalty=(
            float(payload["presence_penalty"])
            if payload.get("presence_penalty") is not None
            else None
        ),
        max_tokens=int(payload.get("max_tokens", 1024)),
        timeout=int(payload.get("timeout", 120)),
        ts_subset=str(payload.get("ts_subset", "all")),
    )


def load_eval_suite_config(config_path: str | Path) -> EvalSuiteConfig:
    path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload = require_mapping(payload, name="eval suite config")

    defaults = require_mapping(payload.get("defaults", {}) or {}, name="defaults")
    suite_name = str(payload.get("name") or path.stem)
    output_root = Path(
        coerce_path(payload.get("output_root") or (REPO_ROOT / "results" / suite_name))
        or REPO_ROOT / "results" / suite_name
    )

    raw_benchmarks = require_list(payload.get("benchmarks"), name="benchmarks")
    if not raw_benchmarks:
        raise ValueError("Eval suite config must define at least one benchmark.")

    benchmarks: list[StaticBenchmarkSpec] = []
    seen_names: set[str] = set()
    for raw_item in raw_benchmarks:
        item = require_mapping(raw_item, name="benchmark")
        spec = _load_static_benchmark_spec(
            item,
            defaults=defaults,
            suite_output_root=output_root,
        )
        if spec.name in seen_names:
            raise ValueError(f'Duplicate benchmark suite name "{spec.name}".')
        seen_names.add(spec.name)
        benchmarks.append(spec)

    api_models = [
        dict(require_mapping(item, name="api model"))
        for item in require_list(payload.get("api_models"), name="api_models")
    ]
    vllm_models = [
        dict(require_mapping(item, name="vllm model"))
        for item in require_list(payload.get("vllm_models"), name="vllm_models")
    ]
    if not api_models and not vllm_models:
        raise ValueError("Eval suite config must define api_models or vllm_models.")

    vllm_defaults = require_mapping(payload.get("vllm_defaults", {}) or {}, name="vllm_defaults")
    return EvalSuiteConfig(
        name=suite_name,
        output_root=str(output_root),
        benchmarks=benchmarks,
        api_models=api_models,
        vllm_models=vllm_models,
        vllm_defaults=dict(vllm_defaults),
    )
