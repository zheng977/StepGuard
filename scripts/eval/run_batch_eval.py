'''
  python scripts/eval/run_batch_eval.py \
    --config configs/batch_eval.example.yaml \
    --concurrency 128 --limit 20
'''

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
from typing import Any
import urllib.request
from urllib.error import URLError
from urllib.request import Request, urlopen
import importlib.util

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass
class APIModelConfig:
    name: str
    model: str
    base_url: str
    api_key: str
    prompt_name: str | None = None
    prompt_file: str | None = None
    response_parser: str | None = None


@dataclass
class VLLMModelConfig:
    name: str
    model: str
    model_path: str
    prompt_name: str | None = None
    prompt_file: str | None = None
    response_parser: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    api_key: str = "EMPTY"
    host: str = "127.0.0.1"
    port: int = 18000
    max_model_len: int = 32768
    dtype: str = "auto"
    startup_timeout: int = 600
    vllm_extra_args: list[str] = field(default_factory=list)


@dataclass
class BatchEvalConfig:
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
    api_models: list[APIModelConfig] = field(default_factory=list)
    vllm_models: list[VLLMModelConfig] = field(default_factory=list)


@dataclass
class BatchRunResult:
    name: str
    backend: str
    status: str
    returncode: int | None
    output_root: str
    error: str | None = None


@dataclass
class RunningVLLM:
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: TextIOWrapper
    stream_thread: threading.Thread


def _benchmark_registry() -> dict[str, Any]:
    from evals.benchmarks import BENCHMARK_REGISTRY

    return BENCHMARK_REGISTRY


def _require_mapping(payload: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {name} to be a mapping.")
    return payload


def _expand_env(value: str) -> str:
    return os.path.expandvars(str(value))


def _coerce_path(value: str | None, *, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(_expand_env(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _load_api_models(items: Any) -> list[APIModelConfig]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("Expected api_models to be a list.")
    models: list[APIModelConfig] = []
    for raw_item in items:
        item = _require_mapping(raw_item, name="api model")
        models.append(
            APIModelConfig(
                name=str(item["name"]),
                model=_expand_env(str(item["model"])),
                base_url=_expand_env(str(item["base_url"])),
                api_key=_expand_env(str(item["api_key"])),
                prompt_name=_expand_env(str(item["prompt_name"])) if item.get("prompt_name") is not None else None,
                prompt_file=_coerce_path(item.get("prompt_file"), base_dir=REPO_ROOT),
                response_parser=(
                    _expand_env(str(item["response_parser"])) if item.get("response_parser") is not None else None
                ),
            )
        )
    return models


def _load_vllm_models(items: Any, *, defaults: dict[str, Any]) -> list[VLLMModelConfig]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("Expected vllm_models to be a list.")
    models: list[VLLMModelConfig] = []
    default_host = _expand_env(str(defaults.get("host", "127.0.0.1")))
    default_port = int(defaults.get("port", 18000))
    default_api_key = _expand_env(str(defaults.get("api_key", "EMPTY")))
    default_max_model_len = int(_expand_env(str(defaults.get("max_model_len", 32768))))
    default_dtype = _expand_env(str(defaults.get("dtype", "auto")))
    default_startup_timeout = int(_expand_env(str(defaults.get("startup_timeout", 600))))
    default_extra_args = defaults.get("vllm_extra_args", []) or []
    if not isinstance(default_extra_args, list):
        raise ValueError("Expected vllm_defaults.vllm_extra_args to be a list.")
    for raw_item in items:
        item = _require_mapping(raw_item, name="vllm model")
        model_path = Path(_expand_env(str(item["model_path"]))).expanduser()
        models.append(
            VLLMModelConfig(
                name=_expand_env(str(item["name"])),
                model=_expand_env(str(item["model"])),
                model_path=str(model_path),
                prompt_name=_expand_env(str(item["prompt_name"])) if item.get("prompt_name") is not None else None,
                prompt_file=_coerce_path(item.get("prompt_file"), base_dir=REPO_ROOT),
                response_parser=(
                    _expand_env(str(item["response_parser"])) if item.get("response_parser") is not None else None
                ),
                tensor_parallel_size=int(_expand_env(str(item.get("tensor_parallel_size", 1)))),
                gpu_memory_utilization=float(_expand_env(str(item.get("gpu_memory_utilization", 0.9)))),
                api_key=_expand_env(str(item.get("api_key", default_api_key))),
                host=_expand_env(str(item.get("host", default_host))),
                port=int(_expand_env(str(item.get("port", default_port)))),
                max_model_len=int(_expand_env(str(item.get("max_model_len", default_max_model_len)))),
                dtype=_expand_env(str(item.get("dtype", default_dtype))),
                startup_timeout=int(_expand_env(str(item.get("startup_timeout", default_startup_timeout)))),
                vllm_extra_args=[_expand_env(str(arg)) for arg in item.get("vllm_extra_args", default_extra_args)],
            )
        )
    return models


def load_batch_eval_config(config_path: str | Path) -> BatchEvalConfig:
    path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload = _require_mapping(payload, name="batch eval config")
    benchmark = str(payload["benchmark"])
    vllm_defaults = payload.get("vllm_defaults", {}) or {}
    vllm_defaults = _require_mapping(vllm_defaults, name="vllm_defaults")
    output_root = payload.get("output_root")
    if output_root is None:
        output_root = REPO_ROOT / "results" / benchmark
    else:
        output_root = _coerce_path(str(output_root), base_dir=REPO_ROOT)
    return BatchEvalConfig(
        benchmark=benchmark,
        input_path=_coerce_path(payload.get("input"), base_dir=REPO_ROOT),
        output_root=str(output_root),
        prompt_name=_expand_env(str(payload.get("prompt_name", "stepguard"))),
        prompt_file=_coerce_path(payload.get("prompt_file"), base_dir=REPO_ROOT),
        response_parser=_expand_env(str(payload.get("response_parser", "stepguard"))),
        limit=int(payload["limit"]) if payload.get("limit") is not None else None,
        concurrency=int(payload.get("concurrency", 1)),
        temperature=float(payload.get("temperature", 0.0)),
        top_p=float(payload["top_p"]) if payload.get("top_p") is not None else None,
        presence_penalty=float(payload["presence_penalty"]) if payload.get("presence_penalty") is not None else None,
        max_tokens=int(payload.get("max_tokens", 1024)),
        timeout=int(payload.get("timeout", 120)),
        ts_subset=str(payload.get("ts_subset", "all")),
        api_models=_load_api_models(payload.get("api_models")),
        vllm_models=_load_vllm_models(payload.get("vllm_models"), defaults=vllm_defaults),
    )


def validate_batch_eval_config(config: BatchEvalConfig) -> None:
    benchmark_registry = _benchmark_registry()
    if config.benchmark not in benchmark_registry:
        raise ValueError(
            f'Unsupported benchmark "{config.benchmark}". '
            f'Expected one of: {", ".join(sorted(benchmark_registry))}.'
        )
    if not config.api_models and not config.vllm_models:
        raise ValueError("At least one model must be configured in api_models or vllm_models.")
    seen_names: set[str] = set()
    for model in [*config.api_models, *config.vllm_models]:
        if model.name in seen_names:
            raise ValueError(f'Duplicate model name "{model.name}" in batch config.')
        seen_names.add(model.name)
    for model in config.vllm_models:
        if not Path(model.model_path).exists():
            raise ValueError(f'vLLM model_path does not exist for "{model.name}": {model.model_path}')


def _append_optional_arg(command: list[str], name: str, value: str | int | float | None) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def _resolve_prompt_name(config: BatchEvalConfig, *, prompt_name_override: str | None = None) -> str | None:
    if prompt_name_override is not None and str(prompt_name_override).strip():
        return str(prompt_name_override).strip()
    if config.prompt_name is not None and str(config.prompt_name).strip():
        return str(config.prompt_name).strip()
    return None


def build_eval_command(
    config: BatchEvalConfig,
    *,
    backend: str,
    model: str,
    base_url: str,
    api_key: str,
    prompt_name: str | None = None,
    prompt_file: str | None = None,
    response_parser: str | None = None,
    limit_override: int | None = None,
    concurrency_override: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval" / "run_eval.py"),
        "--benchmark",
        config.benchmark,
        "--output-root",
        str(config.output_root),
        "--backend",
        backend,
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--concurrency",
        str(concurrency_override if concurrency_override is not None else config.concurrency),
        "--temperature",
        str(config.temperature),
        "--max-tokens",
        str(config.max_tokens),
        "--timeout",
        str(config.timeout),
        "--response-parser",
        str(response_parser or config.response_parser),
    ]
    if config.top_p is not None:
        command.extend(["--top-p", str(config.top_p)])
    if config.presence_penalty is not None:
        command.extend(["--presence-penalty", str(config.presence_penalty)])
    _append_optional_arg(command, "--input", config.input_path)
    _append_optional_arg(command, "--limit", limit_override if limit_override is not None else config.limit)
    if prompt_file is not None:
        command.extend(["--prompt-file", prompt_file])
    elif prompt_name is not None and str(prompt_name).strip():
        command.extend(["--prompt-name", str(prompt_name).strip()])
    elif config.prompt_file is not None:
        command.extend(["--prompt-file", config.prompt_file])
    else:
        effective_prompt_name = _resolve_prompt_name(config)
        if effective_prompt_name is not None:
            command.extend(["--prompt-name", effective_prompt_name])
        else:
            raise ValueError("Batch eval config must resolve to a prompt source. Set prompt_name or prompt_file.")
    if config.benchmark == "ts_bench":
        command.extend(["--ts-subset", config.ts_subset])
    return command


def build_vllm_launch_command(model: VLLMModelConfig) -> list[str]:
    launcher_python = _resolve_vllm_python()
    launch_mode = detect_vllm_launch_mode(launcher_python=launcher_python)
    if launch_mode == "cli":
        cli_path = _resolve_vllm_cli(launcher_python)
        if cli_path is None:
            raise RuntimeError("detect_vllm_launch_mode returned cli but no vllm executable was found.")
        command = [
            launcher_python,
            cli_path,
            "serve",
            model.model_path,
            "--served-model-name",
            model.model,
            "--host",
            model.host,
            "--port",
            str(model.port),
            "--api-key",
            model.api_key,
            "--tensor-parallel-size",
            str(model.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(model.gpu_memory_utilization),
            "--max-model-len",
            str(model.max_model_len),
            "--dtype",
            model.dtype,
        ]
    elif launch_mode == "module":
        command = [
            launcher_python,
            "-m",
            "vllm",
            "serve",
            model.model_path,
            "--served-model-name",
            model.model,
            "--host",
            model.host,
            "--port",
            str(model.port),
            "--api-key",
            model.api_key,
            "--tensor-parallel-size",
            str(model.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(model.gpu_memory_utilization),
            "--max-model-len",
            str(model.max_model_len),
            "--dtype",
            model.dtype,
        ]
    elif launch_mode == "legacy_api_server":
        command = [
            launcher_python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model.model_path,
            "--served-model-name",
            model.model,
            "--host",
            model.host,
            "--port",
            str(model.port),
            "--api-key",
            model.api_key,
            "--tensor-parallel-size",
            str(model.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(model.gpu_memory_utilization),
            "--max-model-len",
            str(model.max_model_len),
            "--dtype",
            model.dtype,
        ]
    else:
        raise RuntimeError(f"Unsupported vLLM launch mode: {launch_mode}")
    command.extend(model.vllm_extra_args)
    return command


def _resolve_vllm_python() -> str:
    return os.getenv("AGENTGUARD_VLLM_PYTHON", sys.executable)


def _resolve_vllm_cli(launcher_python: str) -> str | None:
    if launcher_python == sys.executable:
        return shutil.which("vllm")
    candidate = Path(launcher_python).resolve().parent / "vllm"
    return str(candidate) if candidate.is_file() else None


def detect_vllm_launch_mode(*, launcher_python: str | None = None) -> str:
    forced_mode = os.getenv("AGENTGUARD_VLLM_LAUNCH_MODE")
    if forced_mode:
        return forced_mode.strip()

    launcher_python = launcher_python or _resolve_vllm_python()
    cli_path = _resolve_vllm_cli(launcher_python)
    if cli_path and Path(cli_path).is_file():
        return "cli"
    if launcher_python != sys.executable:
        return "module"
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.origin is not None:
        module_dir = Path(spec.origin).resolve().parent
        if (module_dir / "__main__.py").is_file():
            return "module"
    if importlib.util.find_spec("vllm.entrypoints.openai.api_server") is not None:
        return "legacy_api_server"
    raise RuntimeError(
        "Cannot find a usable vLLM launcher. Tried: `vllm` CLI, `python -m vllm`, and "
        "`python -m vllm.entrypoints.openai.api_server`. Install vllm in the active environment first, "
        "or set AGENTGUARD_VLLM_PYTHON / AGENTGUARD_VLLM_LAUNCH_MODE."
    )


def _socket_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def _build_auth_headers(api_key: str | None) -> dict[str, str]:
    if api_key is None or not str(api_key).strip():
        return {}
    return {"Authorization": f"Bearer {str(api_key).strip()}"}


def _extract_served_model_ids(base_url: str, *, api_key: str | None = None) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    request = Request(url, headers=_build_auth_headers(api_key))
    # Bypass proxy for local vLLM health checks
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    model_ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model_ids.append(str(item["id"]))
    return model_ids


def wait_for_http_ready(
    base_url: str,
    *,
    timeout_seconds: int,
    expected_model: str | None = None,
    api_key: str | None = None,
    process: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.time() + timeout_seconds
    url = base_url.rstrip("/") + "/models"
    printed_wait_message = False
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM process exited before becoming ready (returncode={process.returncode}).")
        try:
            model_ids = _extract_served_model_ids(base_url, api_key=api_key)
            if expected_model is not None and expected_model not in model_ids:
                raise RuntimeError(
                    f'vLLM responded at {url}, but expected model "{expected_model}" was not found. '
                    f"Available models: {model_ids}"
                )
            if model_ids or expected_model is None:
                print(f"[batch_eval] vLLM ready at {base_url} with models: {model_ids}")
                return
        except (URLError, TimeoutError, OSError):
            if not printed_wait_message:
                print(f"[batch_eval] Waiting for vLLM to become ready at {url} ...")
                printed_wait_message = True
        except RuntimeError:
            raise
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for vLLM server: {url}")


def _stream_process_output(
    stream: TextIOWrapper | None,
    log_handle: TextIOWrapper,
    *,
    prefix: str,
) -> None:
    if stream is None:
        return
    try:
        for raw_line in stream:
            log_handle.write(raw_line)
            log_handle.flush()
            if _should_echo_vllm_line(raw_line):
                print(f"{prefix}{raw_line.rstrip()}")
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


def _should_echo_vllm_line(line: str) -> bool:
    """Keep noisy access logs in the log file unless explicitly requested."""
    mode = os.getenv("AGENTGUARD_VLLM_STDOUT", "important").strip().lower()
    if mode in {"1", "true", "yes", "all"}:
        return True
    if mode in {"0", "false", "no", "none", "quiet"}:
        return False

    noisy_fragments = (
        '"POST /v1/chat/completions HTTP/1.1" 200 OK',
        '"GET /v1/models HTTP/1.1" 200 OK',
        '"GET /models HTTP/1.1" 200 OK',
    )
    if any(fragment in line for fragment in noisy_fragments):
        return False

    important_fragments = (
        "ERROR",
        "CRITICAL",
        "Traceback",
        "Exception",
        "RuntimeError",
        "ValueError",
        "CUDA out of memory",
        "Killed",
    )
    return any(fragment in line for fragment in important_fragments)


def _resolve_log_dir() -> Path:
    log_dir = REPO_ROOT / "results" / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_vllm_model_path(model: VLLMModelConfig) -> str:
    """Patch tokenizer_config.extra_special_tokens list for older vLLM stacks."""
    source = Path(model.model_path)
    tokenizer_config = source / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return model.model_path

    try:
        payload = _read_json(tokenizer_config)
    except json.JSONDecodeError:
        return model.model_path

    if not isinstance(payload.get("extra_special_tokens"), list):
        return model.model_path

    cache_key = f"{model.name}_{abs(hash(str(source.resolve()))) % (10**12)}"
    target = Path("/tmp") / "agentguard_vllm_safe_models" / cache_key
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.mkdir(parents=True, exist_ok=True)

    patched = dict(payload)
    patched.pop("extra_special_tokens", None)

    for item in source.iterdir():
        destination = target / item.name
        if item.name == "tokenizer_config.json":
            destination.write_text(
                json.dumps(patched, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            destination.symlink_to(item)

    print(f"[batch_eval] Patched tokenizer_config for {model.name}: {target}")
    return str(target)


def start_vllm_process(model: VLLMModelConfig) -> RunningVLLM:
    if _socket_is_open(model.host, model.port):
        raise RuntimeError(
            f'Port {model.port} is already in use before starting vLLM model "{model.name}". '
            "Use a different port or stop the existing service."
        )
    prepared_model_path = _prepare_vllm_model_path(model)
    log_path = _resolve_log_dir() / f"vllm_{model.name}_{int(time.time())}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    print(
        f'[batch_eval] Starting vLLM model "{model.name}" on http://{model.host}:{model.port}/v1 '
        f'(served_model_name={model.model}, model_path={prepared_model_path})'
    )
    print(f"[batch_eval] vLLM log: {log_path}")
    launch_model = VLLMModelConfig(
        name=model.name,
        model=model.model,
        model_path=prepared_model_path,
        prompt_name=model.prompt_name,
        prompt_file=model.prompt_file,
        response_parser=model.response_parser,
        tensor_parallel_size=model.tensor_parallel_size,
        gpu_memory_utilization=model.gpu_memory_utilization,
        api_key=model.api_key,
        host=model.host,
        port=model.port,
        max_model_len=model.max_model_len,
        dtype=model.dtype,
        startup_timeout=model.startup_timeout,
        vllm_extra_args=list(model.vllm_extra_args),
    )
    process = subprocess.Popen(  # noqa: S603
        build_vllm_launch_command(launch_model),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stream_thread = threading.Thread(
        target=_stream_process_output,
        args=(process.stdout, log_handle),
        kwargs={"prefix": f"[vllm:{model.name}] "},
        daemon=True,
    )
    stream_thread.start()
    return RunningVLLM(
        process=process,
        log_path=log_path,
        log_handle=log_handle,
        stream_thread=stream_thread,
    )


def stop_vllm_process(running: RunningVLLM | None) -> None:
    if running is None:
        return
    process = running.process
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    running.stream_thread.join(timeout=5)
    running.log_handle.close()


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)  # noqa: S603


def run_batch_eval(
    config: BatchEvalConfig,
    *,
    limit_override: int | None = None,
    concurrency_override: int | None = None,
) -> list[BatchRunResult]:
    results: list[BatchRunResult] = []
    for model in config.api_models:
        command = build_eval_command(
            config,
            backend="api",
            model=model.model,
            base_url=model.base_url,
            api_key=model.api_key,
            prompt_name=model.prompt_name,
            prompt_file=model.prompt_file,
            response_parser=model.response_parser,
            limit_override=limit_override,
            concurrency_override=concurrency_override,
        )
        completed = run_subprocess(command)
        results.append(
            BatchRunResult(
                name=model.name,
                backend="api",
                status="success" if completed.returncode == 0 else "failed",
                returncode=completed.returncode,
                output_root=str(config.output_root),
                error=None if completed.returncode == 0 else f"run_eval.py exited with code {completed.returncode}",
            )
        )

    for model in config.vllm_models:
        running: RunningVLLM | None = None
        log_path: Path | None = None
        try:
            running = start_vllm_process(model)
            log_path = running.log_path
            wait_for_http_ready(
                f"http://{model.host}:{model.port}/v1",
                timeout_seconds=model.startup_timeout,
                expected_model=model.model,
                api_key=model.api_key,
                process=running.process,
            )
            command = build_eval_command(
                config,
                backend="vllm",
                model=model.model,
                base_url=f"http://{model.host}:{model.port}/v1",
                api_key=model.api_key,
                prompt_name=model.prompt_name,
                prompt_file=model.prompt_file,
                response_parser=model.response_parser,
                limit_override=limit_override,
                concurrency_override=concurrency_override,
            )
            completed = run_subprocess(command)
            results.append(
                BatchRunResult(
                    name=model.name,
                    backend="vllm",
                    status="success" if completed.returncode == 0 else "failed",
                    returncode=completed.returncode,
                    output_root=str(config.output_root),
                    error=None
                    if completed.returncode == 0
                    else f"run_eval.py exited with code {completed.returncode}; vllm_log={log_path}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                BatchRunResult(
                    name=model.name,
                    backend="vllm",
                    status="failed",
                    returncode=None,
                    output_root=str(config.output_root),
                    error=str(exc) if log_path is None else f"{exc}; vllm_log={log_path}",
                )
            )
        finally:
            stop_vllm_process(running)
    return results


@dataclass
class LogprobsCaseResult:
    case_id: str
    gold_label: int
    risk_source: str
    failure_mode: str
    p_safe: float
    p_unsafe: float
    entropy: float
    pred: int
    fmt: str


def _run_logprobs_for_model(
    *,
    model_name: str,
    served_model_name: str,
    base_url: str,
    api_key: str,
    cases: list,
    profile: Any,
    top_logprobs_n: int = 20,
) -> list[LogprobsCaseResult]:
    """Run logprobs extraction for a single model on all cases."""
    from evals.logprobs import binary_entropy, extract_binary_probs
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    results: list[LogprobsCaseResult] = []

    for case in cases:
        prompt = profile.build_prompt(
            history=case.history,
            action=case.action,
            context=case.context,
        )
        try:
            response = client.chat.completions.create(
                model=served_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs_n,
            )
        except Exception as exc:
            print(f"  [WARN] {model_name} case={case.case_id}: inference failed: {exc}")
            results.append(LogprobsCaseResult(
                case_id=case.case_id,
                gold_label=case.gold_label,
                risk_source=str(case.metadata.get("risk_source", "unknown")),
                failure_mode=str(case.metadata.get("failure_mode", "unknown")),
                p_safe=0.0, p_unsafe=0.0, entropy=0.0, pred=-1, fmt="error",
            ))
            continue

        choice = response.choices[0]
        if choice.logprobs and choice.logprobs.content:
            top = choice.logprobs.content[0].top_logprobs
            p_safe, p_unsafe, fmt = extract_binary_probs(top)
            h = binary_entropy(p_safe, p_unsafe)
            pred = 1 if p_unsafe > p_safe else 0
        else:
            p_safe, p_unsafe, fmt, h, pred = 0.0, 0.0, "no_logprobs", 0.0, -1

        results.append(LogprobsCaseResult(
            case_id=case.case_id,
            gold_label=case.gold_label,
            risk_source=str(case.metadata.get("risk_source", "unknown")),
            failure_mode=str(case.metadata.get("failure_mode", "unknown")),
            p_safe=p_safe,
            p_unsafe=p_unsafe,
            entropy=h,
            pred=pred,
            fmt=fmt,
        ))

    return results


def _write_logprobs_outputs(
    all_results: dict[str, list[LogprobsCaseResult]],
    output_dir: Path,
) -> None:
    """Generate cross-model comparison outputs: CSV + aggregation + summary."""
    from evals.logprobs import kl_divergence

    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(all_results.keys())

    # ── 1. Per-case CSV ──
    csv_path = output_dir / "logprobs_per_case.csv"
    # Build a case_id-indexed dict for alignment
    case_ids: list[str] = []
    case_meta: dict[str, LogprobsCaseResult] = {}
    if model_names:
        first_model = model_names[0]
        for r in all_results[first_model]:
            case_ids.append(r.case_id)
            case_meta[r.case_id] = r

    header = ["case_id", "gold_label", "risk_source", "failure_mode"]
    for mn in model_names:
        header.extend([f"{mn}_p_safe", f"{mn}_p_unsafe", f"{mn}_entropy", f"{mn}_pred", f"{mn}_fmt"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cid in case_ids:
            meta = case_meta[cid]
            row: list[Any] = [cid, meta.gold_label, meta.risk_source, meta.failure_mode]
            for mn in model_names:
                # Find result for this case_id in this model
                r = next((x for x in all_results[mn] if x.case_id == cid), None)
                if r:
                    row.extend([f"{r.p_safe:.6f}", f"{r.p_unsafe:.6f}", f"{r.entropy:.6f}", r.pred, r.fmt])
                else:
                    row.extend(["", "", "", "", ""])
            writer.writerow(row)
    print(f"[logprobs] Per-case CSV: {csv_path}")

    # ── 2. Model-level summary ──
    summary: dict[str, Any] = {"models": {}}
    for mn in model_names:
        results = all_results[mn]
        valid = [r for r in results if r.pred >= 0]
        n = len(valid)
        if n == 0:
            summary["models"][mn] = {"n": 0, "error": "no valid results"}
            continue
        correct = sum(1 for r in valid if r.pred == r.gold_label)
        avg_entropy = sum(r.entropy for r in valid) / n
        avg_p_unsafe = sum(r.p_unsafe for r in valid) / n

        # Per-label stats
        safe_cases = [r for r in valid if r.gold_label == 0]
        unsafe_cases = [r for r in valid if r.gold_label == 1]

        summary["models"][mn] = {
            "n": n,
            "accuracy": round(correct / n, 4),
            "avg_entropy": round(avg_entropy, 4),
            "avg_p_unsafe": round(avg_p_unsafe, 4),
            "format": results[0].fmt if results else "unknown",
            "safe_avg_entropy": round(sum(r.entropy for r in safe_cases) / len(safe_cases), 4) if safe_cases else None,
            "unsafe_avg_entropy": round(sum(r.entropy for r in unsafe_cases) / len(unsafe_cases), 4) if unsafe_cases else None,
            "safe_avg_p_unsafe": round(sum(r.p_unsafe for r in safe_cases) / len(safe_cases), 4) if safe_cases else None,
            "unsafe_avg_p_unsafe": round(sum(r.p_unsafe for r in unsafe_cases) / len(unsafe_cases), 4) if unsafe_cases else None,
        }

    # ── 3. Cross-model KL divergence matrix ──
    kl_matrix: dict[str, dict[str, float]] = {}
    for mn_a in model_names:
        kl_matrix[mn_a] = {}
        for mn_b in model_names:
            if mn_a == mn_b:
                kl_matrix[mn_a][mn_b] = 0.0
                continue
            # Average KL(a || b) across shared valid cases
            kl_vals: list[float] = []
            results_a = {r.case_id: r for r in all_results[mn_a]}
            results_b = {r.case_id: r for r in all_results[mn_b]}
            for cid in case_ids:
                ra, rb = results_a.get(cid), results_b.get(cid)
                if ra and rb and ra.pred >= 0 and rb.pred >= 0:
                    kl_vals.append(kl_divergence((ra.p_safe, ra.p_unsafe), (rb.p_safe, rb.p_unsafe)))
            kl_matrix[mn_a][mn_b] = round(sum(kl_vals) / len(kl_vals), 6) if kl_vals else float("nan")
    summary["kl_divergence_matrix"] = kl_matrix

    # ── 4. Per-risk-source aggregation ──
    risk_sources: set[str] = set()
    for results in all_results.values():
        for r in results:
            risk_sources.add(r.risk_source)

    risk_agg: dict[str, dict[str, Any]] = {}
    for rs in sorted(risk_sources):
        risk_agg[rs] = {}
        for mn in model_names:
            rs_results = [r for r in all_results[mn] if r.risk_source == rs and r.pred >= 0]
            if not rs_results:
                continue
            n_rs = len(rs_results)
            risk_agg[rs][mn] = {
                "n": n_rs,
                "avg_entropy": round(sum(r.entropy for r in rs_results) / n_rs, 4),
                "avg_p_unsafe": round(sum(r.p_unsafe for r in rs_results) / n_rs, 4),
                "accuracy": round(sum(1 for r in rs_results if r.pred == r.gold_label) / n_rs, 4),
            }
    summary["by_risk_source"] = risk_agg

    summary_path = output_dir / "logprobs_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[logprobs] Summary: {summary_path}")

    # ── 5. Print comparison table ──
    print()
    print("=" * 120)
    print(f"  {'Model':<35} {'Fmt':>10} {'Acc':>7} {'AvgEnt':>8} {'AvgP(un)':>9} "
          f"{'Safe_Ent':>9} {'Unsafe_Ent':>10}")
    print("-" * 120)
    for mn in model_names:
        s = summary["models"].get(mn, {})
        if s.get("error"):
            print(f"  {mn:<35} {'ERROR':>10}")
            continue
        print(f"  {mn:<35} {s.get('format','?'):>10} {s['accuracy']:6.1%} {s['avg_entropy']:>8.4f} "
              f"{s['avg_p_unsafe']:>9.4f} {s.get('safe_avg_entropy', 0):>9.4f} "
              f"{s.get('unsafe_avg_entropy', 0):>10.4f}")
    print("=" * 120)

    # KL divergence matrix
    if len(model_names) > 1:
        print()
        print("KL Divergence Matrix (KL(row || col), averaged over cases):")
        # Short names for display
        short_names = [mn.split("--")[0][:20] for mn in model_names]
        col_w = max(len(sn) for sn in short_names) + 2
        print(f"  {'':>{col_w}}", end="")
        for sn in short_names:
            print(f" {sn:>{col_w}}", end="")
        print()
        for i, mn_a in enumerate(model_names):
            print(f"  {short_names[i]:>{col_w}}", end="")
            for mn_b in model_names:
                v = kl_matrix[mn_a][mn_b]
                if math.isnan(v):
                    print(f" {'nan':>{col_w}}", end="")
                else:
                    print(f" {v:>{col_w}.4f}", end="")
            print()
        print()


def run_batch_logprobs(
    config: BatchEvalConfig,
    *,
    limit_override: int | None = None,
    top_logprobs_n: int = 20,
) -> None:
    """Run logprobs mode: extract probabilities for all models, then compare."""
    from guardrail.prompts import PROFILE_REGISTRY

    # Load benchmark cases once
    benchmark_cls = _benchmark_registry()[config.benchmark]
    adapter_kwargs: dict[str, Any] = {}
    if config.input_path:
        adapter_kwargs["input_path"] = config.input_path
    effective_limit = limit_override if limit_override is not None else config.limit
    if effective_limit is not None:
        adapter_kwargs["limit"] = effective_limit
    adapter = benchmark_cls(**adapter_kwargs)
    cases = adapter.load_cases()
    print(f"[logprobs] Loaded {len(cases)} cases from {config.benchmark}")

    all_results: dict[str, list[LogprobsCaseResult]] = {}
    output_dir = Path(config.output_root) / "logprobs"

    # ── Process API models ──
    for model in config.api_models:
        prompt_name = model.prompt_name or config.prompt_name
        profile = PROFILE_REGISTRY.get(prompt_name)
        if profile is None:
            print(f"[logprobs] SKIP {model.name}: unknown prompt profile '{prompt_name}'")
            continue
        print(f"\n[logprobs] Running {model.name} (API: {model.base_url}) ...")
        results = _run_logprobs_for_model(
            model_name=model.name,
            served_model_name=model.model,
            base_url=model.base_url,
            api_key=model.api_key,
            cases=cases,
            profile=profile,
            top_logprobs_n=top_logprobs_n,
        )
        all_results[model.name] = results
        valid = [r for r in results if r.pred >= 0]
        correct = sum(1 for r in valid if r.pred == r.gold_label)
        print(f"  Done: {len(valid)} valid, acc={correct / len(valid):.1%}" if valid else "  Done: 0 valid")

    # ── Process vLLM models ──
    for model in config.vllm_models:
        prompt_name = model.prompt_name or config.prompt_name
        profile = PROFILE_REGISTRY.get(prompt_name)
        if profile is None:
            print(f"[logprobs] SKIP {model.name}: unknown prompt profile '{prompt_name}'")
            continue

        running: RunningVLLM | None = None
        try:
            running = start_vllm_process(model)
            base_url = f"http://{model.host}:{model.port}/v1"
            wait_for_http_ready(
                base_url,
                timeout_seconds=model.startup_timeout,
                expected_model=model.model,
                api_key=model.api_key,
                process=running.process,
            )
            print(f"\n[logprobs] Running {model.name} ...")
            results = _run_logprobs_for_model(
                model_name=model.name,
                served_model_name=model.model,
                base_url=base_url,
                api_key=model.api_key,
                cases=cases,
                profile=profile,
                top_logprobs_n=top_logprobs_n,
            )
            all_results[model.name] = results
            valid = [r for r in results if r.pred >= 0]
            correct = sum(1 for r in valid if r.pred == r.gold_label)
            print(f"  Done: {len(valid)} valid, acc={correct / len(valid):.1%}" if valid else "  Done: 0 valid")
        except Exception as exc:
            print(f"[logprobs] FAILED {model.name}: {exc}")
        finally:
            stop_vllm_process(running)

    # ── Generate comparison outputs ──
    if all_results:
        _write_logprobs_outputs(all_results, output_dir)
    else:
        print("[logprobs] No results collected.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch benchmark evaluation across API and vLLM models.")
    parser.add_argument("--config", required=True, help="Batch eval YAML config path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional CLI override for per-run limit.")
    parser.add_argument("--concurrency", type=int, default=None, help="Optional CLI override for per-run concurrency.")
    parser.add_argument("--logprobs", action="store_true", help="Logprobs mode: extract p(safe)/p(unsafe) and compare across models.")
    parser.add_argument("--top-logprobs", type=int, default=20, help="Number of top logprobs to request (default: 20).")
    return parser


def _print_summary(results: list[BatchRunResult]) -> None:
    from evals.reporting import (
        collect_result_summary_rows,
        print_batch_run_summary,
        print_static_result_table,
        write_result_index,
    )

    print_batch_run_summary(results)
    output_root = Path(results[0].output_root) if results else None
    if output_root is None or not output_root.exists():
        return
    rows = collect_result_summary_rows(output_root)
    if rows:
        write_result_index(output_root, rows)
        print_static_result_table(rows)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_batch_eval_config(args.config)
    validate_batch_eval_config(config)

    if args.logprobs:
        run_batch_logprobs(
            config,
            limit_override=args.limit,
            top_logprobs_n=args.top_logprobs,
        )
        return 0

    results = run_batch_eval(
        config,
        limit_override=args.limit,
        concurrency_override=args.concurrency,
    )
    _print_summary(results)
    return 0 if all(result.status == "success" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
