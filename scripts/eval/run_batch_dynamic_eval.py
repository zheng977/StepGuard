"""Batch dynamic benchmark evaluation across multiple guard models.

Reads a YAML config, optionally starts/stops vLLM servers, and runs
run_dynamic_eval.py for each guard model sequentially.

Usage:
    python scripts/eval/run_batch_dynamic_eval.py --config configs/agentdyn_dynamic_eval.yaml
    python scripts/eval/run_batch_dynamic_eval.py --config configs/agentdyn_dynamic_eval.yaml --limit-suites shopping
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
AGENTDYN_SRC = REPO_ROOT / "benchmark-repos" / "AgentDyn" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(AGENTDYN_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTDYN_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse vLLM management from run_batch_eval.py
from scripts.eval.run_batch_eval import (
    VLLMModelConfig,
    RunningVLLM,
    start_vllm_process,
    stop_vllm_process,
    wait_for_http_ready,
    _expand_env,
    _coerce_path,
    _require_mapping,
)


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _require_mapping(raw, name="dynamic eval config")


def build_run_command(
    config: dict[str, Any],
    *,
    backend: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    prompt_name: str | None = None,
    prompt_file: str | None = None,
    response_parser: str | None = None,
    limit_suites: list[str] | None = None,
    subset_override: str | None = None,
    attack_override: str | None = None,
    no_guard: bool = False,
) -> list[str]:
    eval_python = _expand_env(str(config.get("eval_python") or sys.executable))
    cmd = [
        eval_python,
        str(REPO_ROOT / "scripts" / "eval" / "run_dynamic_eval.py"),
        "--benchmark", str(config["dynamic_benchmark"]),
        "--output-root", str(config.get("output_root", "results")),
        "--temperature", str(config.get("temperature", 0.0)),
        "--max-tokens", str(config.get("max_tokens", 4096)),
        "--timeout", str(config.get("timeout", 120)),
        "--blocking-mode", str(config.get("blocking_mode", "continue")),
        "--confidence-threshold", str(config.get("confidence_threshold", 0.5)),
        "--response-parser", str(response_parser or config.get("response_parser", "stepguard")),
    ]

    if no_guard:
        cmd.append("--no-guard")
    else:
        if model is None or base_url is None or api_key is None:
            raise ValueError("Guard model, base_url, and api_key are required unless no_guard=True.")
        cmd.extend([
            "--model", model,
            "--backend", backend,
            "--base-url", base_url,
            "--api-key", api_key,
        ])

    if config.get("generic_feedback"):
        cmd.append("--generic-feedback")
    else:
        cmd.extend(["--feedback-mode", str(config.get("feedback_mode", "self_reflect"))])
    if config.get("blocked_history_mode"):
        cmd.extend(["--blocked-history-mode", str(config["blocked_history_mode"])])
    if config.get("max_replans") is not None:
        cmd.extend(["--max-replans", str(config["max_replans"])])
    if config.get("guard_reconsideration"):
        cmd.extend(["--guard-reconsideration", str(config["guard_reconsideration"])])
    if config.get("record_full_guard_context"):
        cmd.append("--record-full-guard-context")
    if config.get("log_level"):
        cmd.extend(["--log-level", str(config["log_level"])])
    if config.get("agent_system_suffix"):
        cmd.extend(["--agent-system-suffix", str(config["agent_system_suffix"])])

    # Prompt
    if prompt_file:
        cmd.extend(["--prompt-file", prompt_file])
    else:
        cmd.extend(["--prompt-name", str(prompt_name or config.get("prompt_name", "stepguard"))])

    # AgentDyn/AgentDojo-specific
    if config["dynamic_benchmark"] in {"agentdyn", "agentdojo"}:
        cmd.extend(["--agent-model", _expand_env(str(config.get("agent_model", "gpt-4o-2024-08-06")))])
        if config.get("agent_api_key"):
            cmd.extend(["--agent-api-key", _expand_env(str(config["agent_api_key"]))])
        if config.get("agent_port"):
            cmd.extend(["--agent-port", str(config["agent_port"])])
        if config.get("agent_base_url"):
            cmd.extend(["--agent-base-url", _expand_env(str(config["agent_base_url"]))])
        if config.get("agent_server_json"):
            cmd.extend(["--agent-server-json", _expand_env(str(config["agent_server_json"]))])
        if config.get("agent_no_proxy"):
            cmd.append("--agent-no-proxy")
        if config.get("proxy"):
            cmd.extend(["--proxy", _expand_env(str(config["proxy"]))])
        if config.get("concurrency"):
            cmd.extend(["--concurrency", str(config["concurrency"])])
        attack = attack_override if attack_override is not None else config.get("attack")
        if attack:
            cmd.extend(["--attack", str(attack)])
        if config.get("run_benign_with_attack") is False:
            cmd.append("--attack-only")
        if config.get("skip_injection_precheck"):
            cmd.append("--skip-injection-precheck")
        if config.get("benchmark_version"):
            cmd.extend(["--benchmark-version", str(config["benchmark_version"])])
        if config.get("logdir"):
            cmd.extend(["--logdir", _expand_env(str(config["logdir"]))])
        if config["dynamic_benchmark"] == "agentdojo" and config.get("agentdojo_source_root"):
            cmd.extend(["--agentdojo-source-root", _expand_env(str(config["agentdojo_source_root"]))])

        default_suites = ["workspace", "travel", "banking", "slack"] if config["dynamic_benchmark"] == "agentdojo" else ["shopping", "github", "dailylife"]
        suites = limit_suites or config.get("suites") or default_suites
        for s in suites:
            cmd.extend(["-s", str(s)])

        for ut in config.get("user_tasks") or []:
            cmd.extend(["-ut", str(ut)])
        for it in config.get("injection_tasks") or []:
            cmd.extend(["-it", str(it)])

    # AgentHarm-specific
    if config["dynamic_benchmark"] == "agentharm":
        cmd.extend(["--agent-model", _expand_env(str(config.get("agent_model", "gpt-4o-2024-08-06")))])
        if config.get("agent_api_key"):
            cmd.extend(["--agent-api-key", _expand_env(str(config["agent_api_key"]))])
        if config.get("agent_base_url"):
            cmd.extend(["--agent-base-url", _expand_env(str(config["agent_base_url"]))])
        if config.get("agent_server_json"):
            cmd.extend(["--agent-server-json", _expand_env(str(config["agent_server_json"]))])
        if config.get("agent_no_proxy"):
            cmd.append("--agent-no-proxy")
        if config.get("proxy"):
            cmd.extend(["--proxy", _expand_env(str(config["proxy"]))])
        if config.get("concurrency"):
            cmd.extend(["--concurrency", str(config["concurrency"])])
        subset = subset_override or config.get("subset")
        if subset:
            cmd.extend(["--subset", str(subset)])
        if config.get("max_turns"):
            cmd.extend(["--max-turns", str(config["max_turns"])])
        if config.get("dataset_path"):
            cmd.extend(["--dataset-path", _expand_env(str(config["dataset_path"]))])
        if config.get("agentharm_source_root"):
            cmd.extend(["--agentharm-source-root", _expand_env(str(config["agentharm_source_root"]))])
        if config.get("tools_root"):
            cmd.extend(["--tools-root", _expand_env(str(config["tools_root"]))])
        if config.get("graders_module_path"):
            cmd.extend(["--graders-module-path", _expand_env(str(config["graders_module_path"]))])
        if config.get("n_irrelevant_tools"):
            cmd.extend(["--n-irrelevant-tools", str(config["n_irrelevant_tools"])])
        if config.get("judge_model"):
            cmd.extend(["--judge-model", _expand_env(str(config["judge_model"]))])
        if config.get("judge_base_url"):
            cmd.extend(["--judge-base-url", _expand_env(str(config["judge_base_url"]))])
        if config.get("judge_server_json"):
            cmd.extend(["--judge-server-json", _expand_env(str(config["judge_server_json"]))])
        if config.get("judge_api_key"):
            cmd.extend(["--judge-api-key", _expand_env(str(config["judge_api_key"]))])
        if config.get("limit"):
            cmd.extend(["--limit", str(config["limit"])])
        for bid in config.get("behavior_ids") or []:
            cmd.extend(["-bid", str(bid)])

    return cmd


def build_child_env(config: dict[str, Any]) -> dict[str, str]:
    """Build environment for subprocess eval runs."""
    env = os.environ.copy()
    default_no_user_site = str(config.get("python_no_user_site", "1"))
    env["PYTHONNOUSERSITE"] = env.get("PYTHONNOUSERSITE", default_no_user_site)
    pythonpath_parts = [str(SRC_ROOT), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if config.get("record_full_guard_context"):
        env["AGENTGUARD_RECORD_FULL_PROMPT"] = "1"
    return env


def _attacks_to_run(config: dict[str, Any]) -> list[str | None]:
    """Return attack variants for benchmarks that support injection attacks."""
    if config.get("dynamic_benchmark") not in {"agentdojo", "agentdyn"}:
        return [None]
    attacks = list(config.get("attacks") or [])
    if attacks:
        return [str(a) for a in attacks]
    if config.get("attack"):
        return [str(config["attack"])]
    return [None]


def _config_for_attack(config: dict[str, Any], attack: str | None, total_attacks: int) -> dict[str, Any]:
    """Copy config and isolate output/log dirs when sweeping attacks."""
    next_config = dict(config)
    if attack is not None:
        next_config["attack"] = attack
    if "attacks" in next_config:
        next_config.pop("attacks", None)

    if attack is not None and total_attacks > 1:
        base_output = Path(str(config.get("output_root", "results")))
        next_config["output_root"] = str(base_output / "attacks" / attack)
        if config.get("logdir"):
            next_config["logdir"] = str(Path(str(config["logdir"])) / attack)
    return next_config


def _variant_label(base: str, *, subset: str | None = None, attack: str | None = None) -> str:
    suffixes = []
    if attack:
        suffixes.append(f"attack={attack}")
    if subset:
        suffixes.append(f"subset={subset}")
    if not suffixes:
        return base
    return f"{base} [{' '.join(suffixes)}]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch dynamic benchmark evaluation.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--limit-suites", nargs="+", default=None, help="Override suites to run.")
    args = parser.parse_args()

    config = load_config(args.config)
    child_env = build_child_env(config)
    results: list[dict[str, Any]] = []
    attacks_to_run = _attacks_to_run(config)

    # AgentHarm can run multiple subsets in one go (harmful + benign).
    # Falls back to singular `subset`, then to [None] for non-AgentHarm benchmarks.
    subsets_to_run: list[str | None] = (
        list(config.get("subsets") or [])
        or ([config["subset"]] if config.get("subset") else [])
        or [None]
    )

    if config.get("run_no_guard"):
        print(f"\n{'='*60}")
        print("  Running baseline: no_defense")
        print(f"{'='*60}")

        for attack in attacks_to_run:
            run_config = _config_for_attack(config, attack, len(attacks_to_run))
            for subset in subsets_to_run:
                tag = _variant_label("no_defense", subset=subset, attack=attack)
                print(f"\n  --- run: {tag} ---")
                cmd = build_run_command(
                    run_config,
                    backend="none",
                    prompt_name=run_config.get("prompt_name"),
                    response_parser=run_config.get("response_parser"),
                    limit_suites=args.limit_suites,
                    subset_override=subset,
                    attack_override=attack,
                    no_guard=True,
                )
                completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=False, env=child_env)
                results.append({
                    "name": tag, "backend": "none",
                    "status": "success" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                })

    # Process API models
    for raw_model in config.get("api_models") or []:
        model = _require_mapping(raw_model, name="api model")
        name = str(model["name"])
        print(f"\n{'='*60}")
        print(f"  Running guard model: {name} (API)")
        print(f"{'='*60}")

        for attack in attacks_to_run:
            run_config = _config_for_attack(config, attack, len(attacks_to_run))
            for subset in subsets_to_run:
                tag = _variant_label(name, subset=subset, attack=attack)
                print(f"\n  --- run: {tag} ---")
                cmd = build_run_command(
                    run_config,
                    backend="api",
                    model=_expand_env(str(model["model"])),
                    base_url=_expand_env(str(model["base_url"])),
                    api_key=_expand_env(str(model["api_key"])),
                    prompt_name=model.get("prompt_name"),
                    prompt_file=_coerce_path(model.get("prompt_file"), base_dir=REPO_ROOT),
                    response_parser=model.get("response_parser"),
                    limit_suites=args.limit_suites,
                    subset_override=subset,
                    attack_override=attack,
                )
                completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=False, env=child_env)
                results.append({
                    "name": tag, "backend": "api",
                    "status": "success" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                })

    # Process vLLM models
    vllm_defaults = config.get("vllm_defaults") or {}
    for raw_model in config.get("vllm_models") or []:
        model = _require_mapping(raw_model, name="vllm model")
        name = str(model["name"])
        model_path = str(Path(_expand_env(str(model["model_path"]))).expanduser())

        # Per-model extra args + default max-num-seqs to widen vLLM batch.
        default_extra = list(vllm_defaults.get("vllm_extra_args") or [])
        extra_args = list(model.get("vllm_extra_args") or default_extra)
        if not any(a == "--max-num-seqs" for a in extra_args):
            max_seqs = model.get("max_num_seqs") or vllm_defaults.get("max_num_seqs") or 128
            extra_args += ["--max-num-seqs", str(max_seqs)]

        vllm_config = VLLMModelConfig(
            name=name,
            model=_expand_env(str(model["model"])),
            model_path=model_path,
            tensor_parallel_size=int(model.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(model.get("gpu_memory_utilization", 0.9)),
            api_key=_expand_env(str(model.get("api_key", vllm_defaults.get("api_key", "EMPTY")))),
            host=_expand_env(str(model.get("host", vllm_defaults.get("host", "127.0.0.1")))),
            port=int(model.get("port", vllm_defaults.get("port", 18000))),
            max_model_len=int(model.get("max_model_len", vllm_defaults.get("max_model_len", 32768))),
            dtype=_expand_env(str(model.get("dtype", vllm_defaults.get("dtype", "auto")))),
            startup_timeout=int(model.get("startup_timeout", vllm_defaults.get("startup_timeout", 600))),
            vllm_extra_args=extra_args,
        )

        print(f"\n{'='*60}")
        print(f"  Running guard model: {name} (vLLM)")
        print(f"{'='*60}")

        running: RunningVLLM | None = None
        try:
            running = start_vllm_process(vllm_config)
            base_url = f"http://{vllm_config.host}:{vllm_config.port}/v1"
            wait_for_http_ready(
                base_url,
                timeout_seconds=vllm_config.startup_timeout,
                expected_model=vllm_config.model,
                api_key=vllm_config.api_key,
                process=running.process,
            )

            for attack in attacks_to_run:
                run_config = _config_for_attack(config, attack, len(attacks_to_run))
                for subset in subsets_to_run:
                    tag = _variant_label(name, subset=subset, attack=attack)
                    print(f"\n  --- run: {tag} ---")
                    cmd = build_run_command(
                        run_config,
                        backend="vllm",
                        model=vllm_config.model,
                        base_url=base_url,
                        api_key=vllm_config.api_key,
                        prompt_name=model.get("prompt_name"),
                        prompt_file=_coerce_path(model.get("prompt_file"), base_dir=REPO_ROOT),
                        response_parser=model.get("response_parser"),
                        limit_suites=args.limit_suites,
                        subset_override=subset,
                        attack_override=attack,
                    )
                    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=False, env=child_env)
                    results.append({
                        "name": tag, "backend": "vllm",
                        "status": "success" if completed.returncode == 0 else "failed",
                        "returncode": completed.returncode,
                    })
        except Exception as exc:
            results.append({
                "name": name, "backend": "vllm",
                "status": "failed", "error": str(exc),
            })
        finally:
            stop_vllm_process(running)

    # Summary
    print(f"\n{'='*60}")
    print("  Batch Dynamic Eval Summary")
    print(f"{'='*60}")
    for r in results:
        status_icon = "OK" if r["status"] == "success" else "FAIL"
        print(f"  [{status_icon}] {r['name']} ({r['backend']})")
        if r.get("error"):
            print(f"       Error: {r['error']}")
    print()

    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
