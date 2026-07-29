"""Run a dynamic benchmark evaluation with a single guard model.

Usage:
    python scripts/eval/run_dynamic_eval.py \
        --benchmark agentdyn \
        --model gpt-4o-mini \
        --api-key $OPENAI_API_KEY \
        --prompt-name stepguard \
        --agent-model gpt-4o-2024-08-06 \
        --suite shopping \
        --attack important_instructions \
        --output-root results/agentdyn_dynamic

    # Benign utility test (no attack)
    python scripts/eval/run_dynamic_eval.py \
        --benchmark agentdyn \
        --model gpt-4o-mini \
        --agent-model gpt-4o-2024-08-06 \
        --suite shopping

    # Single task debug
    python scripts/eval/run_dynamic_eval.py \
        --benchmark agentdyn \
        --model gpt-4o-mini \
        --agent-model gpt-4o-2024-08-06 \
        --suite shopping \
        --user-task user_task_0 \
        --attack important_instructions \
        --injection-task injection_task_0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
AGENTDYN_SRC = REPO_ROOT / "benchmark-repos" / "AgentDyn" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(AGENTDYN_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTDYN_SRC))


def _dynamic_benchmark_choices() -> list[str]:
    try:
        from evals.dynamic import DYNAMIC_BENCHMARK_REGISTRY
    except ModuleNotFoundError:
        return ["agentdyn", "agentdojo", "agentharm"]
    return list(DYNAMIC_BENCHMARK_REGISTRY)


def _response_parser_choices() -> list[str]:
    choices = ["auto", "strict", "safe_unsafe"]
    try:
        from guardrail.prompts import PROFILE_REGISTRY
    except ModuleNotFoundError:
        return choices
    return choices + list(PROFILE_REGISTRY)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run dynamic benchmark evaluation.")

    # Guard model config (same as run_eval.py)
    p.add_argument("--benchmark", required=True, choices=_dynamic_benchmark_choices())
    p.add_argument("--model", default=None, help="Guard model name. Omit or use --no-guard for baseline.")
    p.add_argument("--no-guard", action="store_true", default=False, help="Run without guard (pure agent baseline).")
    p.add_argument("--backend", default="api", choices=["api", "vllm"])
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--prompt-name", default="stepguard")
    p.add_argument("--prompt-file", default=None)
    p.add_argument(
        "--response-parser",
        default="stepguard",
        help="Response parser name. Built-ins: auto, strict, safe_unsafe; prompt profiles "
             "such as stepguard/tsguard are validated when the guardrail is built.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--output-root", default="results")

    # Guard behavior
    p.add_argument("--blocking-mode", default="continue", choices=["continue", "abort"])
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--generic-feedback", action="store_true", default=False,
                   help="Legacy: shortcut for --feedback-mode generic.")
    p.add_argument("--feedback-mode", default="self_reflect",
                   choices=["detailed", "generic", "guard_guided", "self_reflect", "appeal", "silent"],
                   help="Block-message style. Default self_reflect is the paper setting. "
                        "detailed=brief fixed block notice; generic=ToolSafe-minimal; "
                        "guard_guided=RiskSource/category/rating-conditioned concise feedback; "
                        "self_reflect=agent self-diagnose + propose alternative; "
                        "appeal=utility-biased retry guidance for possible false-positive blocks; "
                        "silent=empty tool result with no safety feedback. Overrides --generic-feedback.")
    p.add_argument("--blocked-history-mode", default="clean",
                   choices=["clean", "toolsafe", "last_blocked"],
                   help="How soft-blocked attempts appear to the next agent turn. "
                        "clean hides the blocked attempt; toolsafe keeps it visible and "
                        "returns a synthetic safety-check observation; last_blocked shows "
                        "only the immediately blocked attempt for one replan turn.")
    p.add_argument("--max-replans", type=int, default=3,
                   help="Max soft-block replans per task. Default 3 means the fourth "
                        "blocked replan is allowed to execute. Use -1 to disable.")
    p.add_argument("--guard-reconsideration", default="off", choices=["off", "second_pass"],
                   help="Optional second-pass guard check before blocking. second_pass reruns "
                        "the guard with a calibration suffix when the first pass is unsafe; "
                        "the action is blocked only if the second pass is also unsafe.")
    p.add_argument("--record-full-guard-context", action="store_true", default=False,
                   help="Record full guard input context for every guard judgment. "
                        "Includes user request, available tools, serialized history, "
                        "current action, guard prompt/messages, and raw response in "
                        "records/events. This can make result artifacts much larger.")

    # AgentDyn/AgentDojo-specific (ignored for other dynamic benchmarks)
    p.add_argument("--agent-model", default="gpt-4o-2024-08-06", help="Agent LLM name (API model or vLLM served name).")
    p.add_argument("--agent-api-key", default=None, help="Agent LLM API key (defaults to --api-key).")
    p.add_argument("--agent-base-url", default=None, help="Agent LLM base URL (set for vLLM/local models).")
    p.add_argument("--agent-server-json", default=None,
                   help="Path to a vLLM server.json. If this process is on the serving host, "
                        "the agent endpoint resolves to localhost; otherwise it resolves to a LAN endpoint.")
    p.add_argument("--agent-port", type=int, default=None, help="Agent LLM vLLM port (shorthand for --agent-base-url).")
    p.add_argument("--agent-no-proxy", action="store_true", default=False, help="Agent is local, bypass proxy.")
    p.add_argument("--agent-system-suffix", default=None,
                   help="Prompt-only baseline text appended to the agent system prompt, e.g. Sandwich.")
    p.add_argument("--proxy", default=None, help="HTTP proxy for guard model (e.g. http://127.0.0.1:7890).")
    p.add_argument("--concurrency", type=int, default=1, help="Number of parallel tasks (default: 1).")
    p.add_argument("--suite", "-s", action="append", dest="suites", help="Suite(s) to run. Repeatable.")
    p.add_argument("--attack", default=None, help="Attack type (e.g. important_instructions).")
    p.add_argument("--attack-only", action="store_false", dest="run_benign_with_attack", default=True,
                   help="For AgentDyn/AgentDojo attack runs, skip the benign no-attack phase. "
                        "Use this when benign utility is evaluated in a separate run.")
    p.add_argument("--skip-injection-precheck", action="store_true", default=False,
                   help="AgentDyn only: skip the pre-check that verifies injection tasks "
                        "are solvable as user tasks before running attack pairs.")
    p.add_argument("--user-task", "-ut", action="append", dest="user_tasks")
    p.add_argument("--injection-task", "-it", action="append", dest="injection_tasks")
    p.add_argument("--benchmark-version", default="v1.2.2")
    p.add_argument("--logdir", default=None, help="AgentDyn trace log directory.")
    p.add_argument("--agentdojo-source-root", default=None,
                   help="Path to original AgentDojo src root. Defaults to benchmark-repos/agentdojo/src.")

    # AgentHarm-specific (ignored for other benchmarks)
    p.add_argument("--subset", default=None, choices=["harmful", "benign"], help="AgentHarm subset.")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--agentharm-source-root", default=None,
                   help="Path to inspect_evals/src. Defaults to benchmark-repos/inspect_evals/src.")
    p.add_argument("--tools-root", default=None)
    p.add_argument("--graders-module-path", default=None)
    p.add_argument("--n-irrelevant-tools", type=int, default=0,
                   help="AgentHarm: add N irrelevant official tools to each behavior.")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--judge-server-json", default=None,
                   help="Path to a vLLM server.json for the judge endpoint. "
                        "Uses localhost on the serving host, otherwise a LAN endpoint.")
    p.add_argument("--judge-api-key", default=None)
    p.add_argument("--behavior-id", "-bid", action="append", dest="behavior_ids")
    p.add_argument("--limit", type=int, default=None, help="Limit N behaviors (AgentHarm).")

    # Logging
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Root log level. DEBUG shows per-step agent trace.")

    return p


def main() -> int:
    import logging as _logging

    args = build_parser().parse_args()
    if args.generic_feedback:
        # Preserve the legacy shortcut even though self_reflect is now default.
        args.feedback_mode = "generic"

    from evals.dynamic.config import DynamicEvalConfig
    from evals.dynamic.runner import run_dynamic_eval

    # Root logger covers dynamic benchmark adapters and the guard runner.
    _logging.basicConfig(
        level=getattr(_logging, args.log_level.upper(), _logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    run_dynamic_eval(DynamicEvalConfig.from_namespace(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
