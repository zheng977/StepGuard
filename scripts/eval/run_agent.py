from __future__ import annotations

import argparse
import os

from agent.react import ReactAgent  # type: ignore[reportMissingImports]
from agent.tool_executor import ToolAdapter, ToolExecutor  # type: ignore[reportMissingImports]
from orchestrator import Orchestrator  # type: ignore[reportMissingImports]


class EchoTool(ToolAdapter):
    def call(self, arguments: dict) -> str:
        text = arguments.get("text", "")
        return f"echo: {text}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--api-base", type=str, default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    agent = ReactAgent.from_openai_compatible(
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )

    tools = ToolExecutor()
    tools.register("echo_tool", EchoTool())

    orchestrator = Orchestrator(agent=agent, tool_executor=tools, guardrail=None, max_steps=5)
    trajectory = orchestrator.run(user_request=args.query)

    for step in trajectory.history.steps:
        print(step.model_dump())


if __name__ == "__main__":
    main()
