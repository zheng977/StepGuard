from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.base import BaseAgent
from agent.tool_executor import ToolExecutor
from ag_types import Action, InteractionHistory, Judgment, Observation, Trajectory
from guardrail import BaseGuardrail


@dataclass
class Orchestrator:
    agent: BaseAgent
    tool_executor: ToolExecutor
    guardrail: Optional[BaseGuardrail] = None
    max_steps: int = 10

    def run(self, user_request: str, initial_state: str = "") -> Trajectory:
        history = InteractionHistory(user_request=user_request, initial_state=initial_state)
        judgments: list[Judgment] = []

        for step in range(1, self.max_steps + 1):
            action = self.agent.act(history)
            action.step = step
            history.add_action(action)

            if action.is_final:
                break

            if self.guardrail is not None:
                judgment = self.guardrail.judge(action=action, history=history)
                judgments.append(judgment)
                if judgment.label == 1:
                    blocked = Observation(
                        content=f"[BLOCKED] {judgment.reason or 'unsafe action'}",
                        step=step,
                    )
                    history.add_observation(blocked)
                    self.agent.receive_observation(blocked)
                    continue

            observation = self.tool_executor.execute(action)
            history.add_observation(observation)
            self.agent.receive_observation(observation)

        return Trajectory(history=history, judgments=judgments)
