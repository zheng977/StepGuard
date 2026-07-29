"""Released prompt profiles for StepGuard and evaluation baselines."""

from __future__ import annotations

from ag_types import Action, GuardrailContext, InteractionHistory

from .agentdog import (
    AGENTDOG_JSON_PROMPT_NAME,
    AGENTDOG_TRAJ_PROMPT_NAME,
    AgentDogJsonProfile,
    AgentDogTrajProfile,
)
from .base import (
    DEFAULT_PROMPT_NAME,
    PromptProfile,
    _parse_binary_judgment,
    _parse_fg_labels,
    build_prompt_tag,
    compute_prompt_hash,
)
from .baselines import (
    QWEN3GUARD_PROMPT_NAME,
    QWEN3GUARD_TRAJ_PROMPT_NAME,
    SHIELDAGENT_PROMPT_NAME,
    SHIELDAGENT_TRAJ_PROMPT_NAME,
    TSGUARD_DYNAMIC_PROMPT_NAME,
    TSGUARD_PROMPT_NAME,
    TSGUARD_STRICT_PROMPT_NAME,
    TSGUARD_TRAJ_PROMPT_NAME,
    TSGUARD_TRAJ_STRICT_PROMPT_NAME,
    Qwen3GuardProfile,
    Qwen3GuardTrajProfile,
    ShieldAgentProfile,
    ShieldAgentTrajProfile,
    TSGuardDynamicProfile,
    TSGuardProfile,
    TSGuardStrictProfile,
    TSGuardTrajProfile,
    TSGuardTrajStrictProfile,
)
from .stepguard import (
    STEPGUARD_PROMPT_NAME,
    STEPGUARD_TRAJ_PROMPT_NAME,
    StepGuardProfile,
    StepGuardTrajProfile,
)


PROFILE_REGISTRY: dict[str, PromptProfile] = {
    # Released StepGuard family.
    STEPGUARD_PROMPT_NAME: StepGuardProfile(),
    STEPGUARD_TRAJ_PROMPT_NAME: StepGuardTrajProfile(),
    # Native prompt contracts used by the reported baselines.
    TSGUARD_PROMPT_NAME: TSGuardProfile(),
    TSGUARD_STRICT_PROMPT_NAME: TSGuardStrictProfile(),
    TSGUARD_DYNAMIC_PROMPT_NAME: TSGuardDynamicProfile(),
    TSGUARD_TRAJ_PROMPT_NAME: TSGuardTrajProfile(),
    TSGUARD_TRAJ_STRICT_PROMPT_NAME: TSGuardTrajStrictProfile(),
    SHIELDAGENT_PROMPT_NAME: ShieldAgentProfile(),
    SHIELDAGENT_TRAJ_PROMPT_NAME: ShieldAgentTrajProfile(),
    QWEN3GUARD_PROMPT_NAME: Qwen3GuardProfile(),
    QWEN3GUARD_TRAJ_PROMPT_NAME: Qwen3GuardTrajProfile(),
    AGENTDOG_JSON_PROMPT_NAME: AgentDogJsonProfile(),
    AGENTDOG_TRAJ_PROMPT_NAME: AgentDogTrajProfile(),
}

PROMPT_REGISTRY: dict[str, str] = {
    name: profile.template for name, profile in PROFILE_REGISTRY.items()
}


def get_predictive_prompt_template(prompt_name: str = DEFAULT_PROMPT_NAME) -> str:
    normalized_name = str(prompt_name).strip() or DEFAULT_PROMPT_NAME
    try:
        return PROMPT_REGISTRY[normalized_name]
    except KeyError as exc:
        raise ValueError(
            f'Unknown prompt "{prompt_name}". Expected one of: '
            f'{", ".join(sorted(PROMPT_REGISTRY))}.'
        ) from exc


def build_predictive_prompt(
    *,
    history: InteractionHistory,
    action: Action,
    context: GuardrailContext | None = None,
    prompt_template: str | None = None,
    prompt_name: str = DEFAULT_PROMPT_NAME,
) -> str:
    normalized_name = str(prompt_name).strip() or DEFAULT_PROMPT_NAME
    profile = PROFILE_REGISTRY.get(normalized_name)
    if profile is None:
        if prompt_template is None:
            get_predictive_prompt_template(normalized_name)
        profile = PROFILE_REGISTRY[STEPGUARD_PROMPT_NAME]
    template = prompt_template if prompt_template is not None else profile.template
    return profile.build_prompt(
        history=history,
        action=action,
        context=context,
        template=template,
    )


__all__ = [
    "DEFAULT_PROMPT_NAME",
    "PROFILE_REGISTRY",
    "PROMPT_REGISTRY",
    "build_predictive_prompt",
    "build_prompt_tag",
    "compute_prompt_hash",
    "get_predictive_prompt_template",
    "_parse_binary_judgment",
    "_parse_fg_labels",
]
