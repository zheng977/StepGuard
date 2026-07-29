# Vendored Dynamic Benchmark Sources

This release vendors only the third-party sources used by the public dynamic
evaluation adapters. Their original license files remain in their respective
directories.

| Directory | Used by | Upstream | Pinned revision | License | Local notes |
|---|---|---|---|---|---|
| `AgentDyn` | `agentdyn` | AgentDyn official implementation | Local vendored snapshot | MIT | Contains the AgentGuard pre-action hook used by the released adapter. Runtime outputs under `runs/` are excluded. |
| `AgentAuditor-ASSEBench` | `assebench` | https://github.com/Astarojth/AgentAuditor-ASSEBench | `b1204ee` | See upstream repository | Restored from the public upstream at the pinned revision. |
| `agentdojo` | `agentdojo` | https://github.com/ethz-spylab/agentdojo | `3dce07eb3c10f16ff5af4e32d186fe3b1da6cc1f` | MIT | Includes local compatibility updates in `src/agentdojo/agent_pipeline/llms/openai_llm.py`, `src/agentdojo/models.py`, and `src/agentdojo/task_suite/task_suite.py`. |
| `inspect_evals` | `agentharm` | https://github.com/UKGovernmentBEIS/inspect_evals | `bb15e76de049c88d4ab43284b8e8359c0216988e` | See upstream repository | No local modifications at vendoring time. |

The original projects are independently maintained. Changes to these vendored
directories should be kept minimal and documented in this file.
