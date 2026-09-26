# Third-party notices

The root Apache-2.0 license applies to StepGuard's original contributions. It does not supersede the licenses of vendored components or external dependencies. Existing copyright and license notices in third-party files must be retained.

| Included component | License | License text |
| --- | --- | --- |
| AgentDojo | MIT | [LICENSE](benchmark-repos/agentdojo/LICENSE) |
| AgentDyn | MIT, as included in the vendored snapshot | [LICENSE](benchmark-repos/AgentDyn/LICENSE) |
| AgentAuditor-ASSEBench | Apache-2.0 | [LICENSE](benchmark-repos/AgentAuditor-ASSEBench/LICENSE) |
| inspect_evals framework | MIT | [LICENSE](benchmark-repos/inspect_evals/LICENSE) |
| AgentHarm benchmark | MIT with an additional purpose restriction | [LICENSE](benchmark-repos/inspect_evals/src/inspect_evals/agentharm/LICENSE) |

The AgentHarm license prohibits using its dataset and benchmark for purposes besides improving the safety and security of AI systems. This is a term of that third-party benchmark, not an added restriction on StepGuard's original code, prompt templates, or model weights.

Source revisions and local changes are documented in [benchmark-repos/UPSTREAM.md](benchmark-repos/UPSTREAM.md). The inspect_evals framework license is copied from upstream revision `bb15e76de049c88d4ab43284b8e8359c0216988e`.

The model is derived from Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0). Its license, provenance, and fine-tuning notice are supplied with the [released weights](https://huggingface.co/ninty-seven/StepGuard). Separately obtained evaluation datasets, model weights, and dependencies retain their respective terms.
