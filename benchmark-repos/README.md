# benchmark-repos/

This directory vendors the third-party source required to reproduce the
released dynamic evaluations:

- `AgentDyn/` for AgentDyn;
- `agentdojo/` for AgentDojo;
- `inspect_evals/` for AgentHarm.
- `AgentAuditor-ASSEBench/` for ASSEBench static evaluation.

See [UPSTREAM.md](UPSTREAM.md) for source URLs, pinned revisions, licenses,
and local-patch notes. The three directories are source dependencies, not
generated artifacts. Their original license files are retained.

Other benchmark checkouts, downloaded benchmark payloads, caches, notebooks,
and run outputs are intentionally ignored. Add a new third-party dependency
only when it is required by a released evaluation entry point, and document its
upstream revision and license in `UPSTREAM.md`.
