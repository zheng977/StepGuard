# StepGuard: Learning Step-Level Guardrails with Scalable Supervision and Safety-Utility Balancing

<p align="center"><strong>EMNLP 2026</strong></p>

<p align="center">
  🌐 <a href="https://zheng977.github.io/StepGuard/">Project Page</a>
  &nbsp;|&nbsp; 🤗 <a href="https://huggingface.co/ninty-seven/StepGuard">Model</a>
  &nbsp;|&nbsp; 💻 <a href="https://github.com/zheng977/StepGuard">Code</a>
</p>

> **StepGuard** is a 4B guard model that checks tool actions before execution
> and audits completed agent trajectories. It is trained with **StepGen** for
> scalable step-level supervision and **Balance-GRPO** for balancing safety and
> utility.

## 🔥 News

- **[2026-08-25]** StepGuard is released with model weights, evaluation code,
  the training recipe, and the project page.
- **[2026]** StepGuard is accepted to **EMNLP 2026**.

<p align="center">
  <img src="assets/step_level_guarding_motivation.png" width="780" alt="Why a trajectory-level guard cannot intervene before an injected unsafe action"/>
</p>

## ✨ Highlights

| Component | What it provides |
|---|---|
| **StepGuard** | One model for pre-execution action checking and post-execution trajectory auditing. |
| **StepGen** | Prefix-aligned safe/unsafe trajectories with risky-step localization and benign tool reuse. |
| **Balance-GRPO** | Dynamic advantage reweighting based on the observed safe/unsafe accuracy gap. |

## 🧭 Method

<p align="center">
  <img src="assets/stepguard_overview.png" width="960" alt="StepGuard training pipeline: prefix-aligned SFT followed by Balance-GRPO calibration"/>
</p>

## 🤗 Model and Release Status

| Artifact | Link | Status |
|---|---|---|
| **StepGuard-4B** | [ninty-seven/StepGuard](https://huggingface.co/ninty-seven/StepGuard) | ✅ Released |
| SFT-3K / RL-4K corpus | -- | Planned |
| StepGen data-generation engine | -- | Planned |

## 📊 Main Results

### Static safety evaluation

StepGuard achieves the highest average accuracy among the evaluated
open-weight guard models. Its trajectory-level average accuracy reaches
**83.0**, matching GPT-5.4, while its step-level average accuracy reaches
**84.8**, compared with **81.3** for GPT-5.4.

<p align="center">
  <img src="assets/static_evaluation_results.png" width="1000" alt="Static trajectory-level and step-level safety evaluation results"/>
</p>

### Runtime guarded-agent evaluation

When guarding agents on AgentDojo and AgentDyn, StepGuard lowers mean ASR
from **23.15%** without a guard to **5.25%**—a **17.9-point absolute reduction**
and a **77.3% relative reduction**—while mean utility decreases by only
**2.8 points**. It also obtains a **3.4** malicious score on AgentHarm.

<p align="center">
  <img src="assets/runtime_safety_utility.png" width="1000" alt="Runtime safety and utility trade-off on AgentDojo, AgentDyn, and AgentHarm"/>
</p>

## 🚀 Quick Start

### 1. Environment

```bash
git clone https://github.com/zheng977/StepGuard.git
cd StepGuard
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,serve]

pip install "huggingface_hub[cli]"
```

### 2. Download the model

```bash
huggingface-cli download ninty-seven/StepGuard \
  --local-dir "$PWD/artifacts/StepGuard"
```

### 3. Prepare benchmark data

The repository includes evaluation adapters, configurations, and the source
dependencies for dynamic benchmarks. Static benchmark payloads are not
redistributed: obtain them from their respective upstream projects and place
them at the paths referenced by
`configs/eval_suites/static_core.example.yaml`.

### 4. Validate the static suite

```bash
python scripts/eval/run_eval_suite.py \
  --config configs/eval_suites/static_core.example.yaml \
  --dry-run
```

### 5. Run static evaluation

```bash
export AGENTGUARD_MODEL_PATH="$PWD/artifacts/StepGuard"
CUDA_VISIBLE_DEVICES=0,1 \
  python scripts/eval/run_eval_suite.py \
  --config configs/eval_suites/static_core.example.yaml
```

The suite uses `stepguard` for action-level TS-Bench inputs and
`stepguard_traj` for trajectory-level inputs. It uses greedy decoding and
writes predictions and summaries under `results/`.

## 🧪 Dynamic Evaluation

The paper protocol uses `self_reflect` feedback, clean blocked-action history,
a `0.5` confidence threshold, and at most three replans. Start an
OpenAI-compatible agent endpoint and guard endpoint, then run:

```bash
export AGENT_MODEL=<agent-model-name>
export AGENT_BASE_URL=http://127.0.0.1:8000/v1
export GUARD_MODEL=<guard-model-name>
export GUARD_BASE_URL=http://127.0.0.1:8001/v1
export GUARD_API_KEY=EMPTY

python scripts/eval/run_batch_dynamic_eval.py \
  --config configs/dynamic/self_reflect.example.yaml
```

Detailed protocol and metrics: [Dynamic Evaluation](docs-open/dynamic_evaluation.md).

## 🏋️ Training

The published recipe starts from `Qwen/Qwen3-4B-Instruct-2507`, performs
two-epoch full-parameter SFT, then runs 100 Balance-GRPO rollout updates.
LLaMA-Factory and SLIME remain external dependencies. The SFT launcher,
Balance-GRPO implementation, SLIME adapter, and hyperparameter recipe are in
`training/release/`.

The final SFT-3K/RL-4K corpus and StepGen data-generation engine are planned
for a subsequent release. Until then, the released checkpoints support full
inference and evaluation reproduction.

## 📚 Documentation

- [End-to-End Reproduction](docs-open/reproduction.md)
- [Prompt Contract](docs-open/prompts.md)
- [Static and Dynamic Evaluation](docs-open/evaluation.md)
- [Dynamic Evaluation Protocol](docs-open/dynamic_evaluation.md)
- [Training and Balance-GRPO](docs-open/training.md)

<details>
<summary><b>Repository structure</b></summary>

```text
StepGuard/
├── src/                    # Guard inference, prompts, evaluators, and agent loop
├── configs/                # Static and dynamic evaluation templates
├── scripts/eval/           # Evaluation entry points
├── benchmarks/             # Local benchmark mount points
├── benchmark-repos/        # Third-party benchmark source dependencies
├── training/release/       # SFT, GRPO, and Balance-GRPO assets
├── docs-open/              # Usage and reproduction documentation
└── tests/                  # Unit and configuration smoke tests
```

</details>

## 📑 Citation

Citation metadata will be added with the public paper release. Until then,
please cite the accompanying StepGuard paper and link to this repository.

## 🤝 Acknowledgements

StepGuard uses [Qwen](https://github.com/QwenLM/Qwen),
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and
[SLIME](https://github.com/THUDM/slime) as external training dependencies.
The evaluation stack incorporates third-party benchmark resources documented
in [benchmark-repos/UPSTREAM.md](benchmark-repos/UPSTREAM.md). Please follow
the corresponding upstream licenses and terms when using those assets.
