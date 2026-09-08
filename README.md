# Efficient Post-Training of Qwen2.5-1.5B

**SFT → DPO → RLVR/GRPO → Unified Evaluation**

A reproducible, resource-efficient post-training study for `Qwen/Qwen2.5-1.5B-Instruct`. The repository deliberately separates data curation, training, verification, and evaluation so every reported number is traceable to a configuration and result file.

## Status and results

Metrics are intentionally blank until experiments run. Use `results/<run>/metrics.json` as the source of truth.

| Model | GSM8K | Code | Preference win rate |
|---|---:|---:|---:|
| Base | pending | pending | 50.0% reference |
| SFT | pending | pending | pending |
| DPO | pending | pending | pending |
| RLVR | pending | pending | pending |

## Quick start

Python 3.10+ and a CUDA GPU are recommended for training. Install a PyTorch build appropriate for your CUDA version before the remaining dependencies.

```bash
pip install -r requirements.txt
python -m src.data.prepare_sft --input data/raw/sft.jsonl --output data/processed/sft.jsonl
python -m src.data.prepare_dpo --input data/raw/preference.jsonl --output data/preference/dpo.jsonl
python -m src.sft.train_sft --config configs/sft.yaml
python -m src.dpo.train_dpo --config configs/dpo.yaml
python -m src.evaluation.run_eval --model checkpoints/dpo --tasks gsm8k code preference
```

For GRPO, first turn verified math examples into prompts:

```bash
python -m src.data.prepare_rl --input data/raw/math.jsonl --output data/rl/math.jsonl
python -m src.rl.train_grpo --config configs/grpo.yaml
```

## Data contracts

SFT JSONL accepts either `{instruction, input?, output}` or `{messages: [...]}`. DPO JSONL requires `{prompt, chosen, rejected}`. RL JSONL requires `{prompt, answer}` (optional `id`). All preprocessing outputs normalized JSONL and a sidecar statistics file.

## Experiment discipline

Run identifiers are generated from the config or passed with `--run-name`. Preserve the YAML used for every run. The evaluation harness writes accuracy, response-length and latency statistics; preference evaluation uses an explicit labels file, avoiding model-as-judge claims.

Code execution is disabled by default in evaluation. The optional verifier runs candidate code in a subprocess with a timeout, but it is not a hardened sandbox: only run trusted, local benchmark candidates.

## Layout

- `src/data`: normalization, filtering and splits
- `src/{sft,dpo,rl}`: QLoRA-based training entry points
- `src/evaluation`: GSM8K-style, unit-test and labeled-pair evaluation
- `configs`: versionable experiment settings
- `results`: generated, per-run evidence

See the project plan for the full research matrix: beta, rank, data-scale and preference-quality ablations should each change one config field at a time.
