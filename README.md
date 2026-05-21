# DarkForest Review Artifact

This branch contains the DarkForest implementation and the calibration/test data needed to inspect or rerun the paper's ours-only experiments. It intentionally excludes baseline implementations, generated logs, plots, ablation outputs, and machine-local paths.

## Contents

- `src/darkforest/`: DarkForest parsing, belief construction, calibration, coordination, guardrail, and evaluation utilities.
- `scripts/run_*_darkforest.py`: benchmark runners for MATH, HumanEval, MMLU-Pro, GPQA, FinQA, and LegalBench.
- `data/`: calibration and test data used by the DarkForest runners.
- `tests/`: unit tests for parsing, calibration, belief scoring, guardrails, and dataset loaders.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The runners expect OpenAI-compatible vLLM endpoints. Start your own model servers and either use the default local endpoints or pass explicit endpoint/model arguments such as `--qwen_endpoint`, `--coder_endpoint`, `--mathstral_endpoint`, and `--qwen_model_name`.

## Quick Checks

```bash
pytest tests
python scripts/run_math_darkforest.py --mode dry_run --limit_samples 2
python scripts/run_mmlu_pro_darkforest.py --mode dry_run --limit_eval_samples 2
python scripts/run_gpqa_darkforest.py --mode dry_run --limit_eval_samples 2
python scripts/run_humaneval_darkforest.py --mode dry_run --limit_eval_samples 2
python scripts/run_finqa_legalbench_darkforest.py --benchmark finqa --mode dry_run --limit_eval_samples 2
python scripts/run_finqa_legalbench_darkforest.py --benchmark legalbench --mode dry_run --limit_eval_samples 2
```

## Data Notes

This branch includes only the data required by the ours-only runners:

- MATH: `data/MATH/train.jsonl`, `data/MATH/test.jsonl`
- HumanEval: `data/HumanEval/test.jsonl`, `data/HumanEval/eval_subset.json`
- MMLU-Pro: `data/MMLU-Pro/validation.jsonl`, `data/MMLU-Pro/test.jsonl`, `data/MMLU-Pro/sampled_test.json`
- GPQA: `data/GPQA/dev.json`, `data/GPQA/test.json`
- FinQA: `data/FinQA_Sample/` plus the official evaluator file under `data/FinQA/code/evaluate/evaluate.py`
- LegalBench: `data/LegalBench_Sample/`

Generated files are written under `outputs/` by default and are ignored by git.
