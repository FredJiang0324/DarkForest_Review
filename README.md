<div align="center">

# DarkForest Review Artifact

### DarkForest: Less Talk, Higher Accuracy for Multi-Agent LLMs

<p align="center">
  <a href="https://arxiv.org/abs/2605.25188"><img src="https://img.shields.io/badge/arXiv-2605.25188-b31b1b?style=flat&labelColor=555&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2605.25188"><img src="https://img.shields.io/badge/🤗-HuggingFace-FFD21E?style=flat&labelColor=555" alt="Hugging Face"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-3776AB?style=flat&labelColor=555&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat&labelColor=555" alt="PRs Welcome">
</p>

</div>

This branch contains the DarkForest implementation and the calibration/test data needed to inspect or rerun the paper's experiments.

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

Generated files are written under `outputs/` by default.

## Citation

If you find DarkForest useful in your research, please consider citing:

```bibtex
@misc{li2026darkforesttalkhigheraccuracy,
      title={DarkForest: Less Talk, Higher Accuracy for Multi-Agent LLMs},
      author={Yi Li and Songtao Wei and Dongming Jiang and Zhichun Guo and Qiannan Li and Bingzhe Li},
      year={2026},
      eprint={2605.25188},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.25188},
}
```
