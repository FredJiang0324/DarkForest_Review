# FinQA Text-Only Sample

This directory contains a strict text-only FinQA sample with 300 examples.

## Table Evidence Policy

Table evidence is excluded rather than verbalized. The preparation script does not add table headers, table cells, table rows, or table-derived natural-language renderings to the model-facing `context` or `prompt`. It does not call or reuse FinQA `table_row_to_text` for prompts.

The prompt context is built only from original `pre_text`/`post_text` text blocks for examples whose FinQA supporting-fact annotations and gold supporting indices are text-only. Gold answers, gold programs, derivations, and label fields are kept outside the prompt.

## Selection

- Seed: `0`
- Target size: `300`
- Valid strict text-only numeric examples by split: `{'test': 255, 'dev': 178, 'train': 1310, 'private_test': 0}`
- Selected examples by split: `{'test': 255, 'dev': 45}`
- Selection priority: public `test` split first, then `dev`, then `train` if needed.

Strict candidates must have a gold program, a numeric gold execution answer, text-only `qa.gold_inds`, non-empty `qa.ann_text_rows`, empty `qa.ann_table_rows`, no table operation in the gold program, and at least one usable original text block.

## Files

- `finqa_text_only_sample.jsonl`: sampled JSONL dataset.
- `finqa_text_only_calibration.jsonl`: non-overlapping strict text-only calibration set.
- `metadata.json`: generation metadata and diagnostics.
- `calibration_metadata.json`: calibration set generation metadata and overlap checks.
- `README.md`: this file.

Each JSONL record contains `id`, source location, `question`, text-only `context`, model-facing `prompt`, `gold_answer`, `gold_execution_answer`, `gold_program`, `raw_id`, and audit metadata.

## Calibration Set

`finqa_text_only_calibration.jsonl` contains 100 additional strict text-only examples for calibration. It was generated with seed `1`, excludes all 300 records from `finqa_text_only_sample.jsonl`, and starts numeric ids at `300` to avoid id overlap. The generated set has zero overlap with the main sample by `raw_id`, numeric `id`, and source file/index.

The calibration records are already materialized in this review artifact. The original sampling script is not required for running DarkForest.

## Model Output Format

The prompt asks models to return final lines in this parsable format:

```text
Program: subtract(100, 25)
Final Answer: 75
```

## Evaluation

The evaluator script reuses the official FinQA evaluator/executor from `data/FinQA/code/evaluate/evaluate.py` when importable. Specifically, it calls the official `eval_program` for Execution Accuracy and `equal_program` for Program Accuracy. If direct import fails, it falls back to minimal source-cited logic copied from that file.

Run the evaluator on a prediction file:

```bash
python data/scripts/eval_finqa_metrics.py \
  --data data/FinQA_Sample/finqa_text_only_sample.jsonl \
  --pred outputs/finqa_predictions.jsonl \
  --output outputs/finqa_eval_details.jsonl
```

Execution Accuracy is computed by executing the predicted FinQA-style program and comparing the result to the gold FinQA execution answer. Program Accuracy uses the official FinQA symbolic program equivalence rule when available. Final-answer comparison is reported only as a diagnostic.
