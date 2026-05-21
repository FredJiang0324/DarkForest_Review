# LegalBench Exact-Match Sample

This directory contains a deterministic, task-balanced subset of downloaded LegalBench tasks whose gold answers are finite labels suitable for exact-match evaluation.

Excluded tasks include open citation prediction, definition extraction, numeric tolerance scoring, SSLA entity extraction, manual `rule_qa`, and the borderline multi-label `successor_liability` task.

## Files

- `legalbench_calibration_100.jsonl`: 100 calibration examples from train splits.
- `legalbench_eval_500.jsonl`: 500 evaluation examples from test splits.
- `metadata.json`: machine-readable selection, counts, labels, and sampling metadata.
- `task_selection_report.md`: human-readable included/excluded task report.

## Preparation

The calibration and evaluation records are already materialized in this review artifact. The original sampling step required the full upstream LegalBench data, which is not included here.

## Evaluation Command

```bash
python data/scripts/eval_legalbench_exact_match.py \
  --data data/LegalBench_Sample/legalbench_eval_500.jsonl \
  --pred outputs/legalbench_predictions.jsonl \
  --output outputs/legalbench_eval_details.jsonl
```

The evaluator normalizes gold and predicted final answers with the same conservative exact-match function used during sampling.
