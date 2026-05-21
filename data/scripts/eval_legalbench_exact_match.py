#!/usr/bin/env python3
"""Evaluate LegalBench exact-match sample predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from legalbench_exact_match_utils import load_jsonl, normalize_answer, write_jsonl


FINAL_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$"
)
FINAL_ANSWER_ANYWHERE_RE = re.compile(
    r"(?is)(?:final\s+answer|answer)\s*:\s*(.+)"
)


def load_prediction_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list or JSONL records in {path}")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError(f"Expected prediction objects in {path}")
        return data
    return load_jsonl(path)


def prediction_value(record: Dict[str, Any]) -> Tuple[Any, str]:
    for key in ("prediction", "response", "answer"):
        if key in record:
            return record.get(key), key
    return None, "missing"


def extract_final_answer(raw: Any) -> Tuple[str, bool]:
    if raw is None:
        return "", True
    text = str(raw).strip()
    if not text:
        return "", True

    match = FINAL_ANSWER_LINE_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip(), False

    match = FINAL_ANSWER_ANYWHERE_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip().splitlines()[0].strip(), False

    return text, False


def safe_accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = load_jsonl(args.data)
    predictions = load_prediction_file(args.pred)

    pred_by_id: Dict[str, Dict[str, Any]] = {}
    for pred in predictions:
        if "id" not in pred:
            raise ValueError(f"Prediction record missing id: {pred}")
        pred_id = str(pred["id"])
        if pred_id in pred_by_id:
            raise ValueError(f"Duplicate prediction id: {pred_id}")
        pred_by_id[pred_id] = pred

    data_by_id = {str(record["id"]): record for record in data}
    missing_predictions = sorted(set(data_by_id) - set(pred_by_id), key=lambda x: int(x) if x.isdigit() else x)
    extra_predictions = sorted(set(pred_by_id) - set(data_by_id), key=lambda x: int(x) if x.isdigit() else x)

    details: List[Dict[str, Any]] = []
    per_task = defaultdict(lambda: {"total": 0, "correct": 0, "invalid_parse": 0})

    for data_id, example in data_by_id.items():
        pred_record = pred_by_id.get(data_id)
        if pred_record is None:
            continue
        raw_prediction, prediction_field = prediction_value(pred_record)
        extracted, invalid_parse = extract_final_answer(raw_prediction)
        normalized_prediction = normalize_answer(extracted)
        normalized_gold = normalize_answer(example.get("normalized_answer", example.get("answer")))
        correct = (not invalid_parse) and normalized_prediction == normalized_gold

        task = example["task"]
        per_task[task]["total"] += 1
        per_task[task]["correct"] += int(correct)
        per_task[task]["invalid_parse"] += int(invalid_parse)

        details.append(
            {
                "id": example["id"],
                "task": task,
                "gold": example.get("answer"),
                "prediction": raw_prediction,
                "prediction_field": prediction_field,
                "extracted_prediction": extracted,
                "normalized_gold": normalized_gold,
                "normalized_prediction": normalized_prediction,
                "correct": bool(correct),
                "invalid_parse": bool(invalid_parse),
            }
        )

    total = len(details)
    correct = sum(1 for item in details if item["correct"])
    invalid_parse = sum(1 for item in details if item["invalid_parse"])
    per_task_summary = {}
    for task in sorted(per_task):
        stats = per_task[task]
        per_task_summary[task] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "exact_match_accuracy": safe_accuracy(stats["correct"], stats["total"]),
            "invalid_parse": stats["invalid_parse"],
            "invalid_parse_rate": safe_accuracy(stats["invalid_parse"], stats["total"]),
        }

    summary = {
        "total": total,
        "correct": correct,
        "exact_match_accuracy": safe_accuracy(correct, total),
        "invalid_parse": invalid_parse,
        "invalid_parse_rate": safe_accuracy(invalid_parse, total),
        "dataset_total": len(data),
        "predictions_total": len(predictions),
        "missing_predictions": len(missing_predictions),
        "extra_predictions": len(extra_predictions),
        "per_task_accuracy": {
            task: stats["exact_match_accuracy"] for task, stats in per_task_summary.items()
        },
        "per_task_counts": {
            task: {"total": stats["total"], "correct": stats["correct"]}
            for task, stats in per_task_summary.items()
        },
        "per_task": per_task_summary,
    }

    if args.output is not None:
        write_jsonl(args.output, details)

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
