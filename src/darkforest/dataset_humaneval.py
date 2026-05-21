from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class HumanEvalSample:
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str

    def to_problem(self) -> dict:
        return asdict(self)


def _load_json_or_jsonl(path: str | Path) -> List[dict]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"HumanEval data file not found: {resolved}")
    if resolved.suffix == ".jsonl":
        rows = []
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or JSONL file for HumanEval: {resolved}")
    return data


def load_humaneval_samples(path: str | Path) -> List[HumanEvalSample]:
    samples = []
    for row in _load_json_or_jsonl(path):
        missing = [key for key in ["task_id", "prompt", "canonical_solution", "test", "entry_point"] if key not in row]
        if missing:
            raise ValueError(f"HumanEval sample missing fields {missing}: {row.get('task_id')}")
        samples.append(
            HumanEvalSample(
                task_id=str(row["task_id"]),
                prompt=str(row["prompt"]),
                canonical_solution=str(row["canonical_solution"]),
                test=str(row["test"]),
                entry_point=str(row["entry_point"]),
            )
        )
    return samples


def split_calibration_and_eval(
    full_data_path: str | Path,
    eval_subset_path: str | Path,
) -> Tuple[List[HumanEvalSample], List[HumanEvalSample]]:
    full = load_humaneval_samples(full_data_path)
    eval_subset = load_humaneval_samples(eval_subset_path)
    eval_ids = {sample.task_id for sample in eval_subset}
    full_by_id = {sample.task_id: sample for sample in full}
    missing = sorted(eval_ids - set(full_by_id))
    if missing:
        raise ValueError(f"Eval subset contains task_ids not in full HumanEval data: {missing[:5]}")
    calibration = [sample for sample in full if sample.task_id not in eval_ids]
    eval_ordered = [full_by_id[sample.task_id] for sample in eval_subset]
    return calibration, eval_ordered


def write_humaneval_problem_file(samples: Iterable[HumanEvalSample], path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_problem(), ensure_ascii=False) + "\n")
    return resolved


def default_humaneval_paths(root_dir: str | Path) -> tuple[Path, Path]:
    root = Path(root_dir)
    return (
        root / "data/HumanEval/test.jsonl",
        root / "data/HumanEval/eval_subset.json",
    )
