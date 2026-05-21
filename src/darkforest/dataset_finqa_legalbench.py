from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class DomainQASample:
    idx: int
    benchmark: str
    sample_id: Any
    prompt: str
    question: Optional[str]
    category: str
    gold_answer: Any = None
    normalized_gold_answer: Optional[str] = None
    gold_execution_answer: Any = None
    gold_program: Optional[str] = None
    answer_choices: List[str] = field(default_factory=list)
    raw_sample: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(item)
    return rows


def load_domain_qa_samples(path: str | Path, benchmark: str, split: str) -> List[DomainQASample]:
    benchmark = benchmark.lower()
    rows = read_jsonl(path)
    samples: List[DomainQASample] = []
    for idx, row in enumerate(rows):
        if benchmark == "finqa":
            samples.append(
                DomainQASample(
                    idx=idx,
                    benchmark="finqa",
                    sample_id=row.get("id", idx),
                    prompt=str(row.get("prompt") or ""),
                    question=row.get("question"),
                    category="FinQA",
                    gold_answer=row.get("gold_answer"),
                    gold_execution_answer=row.get("gold_execution_answer"),
                    gold_program=row.get("gold_program"),
                    raw_sample=row,
                    metadata={
                        "split": split,
                        "source_path": str(path),
                        "raw_id": row.get("raw_id"),
                        "source_split": row.get("source_split"),
                    },
                )
            )
        elif benchmark == "legalbench":
            samples.append(
                DomainQASample(
                    idx=idx,
                    benchmark="legalbench",
                    sample_id=row.get("id", idx),
                    prompt=str(row.get("prompt") or ""),
                    question=row.get("text"),
                    category=str(row.get("task") or "LegalBench"),
                    gold_answer=row.get("answer"),
                    normalized_gold_answer=row.get("normalized_answer"),
                    answer_choices=[str(choice) for choice in (row.get("answer_choices") or [])],
                    raw_sample=row,
                    metadata={
                        "split": split,
                        "source_path": str(path),
                        "task": row.get("task"),
                        "task_safe_name": row.get("task_safe_name"),
                    },
                )
            )
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
    return samples


def default_domain_paths(root_dir: str | Path, benchmark: str) -> Tuple[Path, Path]:
    root = Path(root_dir)
    if benchmark == "finqa":
        return (
            root / "data/FinQA_Sample/finqa_text_only_calibration.jsonl",
            root / "data/FinQA_Sample/finqa_text_only_sample.jsonl",
        )
    if benchmark == "legalbench":
        return (
            root / "data/LegalBench_Sample/legalbench_calibration_100.jsonl",
            root / "data/LegalBench_Sample/legalbench_eval_500.jsonl",
        )
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def build_domain_initial_prompt(sample: DomainQASample) -> str:
    if sample.benchmark == "finqa":
        return (
            f"{sample.prompt.rstrip()}\n\n"
            "Solve the problem from the provided text evidence only. "
            "The Program must be a comma-separated sequence of FinQA operation calls only. "
            "Do not write infix arithmetic such as op(...) / x or op(...) * 100 outside an operation call. "
            "Reference previous steps as #0, #1, etc. "
            "For percentage-change questions, compute the ratio in the Program and put the human-readable percent in Final Answer if appropriate. "
            "You may reason step by step, but the final two lines must be exactly:\n"
            "Program: <FinQA-style program>\n"
            "Final Answer: <number>"
        )
    if sample.benchmark == "legalbench":
        prompt = strip_trailing_final_answer_marker(sample.prompt.rstrip())
        choices = "\n".join(f"- {choice}" for choice in sample.answer_choices)
        choice_block = f"\n\nAllowed final answers:\n{choices}" if choices and "answer choices:" not in prompt.lower() else ""
        return (
            f"{prompt}{choice_block}\n\n"
            "Choose exactly one allowed final answer. Do not explain.\n"
            "Finish with exactly one line:\n"
            "Final Answer: <answer>"
        )
    raise ValueError(f"Unsupported benchmark: {sample.benchmark}")


def strip_trailing_final_answer_marker(prompt: str) -> str:
    return re.sub(r"\n*\s*Final\s+Answer\s*:\s*$", "", prompt.rstrip(), flags=re.IGNORECASE)


def iter_by_category(samples: Iterable[DomainQASample]) -> Dict[str, List[DomainQASample]]:
    grouped: Dict[str, List[DomainQASample]] = {}
    for sample in samples:
        grouped.setdefault(sample.category, []).append(sample)
    return grouped
