from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CHOICES = list("ABCDEFGHIJKLMNOP")
GPQA_SUBJECT = "graduate-level science"
GPQA_INITIAL_PROMPT = (
    "The following are multiple choice questions about graduate-level science. "
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice."
)


@dataclass
class GPQASample:
    idx: int
    question: str
    options: List[str]
    answer: Optional[str]
    answer_index: Optional[int]
    category: str = "gpqa"
    question_id: Optional[int] = None
    cot_content: str = ""
    src: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _read_rows(path: str | Path) -> List[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list or JSONL rows for GPQA: {path}")
    return data


def parse_goa_question_and_options(question_with_options: str) -> Tuple[str, List[str]]:
    marker = re.search(r"\s+The options are:\s*", question_with_options)
    if marker is None:
        return question_with_options.strip(), []
    question = question_with_options[: marker.start()]
    options_text = question_with_options[marker.end() :]
    matches = []
    cursor = 0
    for letter in CHOICES:
        match = re.search(rf"\({letter}\)\s+", options_text[cursor:])
        if not match:
            break
        start = cursor + match.start()
        end = cursor + match.end()
        matches.append((letter, start, end))
        cursor = end
    options = []
    for idx, (_, _, match_end) in enumerate(matches):
        start = match_end
        end = matches[idx + 1][1] if idx + 1 < len(matches) else len(options_text)
        option = options_text[start:end].strip()
        if idx + 1 == len(matches) and option.endswith("."):
            option = option[:-1].rstrip()
        options.append(option)
    return question.strip(), options


def _normalize_answer(raw: object) -> Optional[str]:
    if raw is None:
        return None
    answer = str(raw).strip().upper()
    if len(answer) == 1 and answer in CHOICES:
        return answer
    match = re.search(r"\(?([A-P])\)?", answer)
    return match.group(1).upper() if match else None


def load_gpqa_json(path: str | Path, split: Optional[str] = None) -> List[GPQASample]:
    rows = _read_rows(path)
    samples: List[GPQASample] = []
    for idx, row in enumerate(rows):
        raw_question = str(row.get("question") or row.get("prompt") or "")
        question, parsed_options = parse_goa_question_and_options(raw_question)
        options = row.get("options") or row.get("choices") or parsed_options
        options = [str(option).strip() for option in options if str(option).strip()]
        answer = _normalize_answer(row.get("gold_answer", row.get("answer")))
        answer_index = CHOICES.index(answer) if answer in CHOICES else None
        category = str(row.get("category") or row.get("subject") or "gpqa")
        samples.append(
            GPQASample(
                idx=idx,
                question=question,
                options=options,
                answer=answer,
                answer_index=answer_index,
                category=category,
                question_id=int(row["question_id"]) if row.get("question_id") is not None else idx,
                cot_content=str(row.get("cot_content") or row.get("explanation") or ""),
                src=split or str(row.get("src") or ""),
                metadata={
                    "source_path": str(path),
                    "source_format": "goa_gpqa_json",
                    "raw_question": raw_question,
                    "split": split,
                },
            )
        )
    return samples


def load_goa_gpqa_split(path: str | Path, split: Optional[str] = None) -> List[GPQASample]:
    return load_gpqa_json(path, split=split)


def group_validation_by_category(samples: Iterable[GPQASample]) -> Dict[str, List[GPQASample]]:
    grouped: Dict[str, List[GPQASample]] = {}
    for sample in samples:
        grouped.setdefault(sample.category, []).append(sample)
    return grouped


def format_gpqa_example(sample: GPQASample, including_answer: bool = True) -> str:
    prompt = "Question:\n"
    prompt += sample.question + "\n"
    prompt += "Options:\n"
    for idx, option in enumerate(sample.options):
        prompt += f"{CHOICES[idx]}. {option}\n"
    if including_answer:
        cot_content = sample.cot_content.strip()
        if not cot_content:
            cot_content = f"Answer: Let's think step by step. The answer is ({sample.answer})."
        else:
            cot_content = cot_content.replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def build_gpqa_cot_prompt(
    sample: GPQASample,
    validation_by_category: Dict[str, List[GPQASample]],
    ntrain: int = 0,
    exclude_question_id: Optional[int] = None,
) -> str:
    prompt = GPQA_INITIAL_PROMPT + "\n\n"
    if ntrain <= 0:
        prompt += format_gpqa_example(sample, including_answer=False)
        return prompt
    examples = validation_by_category.get(sample.category, [])
    selected = []
    for example in examples:
        if exclude_question_id is not None and example.question_id == exclude_question_id:
            continue
        selected.append(example)
        if len(selected) >= max(0, ntrain):
            break
    for example in selected:
        prompt += format_gpqa_example(example, including_answer=True)
    prompt += format_gpqa_example(sample, including_answer=False)
    return prompt


def default_gpqa_paths(root_dir: str | Path) -> tuple[Path, Path]:
    root = Path(root_dir)
    return (
        root / "data/GPQA/dev.json",
        root / "data/GPQA/test.json",
    )
