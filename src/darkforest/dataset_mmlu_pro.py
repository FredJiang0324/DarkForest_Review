from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CHOICES = list("ABCDEFGHIJKLMNOP")
OFFICIAL_INITIAL_PROMPT = (
    'The following are multiple choice questions (with answers) about {$}. '
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice."
)


@dataclass
class MMLUProSample:
    idx: int
    question: str
    options: List[str]
    answer: Optional[str]
    answer_index: Optional[int]
    category: str
    question_id: Optional[int] = None
    cot_content: str = ""
    src: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _read_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_mmlu_pro_jsonl(path: str | Path) -> List[MMLUProSample]:
    samples = []
    for idx, row in enumerate(_read_jsonl(path)):
        options = [str(opt) for opt in row.get("options", []) if str(opt) != "N/A"]
        answer = row.get("answer")
        answer_index = row.get("answer_index")
        if answer is None and answer_index is not None:
            answer = CHOICES[int(answer_index)]
        if answer_index is None and answer in CHOICES:
            answer_index = CHOICES.index(answer)
        samples.append(
            MMLUProSample(
                idx=idx,
                question=str(row["question"]),
                options=options,
                answer=str(answer) if answer is not None else None,
                answer_index=int(answer_index) if answer_index is not None else None,
                category=str(row["category"]),
                question_id=int(row["question_id"]) if row.get("question_id") is not None else None,
                cot_content=str(row.get("cot_content") or ""),
                src=row.get("src"),
                metadata={"source_path": str(path), "source_format": "mmlu_pro_jsonl"},
            )
        )
    return samples


def parse_goa_question_and_options(question_with_options: str) -> Tuple[str, List[str]]:
    marker = " The options are: "
    if marker not in question_with_options:
        raise ValueError("GoA MMLU-Pro question is missing 'The options are:' marker")
    question, options_text = question_with_options.split(marker, 1)
    matches = []
    cursor = 0
    for letter in CHOICES:
        match = re.search(rf"\({letter}\)\s+", options_text[cursor:])
        if not match:
            if letter == "A":
                raise ValueError("Could not parse GoA MMLU-Pro options")
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
        if idx + 1 == len(matches):
            option = option.rstrip()
            if option.endswith("."):
                option = option[:-1].rstrip()
        options.append(option)
    return question.strip(), options


def _normalize_question_key(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def load_goa_mmlu_pro_sampled_test(
    path: str | Path,
    full_test_path: str | Path | None = None,
) -> List[MMLUProSample]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    full_by_question_category: Dict[Tuple[str, str], MMLUProSample] = {}
    if full_test_path is not None:
        for sample in load_mmlu_pro_jsonl(full_test_path):
            full_by_question_category[(_normalize_question_key(sample.question), sample.category)] = sample
    samples = []
    for idx, row in enumerate(rows):
        question, options = parse_goa_question_and_options(str(row["question"]))
        full_match = full_by_question_category.get((_normalize_question_key(question), str(row.get("category", "unknown"))))
        if full_match is not None:
            options = list(full_match.options)
        answer = str(row.get("gold_answer")) if row.get("gold_answer") is not None else None
        samples.append(
            MMLUProSample(
                idx=idx,
                question=question,
                options=options,
                answer=answer,
                answer_index=CHOICES.index(answer) if answer in CHOICES else None,
                category=str(row.get("category", "unknown")),
                question_id=None,
                cot_content="",
                src="goa_sampled_test",
                metadata={
                    "source_path": str(path),
                    "source_format": "goa_mmlu_pro_sampled_test",
                    "raw_question": row["question"],
                },
            )
        )
    return samples


def group_validation_by_category(samples: Iterable[MMLUProSample]) -> Dict[str, List[MMLUProSample]]:
    grouped: Dict[str, List[MMLUProSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.category, []).append(sample)
    return grouped


def format_mmlu_pro_example(sample: MMLUProSample, including_answer: bool = True) -> str:
    prompt = "Question:\n"
    prompt += sample.question + "\n"
    prompt += "Options:\n"
    for idx, option in enumerate(sample.options):
        prompt += f"{CHOICES[idx]}. {option}\n"
    if including_answer:
        cot_content = sample.cot_content or f"A: Let's think step by step. The answer is ({sample.answer})."
        cot_content = cot_content.replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def build_mmlu_pro_cot_prompt(
    sample: MMLUProSample,
    validation_by_category: Dict[str, List[MMLUProSample]],
    ntrain: int = 5,
    exclude_question_id: Optional[int] = None,
) -> str:
    subject = sample.category
    prompt = OFFICIAL_INITIAL_PROMPT.replace("{$}", subject) + "\n\n"
    examples = validation_by_category.get(subject, [])
    selected = []
    for example in examples:
        if exclude_question_id is not None and example.question_id == exclude_question_id:
            continue
        selected.append(example)
        if len(selected) >= ntrain:
            break
    for example in selected:
        prompt += format_mmlu_pro_example(example, including_answer=True)
    prompt += format_mmlu_pro_example(sample, including_answer=False)
    return prompt


def verify_goa_sampled_test_subset(
    full_test_path: str | Path,
    goa_sampled_path: str | Path,
) -> dict:
    full = load_mmlu_pro_jsonl(full_test_path)
    sampled = load_goa_mmlu_pro_sampled_test(goa_sampled_path, full_test_path=full_test_path)
    full_keys = {(sample.question.strip(), sample.answer, sample.category) for sample in full}
    full_question_category = {(sample.question.strip(), sample.category) for sample in full}
    missing = [
        sample.idx
        for sample in sampled
        if (sample.question.strip(), sample.answer, sample.category) not in full_keys
    ]
    missing_question_category = [
        sample.idx
        for sample in sampled
        if (sample.question.strip(), sample.category) not in full_question_category
    ]
    return {
        "full_test_count": len(full),
        "sampled_count": len(sampled),
        "missing_by_question_answer_category": len(missing),
        "missing_indices": missing[:20],
        "missing_by_question_category": len(missing_question_category),
        "missing_question_category_indices": missing_question_category[:20],
    }


def default_mmlu_pro_paths(root_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(root_dir)
    return (
        root / "data/MMLU-Pro/validation.jsonl",
        root / "data/MMLU-Pro/test.jsonl",
        root / "data/MMLU-Pro/sampled_test.json",
    )
