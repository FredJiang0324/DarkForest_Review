import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.dataset_gpqa import (  # noqa: E402
    build_gpqa_cot_prompt,
    load_goa_gpqa_split,
    parse_goa_question_and_options,
)


def test_parse_goa_question_and_options():
    question, options = parse_goa_question_and_options(
        "What is 2+2? The options are: (A) 3 (B) 4 (C) 5."
    )
    assert question == "What is 2+2?"
    assert options == ["3", "4", "5"]


def test_load_gpqa_jsonl_and_build_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "validation.jsonl"
        row = {
            "question_id": 1,
            "question": "What is 2+2?",
            "options": ["3", "4"],
            "answer": "B",
            "answer_index": 1,
            "cot_content": "A: Let's think step by step. 2+2=4. The answer is (B).",
            "category": "math",
            "src": "unit",
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        samples = load_goa_gpqa_split(path, split="dev")
        prompt = build_gpqa_cot_prompt(samples[0], {"math": samples}, ntrain=1)
    assert samples[0].answer == "B"
    assert "The following are multiple choice questions" in prompt
    assert "Answer: Let's think step by step." in prompt
    assert "the answer is (X)" in prompt


def test_load_goa_gpqa_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sampled.json"
        rows = [
            {
                "question": "What is 2+2? The options are: (A) 3 (B) 4.",
                "gold_answer": "B",
                "category": "math",
            }
        ]
        path.write_text(json.dumps(rows), encoding="utf-8")
        samples = load_goa_gpqa_split(path, split="test")
    assert len(samples) == 1
    assert samples[0].question == "What is 2+2?"
    assert samples[0].options == ["3", "4"]
    assert samples[0].answer == "B"


def test_zero_shot_prompt_does_not_include_dev_answer():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dev.json"
        rows = [
            {
                "question": "Calibration question? The options are: (A) no (B) yes.",
                "gold_answer": "B",
            }
        ]
        path.write_text(json.dumps(rows), encoding="utf-8")
        samples = load_goa_gpqa_split(path, split="dev")
        prompt = build_gpqa_cot_prompt(samples[0], {"gpqa": samples}, ntrain=0)
    assert "Calibration question?" in prompt
    assert "The answer is (B)." not in prompt


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
