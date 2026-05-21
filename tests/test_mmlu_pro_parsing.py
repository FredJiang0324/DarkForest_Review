import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.parsing_mmlu_pro import extract_mmlu_pro_answer, parse_mmlu_pro_agent_output  # noqa: E402


def test_extract_answer_is_format():
    parsed = extract_mmlu_pro_answer("After reasoning, the answer is (C).", num_choices=10)
    assert parsed["answer"] == "C"
    assert parsed["invalid_parse"] is False


def test_extract_answer_colon_format():
    parsed = extract_mmlu_pro_answer("Answer: D", num_choices=10)
    assert parsed["answer"] == "D"


def test_extract_last_standalone_choice():
    parsed = extract_mmlu_pro_answer("A is tempting, but C is correct", num_choices=4)
    assert parsed["answer"] == "C"


def test_invalid_empty_output():
    parsed = extract_mmlu_pro_answer("", num_choices=10)
    assert parsed["invalid_parse"] is True
    assert parsed["answer"] is None


def test_reasoning_excerpt_excludes_final_answer_marker():
    parsed = parse_mmlu_pro_agent_output(
        "qwen",
        "A is tempting, but the passage points to C. Therefore the answer is (C).",
        num_choices=4,
    )
    assert parsed.parsed_answer == "C"
    assert parsed.reasoning_excerpt == "A is tempting, but the passage points to C. Therefore"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
