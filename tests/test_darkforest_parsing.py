import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.parsing import (  # noqa: E402
    extract_last_boxed,
    math_exact_match,
    normalize_math_answer,
    parse_agent_response,
)


def test_strict_json_parses():
    raw = '{"reasoning": "Use arithmetic.", "answer": "42", "confidence_level": 0.8}'
    parsed = parse_agent_response("qwen", raw)
    assert parsed.parsed_reasoning == "Use arithmetic."
    assert parsed.parsed_answer == "42"
    assert parsed.normalized_answer == "42"
    assert parsed.confidence == 0.8
    assert parsed.malformed_json is False
    assert parsed.invalid_parse is False
    assert parsed.parse_method == "strict_json"


def test_confidence_string_parses():
    raw = '{"reasoning": "ok", "answer": "3/4", "confidence_level": "0.65"}'
    parsed = parse_agent_response("mathstral_1", raw)
    assert parsed.confidence == 0.65


def test_malformed_json_with_boxed_falls_back():
    raw = 'Reasoning text. The answer is \\boxed{\\frac{1}{2}}.'
    parsed = parse_agent_response("qwen", raw)
    assert parsed.parsed_answer == "\\frac{1}{2}"
    assert parsed.normalized_answer == "\\frac{1}{2}"
    assert parsed.malformed_json is True
    assert parsed.invalid_parse is False


def test_missing_confidence_is_none():
    raw = '{"reasoning": "ok", "answer": "5"}'
    parsed = parse_agent_response("mathstral_2", raw)
    assert parsed.confidence is None
    assert parsed.invalid_parse is False


def test_invalid_output_sets_invalid_parse():
    parsed = parse_agent_response("qwen", "I have no idea.")
    assert parsed.invalid_parse is True
    assert parsed.parsed_answer is None


def test_malformed_json_with_latex_reasoning_recovers_answer_field():
    raw = (
        'Reasoning with invalid JSON escapes \\dfrac{1}{2}. '
        '{"reasoning": "Use \\dfrac{1}{2}", "answer": "3", "confidence_level": 0.95}'
    )
    parsed = parse_agent_response("qwen", raw)
    assert parsed.parsed_answer == "3"
    assert parsed.normalized_answer == "3"
    assert parsed.confidence == 0.95
    assert parsed.malformed_json is True
    assert parsed.invalid_parse is False
    assert parsed.parse_method == "jsonish_field"


def test_freeform_cot_boxed_answer_fills_structured_fields():
    raw = "We solve the equation carefully.\nThe only possible value is 7.\nThe answer is \\boxed{7}"
    parsed = parse_agent_response("qwen", raw)
    assert parsed.parsed_answer == "7"
    assert parsed.normalized_answer == "7"
    assert parsed.parsed_reasoning is not None
    assert "solve the equation" in parsed.parsed_reasoning
    assert parsed.confidence is None
    assert parsed.malformed_json is True
    assert parsed.invalid_parse is False
    assert parsed.parse_method == "boxed"


def test_freeform_final_assignment_extracts_rhs():
    raw = "After simplifying, the equation gives x = 3."
    parsed = parse_agent_response("qwen", raw)
    assert parsed.parsed_answer == "3"
    assert parsed.normalized_answer == "3"
    assert parsed.parse_method == "final_line"


def test_extracts_last_boxed():
    text = "First \\boxed{1}, then final \\boxed{2}."
    assert extract_last_boxed(text) == "2"


def test_extracts_nested_boxed_fraction():
    assert extract_last_boxed(r"The answer is \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"


def test_normalization():
    assert normalize_math_answer(" $  42 . $ ") == "42"
    assert normalize_math_answer(r"\dfrac{1}{2}") == r"\frac{1}{2}"
    assert normalize_math_answer(r"\tfrac{1}{2}") == r"\frac{1}{2}"
    assert normalize_math_answer(r"\left( x \right)") == "(x)"
    assert normalize_math_answer(r"\( \frac{1}{2} \)") == r"\frac{1}{2}"
    assert normalize_math_answer(r"\text{D}") == "D"
    assert normalize_math_answer(r"575\text{ students}") == "575"
    assert normalize_math_answer(r"\$0.50") == "0.50"
    assert normalize_math_answer(r"\pm\frac{3}{4}") == r"\frac{3}{4},-\frac{3}{4}"
    assert normalize_math_answer("π") == r"\pi"
    assert math_exact_match(r" \boxed{\dfrac{1}{2}} ", r"\frac{1}{2}")


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
