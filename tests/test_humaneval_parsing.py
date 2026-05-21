import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.parsing_humaneval import extract_code_completion  # noqa: E402


PROMPT = 'def add_one(x):\n    """Return x plus one."""\n'


def test_extracts_fenced_completion_body():
    raw = "```python\n    return x + 1\n```"
    parsed = extract_code_completion(raw, prompt=PROMPT, entry_point="add_one")
    assert parsed["completion"] == "    return x + 1"
    assert parsed["invalid_parse"] is False


def test_extracts_body_from_full_function():
    raw = 'def add_one(x):\n    """Return x plus one."""\n    y = x + 1\n    return y\n'
    parsed = extract_code_completion(raw, prompt=PROMPT, entry_point="add_one")
    assert parsed["completion"] == "    y = x + 1\n    return y"
    assert parsed["invalid_parse"] is False


def test_unindented_return_is_made_appendable():
    parsed = extract_code_completion("return x + 1", prompt=PROMPT, entry_point="add_one")
    assert parsed["completion"] == "    return x + 1"
    assert parsed["invalid_parse"] is False


def test_json_code_field_parses():
    parsed = extract_code_completion('{"code": "return x + 1"}', prompt=PROMPT, entry_point="add_one")
    assert parsed["completion"] == "    return x + 1"
    assert parsed["parse_method"].startswith("json_code")


def test_empty_output_is_invalid():
    parsed = extract_code_completion("", prompt=PROMPT, entry_point="add_one")
    assert parsed["invalid_parse"] is True
    assert parsed["completion"] is None


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
