from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from .schemas import ParsedAgentOutput


def _clip_float(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def extract_last_boxed(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    token = "\\boxed"
    found = []
    start = 0
    while True:
        idx = text.find(token, start)
        if idx < 0:
            break
        pos = idx + len(token)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            start = idx + len(token)
            continue
        depth = 0
        content_start = pos + 1
        end = pos
        while end < len(text):
            char = text[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[content_start:end])
                    break
            end += 1
        start = max(end + 1, idx + len(token))
    return found[-1] if found else None


def remove_boxed(value: Optional[str]) -> str:
    if value is None:
        return ""
    stripped = value.strip()
    boxed = extract_last_boxed(stripped)
    if boxed is not None and stripped.startswith("\\boxed"):
        return boxed.strip()
    return stripped


def normalize_math_answer(value: Optional[str]) -> str:
    if value is None:
        return ""
    answer = remove_boxed(str(value))
    answer = answer.strip()
    answer = answer.replace("−", "-").replace("π", "\\pi")
    answer = answer.replace("\\$", "").replace("$", "")
    while len(answer) >= 2 and answer[0] == "$" and answer[-1] == "$":
        answer = answer[1:-1].strip()
    if answer.startswith(r"\(") and answer.endswith(r"\)"):
        answer = answer[2:-2].strip()
    if answer.startswith(r"\[") and answer.endswith(r"\]"):
        answer = answer[2:-2].strip()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "\\;": "",
        "\\:": "",
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
    }
    for old, new in replacements.items():
        answer = answer.replace(old, new)
    answer = re.sub(r"\\text\s*\{\s*([^{}]*?)\s*\}", r"\1", answer)
    answer = answer.strip().rstrip(".,;:")
    answer = re.sub(r"\s+", "", answer)
    answer = re.sub(r"(?<=\d)[A-Za-z]+$", "", answer)
    if answer.startswith("\\pm") and len(answer) > len("\\pm"):
        expr = answer[len("\\pm") :]
        answer = f"{expr},-{expr}"
    if "," in answer:
        parts = [part for part in answer.split(",") if part]
        if len(parts) == 2 and parts[0].startswith("-") and parts[0][1:] == parts[1]:
            answer = f"{parts[1]},{parts[0]}"
    return answer


def math_exact_match(pred: Optional[str], gold: Optional[str]) -> bool:
    pred_norm = normalize_math_answer(pred)
    gold_norm = normalize_math_answer(gold)
    return bool(pred_norm) and bool(gold_norm) and pred_norm == gold_norm


def math_verify_match(pred: Optional[str], gold: Optional[str]) -> bool:
    if math_exact_match(pred, gold):
        return True
    pred_norm = normalize_math_answer(pred)
    gold_norm = normalize_math_answer(gold)
    if not pred_norm or not gold_norm:
        return False
    try:
        from math_verify import parse, verify  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "answer_match_backend=math_verify requires the optional math_verify package"
        ) from exc
    try:
        return bool(
            verify(parse(f"${gold_norm}$"), parse(f"${pred_norm}$"))
            or verify(parse(gold_norm), parse(pred_norm))
            or verify(parse(gold_norm), parse(pred_norm.replace(r"\(", "").replace(r"\)", "")))
        )
    except Exception:
        return False


def math_answers_match(pred: Optional[str], gold: Optional[str], backend: str = "exact") -> bool:
    if backend == "exact":
        return math_exact_match(pred, gold)
    if backend == "math_verify":
        return math_verify_match(pred, gold)
    raise ValueError(f"Unknown answer match backend: {backend}")


def parse_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip().rstrip("%")
            if value == "":
                return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if "%" in str(value):
        parsed = parsed / 100.0
    return _clip_float(parsed)


def _json_dict_candidates(text: str) -> Iterable[Tuple[Dict[str, Any], str]]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj, "json_substring"

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        try:
            obj = json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            return
        if isinstance(obj, dict):
            yield obj, "json_recovered_bounds"


def _extract_from_json(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    reasoning = obj.get("reasoning")
    answer = obj.get("answer")
    confidence = obj.get("confidence_level")
    if confidence is None:
        confidence = obj.get("confidence")
    if answer is not None:
        answer = str(answer).strip()
    if reasoning is not None:
        reasoning = str(reasoning).strip()
    return reasoning or None, answer or None, parse_confidence(confidence)


def _unescape_jsonish_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace(r"\"", '"').replace(r"\\", "\\").strip()


def _extract_jsonish_string_field(text: str, field: str) -> Optional[str]:
    pattern = re.compile(
        rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"',
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return _unescape_jsonish_string(matches[-1].group(1)).strip()


def _extract_jsonish_confidence(text: str) -> Optional[float]:
    raw = _extract_jsonish_string_field(text, "confidence_level")
    if raw is None:
        raw = _extract_jsonish_string_field(text, "confidence")
    if raw is not None:
        parsed = parse_confidence(raw)
        if parsed is not None:
            return parsed
    match = re.search(
        r'"(?:confidence_level|confidence)"\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',
        text,
        re.IGNORECASE,
    )
    if match:
        return parse_confidence(match.group(1))
    return None


def _extract_jsonish_fields(text: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    reasoning = _extract_jsonish_string_field(text, "reasoning")
    answer = _extract_jsonish_string_field(text, "answer")
    confidence = _extract_jsonish_confidence(text)
    return reasoning or None, answer or None, confidence


def _compact_freeform_reasoning(text: str, parsed_answer: Optional[str]) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None
    lines = [line.rstrip() for line in stripped.splitlines()]
    if parsed_answer:
        final_boxed = extract_last_boxed(stripped)
        if final_boxed is not None and normalize_math_answer(final_boxed) == normalize_math_answer(parsed_answer):
            for idx in range(len(lines) - 1, -1, -1):
                if "\\boxed" in lines[idx]:
                    lines = lines[:idx] + [lines[idx].replace("\\boxed{" + final_boxed + "}", "[final answer omitted]")]
                    break
    reasoning = "\n".join(line for line in lines if line.strip()).strip()
    if not reasoning:
        return None
    max_chars = 12000
    if len(reasoning) > max_chars:
        return reasoning[:max_chars] + "\n[truncated]"
    return reasoning


def _looks_like_math_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    if re.search(r"\d", stripped):
        return True
    if "\\" in stripped or any(ch in stripped for ch in "=+-*/^_()[]{}"):
        return True
    if stripped.lower() in {"yes", "no", "true", "false"}:
        return True
    return False


def _clean_answer_candidate(candidate: str) -> Optional[str]:
    candidate = candidate.strip().strip("\"'")
    boxed_candidate = extract_last_boxed(candidate)
    if boxed_candidate is not None:
        return boxed_candidate.strip()
    candidate = candidate.rstrip(".").strip()
    candidate = re.sub(r"(?i)^(?:therefore|thus|hence|so)[,\s]+", "", candidate).strip()
    candidate = re.sub(r"(?i)^(?:the\s+)?answer\s+is\s+", "", candidate).strip()
    candidate = re.sub(r"(?i)^(?:the\s+)?final\s+answer\s+is\s+", "", candidate).strip()

    simple_assignment = re.search(
        r"(?i)(?:^|[\s,;])(?:we\s+get\s+|we\s+have\s+|therefore\s+)?[a-z](?:_\{?[a-z0-9]+\}?)?\s*=\s*(.+)$",
        candidate,
    )
    if simple_assignment:
        candidate = simple_assignment.group(1).strip()

    return candidate if _looks_like_math_answer(candidate) else None


def _heuristic_answer(text: str) -> Tuple[Optional[str], str]:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip(), "boxed"

    field_patterns = [
        r"(?im)^\s*answer\s*[:=]\s*[\"']?([^\"'\n]+)",
        r"(?im)^\s*final answer\s*[:=]\s*[\"']?([^\"'\n]+)",
        r"(?i)(?:the\s+)?final answer\s+is\s+([^\n]+)",
        r"(?i)(?:therefore|thus|hence|so)[,\s]*(?:the\s+)?answer\s+is\s+([^\n]+)",
        r"(?i)the answer is\s+([^\n]+)",
    ]
    for pattern in field_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = _clean_answer_candidate(match.group(1))
            if candidate is not None:
                return candidate, "field_like"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        final_line = lines[-1].strip().strip("\"'")
        final_line = re.sub(r"(?i)^final answer\s*[:=]\s*", "", final_line).strip()
        final_line = re.sub(r"(?i)^the answer is\s*", "", final_line).strip()
        candidate = _clean_answer_candidate(final_line)
        if candidate is not None:
            return candidate, "final_line"

    return None, "heuristic_failed"


def parse_agent_response(
    agent_key: str,
    raw_response: Optional[str],
    latency_sec: float = 0.0,
    usage: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> ParsedAgentOutput:
    text = raw_response or ""
    parsed_reasoning = None
    parsed_answer = None
    confidence = None
    parse_method = "invalid"
    malformed_json = True

    stripped = text.strip()
    if stripped:
        try:
            strict_obj = json.loads(stripped)
            if isinstance(strict_obj, dict):
                parsed_reasoning, parsed_answer, confidence = _extract_from_json(strict_obj)
                parse_method = "strict_json"
                malformed_json = False
        except json.JSONDecodeError:
            strict_obj = None

        if parsed_answer is None:
            for obj, method in _json_dict_candidates(text):
                parsed_reasoning, parsed_answer, confidence = _extract_from_json(obj)
                if parsed_answer is not None:
                    parse_method = method
                    break

        if parsed_answer is None:
            parsed_reasoning, parsed_answer, confidence = _extract_jsonish_fields(text)
            if parsed_answer is not None:
                parse_method = "jsonish_field"

        if parsed_answer is None:
            parsed_answer, heuristic_method = _heuristic_answer(text)
            if parsed_answer is not None:
                parse_method = heuristic_method

        if parsed_answer is not None and parsed_reasoning is None:
            parsed_reasoning = _compact_freeform_reasoning(text, parsed_answer)

    normalized_answer = normalize_math_answer(parsed_answer) if parsed_answer is not None else None
    invalid_parse = not bool(normalized_answer)
    if invalid_parse:
        parsed_answer = None
        normalized_answer = None
    return ParsedAgentOutput(
        agent_key=agent_key,
        raw_response=text,
        parsed_reasoning=parsed_reasoning,
        parsed_answer=parsed_answer,
        normalized_answer=normalized_answer,
        confidence=confidence,
        malformed_json=malformed_json,
        invalid_parse=invalid_parse,
        parse_method=parse_method,
        error=error,
        latency_sec=latency_sec,
        usage=usage or {},
    )


def extract_final_answer(raw_response: Optional[str]) -> Dict[str, Any]:
    text = raw_response or ""
    answer, method = _heuristic_answer(text)
    normalized = normalize_math_answer(answer) if answer is not None else None
    invalid = not bool(normalized)
    if invalid:
        answer = None
        normalized = None
    return {
        "parsed_answer": answer,
        "normalized_answer": normalized,
        "invalid_parse": invalid,
        "parse_method": method if not invalid else "invalid",
    }
