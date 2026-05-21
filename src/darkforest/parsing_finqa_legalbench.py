from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


PROGRAM_LINE_RE = re.compile(r"(?im)^\s*(?:Predicted\s+Program|Program)\s*:\s*(.+?)\s*$")
ANSWER_LINE_RE = re.compile(r"(?im)^\s*(?:Final\s+Answer|Answer)\s*:\s*(.+?)\s*$")
ALL_OPS_RE = re.compile(
    r"\b(add|subtract|multiply|divide|exp|greater|table_max|table_min|table_sum|table_average)\s*\(",
    re.IGNORECASE,
)


@dataclass
class ParsedDomainOutput:
    agent_key: str
    raw_response: str
    parsed_answer: Optional[str]
    normalized_answer: Optional[str]
    parsed_program: Optional[str]
    cluster_key: Optional[str]
    invalid_parse: bool
    parse_method: str
    error: Optional[str]
    latency_sec: float
    confidence: Optional[float] = None
    reasoning_excerpt: Optional[str] = None
    finqa_predicted_execution: Any = None
    finqa_program_valid: Optional[bool] = None
    finqa_execution_valid: Optional[bool] = None
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text).lower().strip()
    if text in {"true", "false"}:
        return "yes" if text == "true" else "no"
    text = text.rstrip(".!?").strip()
    return text


def recover_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    start = stripped.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(stripped)):
            if stripped[idx] == "{":
                depth += 1
            elif stripped[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(stripped[start : idx + 1])
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        break
        start = stripped.find("{", start + 1)
    return None


def _last_line_match(pattern: re.Pattern[str], text: str) -> Optional[str]:
    matches = pattern.findall(text or "")
    if not matches:
        return None
    return str(matches[-1]).strip().strip("`").strip()


def _matching_paren(text: str, open_index: int) -> Optional[int]:
    depth = 0
    for idx in range(open_index, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _split_top_level_args(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return parts


def _extract_first_call(text: str) -> Optional[tuple[str, int, int]]:
    match = ALL_OPS_RE.search(text)
    if not match:
        return None
    open_idx = text.find("(", match.start())
    close_idx = _matching_paren(text, open_idx)
    if close_idx is None:
        return None
    return text[match.start() : close_idx + 1].strip(), match.start(), close_idx + 1


def _normalize_call(call: str) -> str:
    call = re.sub(r"\s+", " ", call.strip())
    return re.sub(r"\s*,\s*", ", ", call)


def repair_finqa_program(program_text: Optional[str]) -> Optional[str]:
    if not program_text:
        return None
    text = program_text.strip().strip("`").strip()
    text = re.sub(r"(?i)^Program\s*:\s*", "", text).strip()

    assignment_calls: List[str] = []
    for line in text.splitlines():
        line = line.strip().strip(";")
        match = re.match(r"^#?\d+\s*=\s*(.+)$", line)
        if not match:
            continue
        call = _extract_first_call(match.group(1))
        if call:
            assignment_calls.append(_normalize_call(call[0]))
    if assignment_calls:
        return ", ".join(assignment_calls)

    first = _extract_first_call(text)
    if first is not None:
        first_call, _, first_end = first
        tail = text[first_end:].strip()
        div_match = re.match(r"^/\s*([^*+\-/,\n]+)(?:\s*\*\s*100(?:\.0)?)?\s*$", tail)
        if div_match:
            denominator = div_match.group(1).strip()
            return f"{_normalize_call(first_call)}, divide(#0, {denominator})"

        # Common model form: divide(subtract(a, b), b) * 100. FinQA official
        # execution for percentages expects the decimal ratio; the final answer
        # may show the human-readable percent.
        op_match = re.match(r"^divide\s*\((.*)\)$", first_call.strip(), flags=re.IGNORECASE | re.DOTALL)
        if op_match and re.match(r"^\s*\*\s*100(?:\.0)?\s*$", tail):
            args = _split_top_level_args(op_match.group(1))
            nested = _extract_first_call(args[0]) if len(args) == 2 else None
            if len(args) == 2 and nested and nested[0].strip() == args[0].strip():
                return f"{_normalize_call(args[0])}, divide(#0, {args[1].strip()})"

    return _normalize_call(text)


def _extract_program_block(text: str) -> Optional[str]:
    match = re.search(r"(?im)^\s*(?:Predicted\s+Program|Program)\s*:\s*", text or "")
    if not match:
        return None
    rest = text[match.end() :]
    answer_match = re.search(r"(?im)^\s*(?:Final\s+Answer|Answer)\s*:", rest)
    if answer_match:
        rest = rest[: answer_match.start()]
    lines = []
    for line in rest.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        lines.append(stripped)
    return "\n".join(lines).strip() or None


def extract_finqa_program(text: str) -> Optional[str]:
    block = _extract_program_block(text)
    if block:
        return repair_finqa_program(block)
    labeled = _last_line_match(PROGRAM_LINE_RE, text)
    if labeled:
        labeled = re.split(r"(?i)\bFinal\s+Answer\s*:", labeled)[0].strip()
        return repair_finqa_program(labeled)
    obj = recover_json_object(text or "")
    if obj:
        for key in ("program", "predicted_program"):
            if obj.get(key) not in (None, ""):
                return repair_finqa_program(str(obj[key]).strip())
        answer = obj.get("answer")
        if isinstance(answer, str) and ALL_OPS_RE.search(answer):
            return repair_finqa_program(answer.strip())
    return None


def extract_final_answer(text: str) -> Optional[str]:
    labeled = _last_line_match(ANSWER_LINE_RE, text)
    if labeled:
        return labeled
    obj = recover_json_object(text or "")
    if obj:
        for key in ("final_answer", "answer", "preferred_answer"):
            if obj.get(key) not in (None, ""):
                return str(obj[key]).strip()
    stripped = (text or "").strip()
    if stripped:
        last = stripped.splitlines()[-1].strip()
        if len(last) <= 80:
            return last
    return None


def extract_legal_answer(text: str, answer_choices: Sequence[str]) -> tuple[Optional[str], str]:
    def coerce_allowed(candidate: Any) -> Optional[str]:
        if candidate is None:
            return None
        candidate_text = str(candidate).strip()
        if not candidate_text:
            return None
        normalized_choices = {normalize_label(choice): str(choice) for choice in answer_choices}
        normalized_candidate = normalize_label(candidate_text)
        if normalized_candidate in normalized_choices:
            return normalized_choices[normalized_candidate]
        if not normalized_choices:
            return candidate_text

        # Common exact-match outputs include "Option C", "C.", "c: text",
        # or "yes, because ...". Coerce only from the short final-answer
        # candidate, not from the full reasoning body.
        leading = re.match(r"(?is)^\s*(?:option|choice)?\s*([a-z0-9]+)\s*(?:[\).:\-]|$)", candidate_text)
        if leading:
            token = normalize_label(leading.group(1))
            if token in normalized_choices:
                return normalized_choices[token]

        for choice_norm, original in sorted(normalized_choices.items(), key=lambda item: -len(item[0])):
            if re.match(rf"(?is)^\s*{re.escape(choice_norm)}\b", normalized_candidate):
                return original
        return None

    labeled = _last_line_match(ANSWER_LINE_RE, text)
    if labeled:
        return coerce_allowed(labeled) or labeled, "final_answer_line"
    obj = recover_json_object(text or "")
    if obj:
        for key in ("answer", "final_answer", "preferred_answer"):
            if obj.get(key) not in (None, ""):
                raw = str(obj[key]).strip()
                return coerce_allowed(raw) or raw, f"json_{key}"
    stripped = (text or "").strip()
    normalized_choices = {normalize_label(choice): str(choice) for choice in answer_choices}
    if stripped and normalize_label(stripped) in normalized_choices:
        return normalized_choices[normalize_label(stripped)], "entire_response_choice"
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        coerced_last = coerce_allowed(lines[-1])
        if coerced_last is not None:
            return coerced_last, "last_line_choice"
    if answer_choices:
        allowed = "|".join(re.escape(normalize_label(choice)) for choice in answer_choices)
        matches = re.findall(rf"\b({allowed})\b", stripped, flags=re.IGNORECASE)
        if matches:
            norm = normalize_label(matches[-1])
            return normalized_choices.get(norm, matches[-1]), "last_allowed_token"
    return None, "no_legal_answer"


def reasoning_excerpt_before_final(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    cut = len(text)
    for pattern in (PROGRAM_LINE_RE, ANSWER_LINE_RE):
        matches = list(pattern.finditer(text))
        if matches:
            cut = min(cut, matches[-1].start())
    excerpt = re.sub(r"\s+", " ", text[:cut].strip())
    return excerpt or None


def parse_domain_agent_output(
    agent_key: str,
    raw_response: str,
    benchmark: str,
    answer_choices: Sequence[str] = (),
    latency_sec: float = 0.0,
    usage: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> ParsedDomainOutput:
    benchmark = benchmark.lower()
    if benchmark == "finqa":
        program = extract_finqa_program(raw_response)
        answer = extract_final_answer(raw_response)
        invalid = not bool(program or answer)
        method = "program_and_answer" if program and answer else "program_only" if program else "answer_only" if answer else "no_parse"
        normalized = normalize_label(answer) if answer is not None else None
        return ParsedDomainOutput(
            agent_key=agent_key,
            raw_response=raw_response,
            parsed_answer=answer,
            normalized_answer=normalized,
            parsed_program=program,
            cluster_key=None,
            invalid_parse=invalid,
            parse_method=method,
            error=error or ("no FinQA program or answer parsed" if invalid else None),
            latency_sec=float(latency_sec or 0.0),
            reasoning_excerpt=reasoning_excerpt_before_final(raw_response),
            usage=usage or {},
        )
    if benchmark == "legalbench":
        answer, method = extract_legal_answer(raw_response, answer_choices)
        normalized = normalize_label(answer)
        normalized_choices = {normalize_label(choice) for choice in answer_choices}
        invalid = not bool(answer) or (bool(normalized_choices) and normalized not in normalized_choices)
        return ParsedDomainOutput(
            agent_key=agent_key,
            raw_response=raw_response,
            parsed_answer=answer,
            normalized_answer=normalized if normalized else None,
            parsed_program=None,
            cluster_key=normalized if normalized and not invalid else None,
            invalid_parse=invalid,
            parse_method=method,
            error=error or ("no valid LegalBench answer parsed" if invalid else None),
            latency_sec=float(latency_sec or 0.0),
            reasoning_excerpt=reasoning_excerpt_before_final(raw_response),
            usage=usage or {},
        )
    raise ValueError(f"Unsupported benchmark: {benchmark}")
