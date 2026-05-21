from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ParsedHumanEvalOutput:
    agent_key: str
    raw_response: str
    parsed_completion: Optional[str]
    normalized_completion: Optional[str]
    invalid_parse: bool
    parse_method: str
    error: Optional[str]
    latency_sec: float
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)


def normalize_code_completion(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    lines = [line.rstrip() for line in code.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    return "\n".join(lines)


def _try_json_code(text: str) -> tuple[Optional[str], Optional[str]]:
    keys = ("completion", "code", "answer", "solution")
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            for key in keys:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value, f"json_{key}"
    return None, None


def _extract_fenced_code(text: str) -> tuple[Optional[str], Optional[str]]:
    match = _FENCE_RE.search(text)
    if match:
        return match.group("code"), "fenced_code"
    return None, None


def _remove_prompt_prefix(text: str, prompt: Optional[str]) -> str:
    if not prompt:
        return text
    if prompt in text:
        return text.split(prompt)[-1]
    return text


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _extract_function_body(text: str, entry_point: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not entry_point:
        return None, None
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    pattern = re.compile(rf"^(?P<indent>\s*)def\s+{re.escape(entry_point)}\s*\(")
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        def_indent = len(match.group("indent").replace("\t", "    "))
        colon = line.find(":")
        if colon >= 0 and line[colon + 1 :].strip():
            return "    " + line[colon + 1 :].strip(), "full_function_one_line"
        body: list[str] = []
        for next_line in lines[idx + 1 :]:
            if next_line.strip() and _line_indent(next_line.expandtabs(4)) <= def_indent:
                break
            body.append(next_line)
        normalized = normalize_code_completion("\n".join(body))
        if normalized:
            return normalized, "full_function_body"
    return None, None


def _drop_non_code_preamble(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    code_start = 0
    code_line = re.compile(
        r"^\s*(return\b|if\b|elif\b|else:|for\b|while\b|try:|except\b|finally:|with\b|"
        r"def\b|class\b|import\b|from\b|raise\b|assert\b|pass\b|break\b|continue\b|"
        r"[A-Za-z_][A-Za-z0-9_]*\s*=|[A-Za-z_][A-Za-z0-9_]*\()"
    )
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if code_line.match(line):
            code_start = idx
            break
    else:
        return text

    kept: list[str] = []
    for line in lines[code_start:]:
        stripped = line.strip()
        lowered = stripped.lower()
        if stripped == "```":
            break
        if lowered.startswith(("explanation:", "note:", "here is", "this code", "the code")):
            break
        if re.match(r"^(def\s+check\s*\(|if\s+__name__\s*==|assert\s+)", stripped):
            break
        kept.append(line)
    return "\n".join(kept)


def _indent_completion_if_needed(code: str) -> str:
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    first = next((line for line in lines if line.strip()), "")
    if not first:
        return code
    if first.startswith((" ", "\t")):
        return code
    return "\n".join(("    " + line if line.strip() else line) for line in lines)


def _compile_error(prompt: str, completion: str) -> Optional[str]:
    try:
        compile(prompt + completion, "<humaneval_completion>", "exec")
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def extract_code_completion(
    raw_response: str,
    prompt: Optional[str] = None,
    entry_point: Optional[str] = None,
) -> Dict[str, Any]:
    text = (raw_response or "").strip()
    if not text:
        return {
            "completion": None,
            "normalized_completion": None,
            "invalid_parse": True,
            "parse_method": "empty",
            "error": "empty response",
        }

    method = "heuristic"
    candidate, json_method = _try_json_code(text)
    if candidate is not None:
        text = candidate
        method = json_method or "json"
    else:
        fenced, fenced_method = _extract_fenced_code(text)
        if fenced is not None:
            text = fenced
            method = fenced_method or "fenced_code"

    text = _remove_prompt_prefix(text, prompt)
    body, body_method = _extract_function_body(text, entry_point)
    if body is not None:
        text = body
        method = body_method or method
    else:
        text = _drop_non_code_preamble(text)

    text = text.replace("```python", "").replace("```", "")
    completion = normalize_code_completion(text)
    if not completion:
        return {
            "completion": None,
            "normalized_completion": None,
            "invalid_parse": True,
            "parse_method": method,
            "error": "no code completion extracted",
        }

    completion = _indent_completion_if_needed(completion)
    normalized = normalize_code_completion(completion)
    compile_error = _compile_error(prompt or "", completion) if prompt else None
    return {
        "completion": completion,
        "normalized_completion": normalized,
        "invalid_parse": compile_error is not None,
        "parse_method": method + ("+compile_error" if compile_error else ""),
        "error": compile_error,
    }


def parse_humaneval_agent_output(
    agent_key: str,
    raw_response: str,
    prompt: str,
    entry_point: str,
    latency_sec: float = 0.0,
    usage: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> ParsedHumanEvalOutput:
    parsed = extract_code_completion(raw_response, prompt=prompt, entry_point=entry_point)
    combined_error = error or parsed.get("error")
    return ParsedHumanEvalOutput(
        agent_key=agent_key,
        raw_response=raw_response,
        parsed_completion=parsed.get("completion"),
        normalized_completion=parsed.get("normalized_completion"),
        invalid_parse=bool(parsed.get("invalid_parse")),
        parse_method=str(parsed.get("parse_method") or "unknown"),
        error=combined_error,
        latency_sec=float(latency_sec or 0.0),
        usage=usage or {},
    )
