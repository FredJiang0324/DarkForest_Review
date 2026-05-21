from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ParsedGPQAOutput:
    agent_key: str
    raw_response: str
    parsed_answer: Optional[str]
    invalid_parse: bool
    parse_method: str
    error: Optional[str]
    latency_sec: float
    reasoning_excerpt: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _choice_pattern(num_choices: int) -> str:
    letters = "".join(chr(65 + idx) for idx in range(num_choices))
    return f"[{letters}]"


def extract_gpqa_answer(text: str, num_choices: int = 10) -> Dict[str, Any]:
    if not text or not text.strip():
        return {
            "answer": None,
            "invalid_parse": True,
            "parse_method": "empty",
            "error": "empty response",
        }
    choice = _choice_pattern(num_choices)
    patterns = [
        (rf"answer\s+is\s+\(?({choice})\)?", "answer_is"),
        (rf"Answer:\s*(?:Let's think step by step\.)?.*?answer\s+is\s+\(?({choice})\)?", "answer_line_cot"),
        (rf"final\s+answer\s*(?:is|:)\s*\(?({choice})\)?", "final_answer"),
        (rf"Answer:\s*\(?({choice})\)?", "answer_colon"),
        (rf"\(({choice})\)", "parenthesized_choice"),
    ]
    for pattern, method in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return {
                "answer": str(matches[-1]).upper(),
                "invalid_parse": False,
                "parse_method": method,
                "error": None,
            }

    standalone = re.findall(rf"\b({choice})\b", text, flags=re.IGNORECASE)
    if standalone:
        return {
            "answer": str(standalone[-1]).upper(),
            "invalid_parse": False,
            "parse_method": "last_standalone_choice",
            "error": None,
        }
    return {
        "answer": None,
        "invalid_parse": True,
        "parse_method": "no_choice_found",
        "error": "no answer choice found",
    }


def extract_gpqa_reasoning_excerpt(text: str, num_choices: int = 10) -> Optional[str]:
    if not text or not text.strip():
        return None
    choice = _choice_pattern(num_choices)
    marker_patterns = [
        rf"(?:the\s+)?answer\s+is\s+\(?{choice}\)?",
        rf"final\s+answer\s*(?:is|:)\s*\(?{choice}\)?",
        rf"Answer:\s*\(?{choice}\)?",
    ]
    cut_at = len(text)
    for pattern in marker_patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            cut_at = min(cut_at, matches[-1].start())
    excerpt = text[:cut_at].strip()
    if not excerpt:
        return None
    excerpt = re.sub(r"\s+", " ", excerpt)
    return excerpt


def parse_gpqa_agent_output(
    agent_key: str,
    raw_response: str,
    num_choices: int,
    latency_sec: float = 0.0,
    usage: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> ParsedGPQAOutput:
    parsed = extract_gpqa_answer(raw_response, num_choices=num_choices)
    return ParsedGPQAOutput(
        agent_key=agent_key,
        raw_response=raw_response,
        parsed_answer=parsed.get("answer"),
        invalid_parse=bool(parsed.get("invalid_parse")),
        parse_method=str(parsed.get("parse_method") or "unknown"),
        error=error or parsed.get("error"),
        latency_sec=float(latency_sec or 0.0),
        reasoning_excerpt=extract_gpqa_reasoning_excerpt(raw_response, num_choices=num_choices),
        usage=usage or {},
    )
