from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .schemas import FIXED_AGENTS


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("Boolean value cannot be None")
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def normalize_usage(
    usage: Optional[Dict[str, Any]],
    prompt: str,
    completion: str,
) -> Dict[str, Any]:
    usage = usage or {}
    if usage:
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        total_tokens = usage.get("total_tokens")
        if input_tokens is not None or output_tokens is not None or total_tokens is not None:
            input_tokens = int(input_tokens or 0)
            output_tokens = int(output_tokens or 0)
            total_tokens = int(total_tokens if total_tokens is not None else input_tokens + output_tokens)
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "token_count_source": usage.get("token_count_source", "api"),
            }
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(completion)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "token_count_source": "estimated",
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=False)
        handle.write("\n")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def completed_indices(path: Path) -> Set[int]:
    indices = set()
    for row in read_jsonl(path):
        if "idx" in row:
            indices.add(int(row["idx"]))
    return indices


def deterministic_limit(samples: Sequence[Any], limit: Optional[int]) -> List[Any]:
    if limit is None:
        return list(samples)
    return list(samples)[: max(0, int(limit))]


def deterministic_split(samples: Sequence[Any], valid_fraction: float, seed: int) -> tuple[List[Any], List[Any]]:
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    valid_count = int(round(len(indices) * valid_fraction))
    valid_ids = set(indices[:valid_count])
    train = [sample for idx, sample in enumerate(samples) if idx not in valid_ids]
    valid = [sample for idx, sample in enumerate(samples) if idx in valid_ids]
    return train, valid


def parse_agent_priors(value: Optional[str], fixed_agents: Optional[Sequence[str]] = None) -> Dict[str, float]:
    agents = list(fixed_agents or FIXED_AGENTS)
    priors = {agent: 1.0 for agent in agents}
    if not value:
        return priors
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid agent prior item: {part!r}")
        key, raw = part.split("=", 1)
        key = key.strip()
        if key not in agents:
            raise ValueError(f"Unknown agent key in --agent_priors: {key!r}")
        priors[key] = float(raw)
    return priors


def resolve_write_path(root_dir: Path, value: Optional[str], default_relative: str) -> Path:
    path = Path(value) if value else root_dir / default_relative
    if not path.is_absolute():
        path = root_dir / path
    resolved = path.resolve()
    root_resolved = root_dir.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside root_dir: {resolved}") from exc
    return resolved


def command_line() -> str:
    return " ".join(sys.argv)


def ensure_fixed_agent_keys(mapping: Dict[str, Any]) -> None:
    missing = [agent for agent in FIXED_AGENTS if agent not in mapping]
    if missing:
        raise ValueError(f"Missing fixed agent keys: {missing}")


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percent / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def sum_usage(usages: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for usage in usages:
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
        total_tokens += int(usage.get("total_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
