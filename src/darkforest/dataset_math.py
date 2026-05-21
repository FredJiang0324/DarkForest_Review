from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .parsing import extract_last_boxed
from .schemas import MathSample
from .utils import deterministic_split


def _candidate_data_paths(root_dir: Path) -> List[Path]:
    return [
        root_dir / "data" / "MATH",
        root_dir / "MATH",
        Path("data/MATH"),
    ]


def resolve_math_data_path(data_path: Optional[str], root_dir: Path) -> Path:
    if data_path:
        path = Path(data_path).expanduser()
        if not path.is_absolute():
            path = root_dir / path
        if not path.exists():
            raise FileNotFoundError(f"MATH data_path does not exist: {path}")
        return path
    for candidate in _candidate_data_paths(root_dir):
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in _candidate_data_paths(root_dir))
    raise FileNotFoundError(
        "MATH data_path was not provided and no local dataset was found. "
        f"Searched: {searched}. No dataset is downloaded automatically."
    )


def _record_gold_answer(record: Dict[str, Any]) -> Optional[str]:
    for key in ("gold_answer", "answer", "final_answer"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    solution = record.get("solution")
    boxed = extract_last_boxed(solution) if solution else None
    return boxed.strip() if boxed else None


def _sample_from_record(
    record: Dict[str, Any],
    idx: int,
    split: str,
    source_path: Path,
    subject: Optional[str] = None,
) -> MathSample:
    question = record.get("problem", record.get("question"))
    if question is None:
        raise ValueError(f"Missing problem/question field in {source_path}")
    metadata = {
        "level": record.get("level"),
        "type": record.get("type", record.get("subject", subject)),
        "subject": record.get("subject", subject or record.get("type")),
        "source_path": str(source_path),
        "split": split,
    }
    return MathSample(
        idx=idx,
        question=str(question),
        solution=record.get("solution"),
        gold_answer=_record_gold_answer(record),
        metadata=metadata,
    )


def _load_jsonl(path: Path, split: str, start_idx: int = 0) -> List[MathSample]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("split") and record.get("split") != split:
                continue
            samples.append(_sample_from_record(record, start_idx + len(samples), split, path))
    return samples


def _load_json_list(path: Path, split: str, start_idx: int = 0) -> List[MathSample]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if split in payload and isinstance(payload[split], list):
            payload = payload[split]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list or object in {path}")
    samples = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        if record.get("split") and record.get("split") != split:
            continue
        samples.append(_sample_from_record(record, start_idx + len(samples), split, path))
    return samples


def _load_original_layout(path: Path, split: str) -> List[MathSample]:
    split_dir = path / split
    if not split_dir.exists():
        raise FileNotFoundError(f"MATH split directory does not exist: {split_dir}")
    samples = []
    for json_path in sorted(split_dir.glob("*/*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        subject = json_path.parent.name
        samples.append(_sample_from_record(record, len(samples), split, json_path, subject=subject))
    return samples


def _load_directory_jsonl_layout(path: Path, split: str) -> Optional[List[MathSample]]:
    split_file = path / f"{split}.jsonl"
    if split_file.exists():
        return _load_jsonl(split_file, split)

    subject_files = sorted(path.glob(f"*/{split}.jsonl"))
    if not subject_files:
        return None

    samples: List[MathSample] = []
    for jsonl_path in subject_files:
        subject = jsonl_path.parent.name
        subject_samples = _load_jsonl(jsonl_path, split, start_idx=len(samples))
        for sample in subject_samples:
            sample.metadata["subject"] = sample.metadata.get("subject") or subject
            sample.metadata["type"] = sample.metadata.get("type") or subject
        samples.extend(subject_samples)
    return samples


def _load_split_from_path(path: Path, split: str) -> List[MathSample]:
    if path.is_file():
        if path.suffix.lower() == ".jsonl":
            return _load_jsonl(path, split)
        if path.suffix.lower() == ".json":
            return _load_json_list(path, split)
        raise ValueError(f"Unsupported MATH file format: {path}")
    jsonl_samples = _load_directory_jsonl_layout(path, split)
    if jsonl_samples is not None:
        return jsonl_samples
    return _load_original_layout(path, split)


def load_math_samples(
    data_path: Optional[str],
    split: str,
    root_dir: Path,
    calibration_valid_fraction: float = 0.2,
    seed: int = 0,
) -> List[MathSample]:
    if split not in {"train", "test", "dev"}:
        raise ValueError(f"Unsupported MATH split: {split}. Expected train, test, or dev.")
    path = resolve_math_data_path(data_path, root_dir)
    if split == "dev":
        train_samples = _load_split_from_path(path, "train")
        _, valid = deterministic_split(train_samples, calibration_valid_fraction, seed)
        for idx, sample in enumerate(valid):
            sample.idx = idx
            sample.metadata["split"] = "dev"
        return valid
    return _load_split_from_path(path, split)


def warn_missing_gold(samples: Iterable[MathSample]) -> List[int]:
    return [sample.idx for sample in samples if not sample.gold_answer]
