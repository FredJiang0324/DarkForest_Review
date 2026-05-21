import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.dataset_math import load_math_samples  # noqa: E402


def test_original_math_layout_loads_and_extracts_gold():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train" / "algebra").mkdir(parents=True)
        (root / "test" / "algebra").mkdir(parents=True)
        (root / "train" / "algebra" / "1.json").write_text(
            json.dumps({"problem": "1+1?", "solution": "Thus \\boxed{2}.", "level": "1", "type": "algebra"}),
            encoding="utf-8",
        )
        (root / "test" / "algebra" / "2.json").write_text(
            json.dumps({"problem": "2+2?", "solution": "Thus \\boxed{4}.", "level": "1", "type": "algebra"}),
            encoding="utf-8",
        )
        train = load_math_samples(str(root), "train", ROOT)
        test = load_math_samples(str(root), "test", ROOT)
    assert len(train) == 1
    assert train[0].gold_answer == "2"
    assert train[0].metadata["subject"] == "algebra"
    assert len(test) == 1
    assert test[0].gold_answer == "4"


def test_directory_jsonl_layout_loads_top_level_split_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_record = {
            "problem": "3+3?",
            "solution": "Thus \\boxed{6}.",
            "level": "Level 1",
            "type": "Algebra",
            "subject": "algebra",
        }
        test_record = {
            "problem": "4+4?",
            "solution": "Thus \\boxed{8}.",
            "level": "Level 1",
            "type": "Algebra",
            "subject": "algebra",
        }
        (root / "train.jsonl").write_text(json.dumps(train_record) + "\n", encoding="utf-8")
        (root / "test.jsonl").write_text(json.dumps(test_record) + "\n", encoding="utf-8")
        train = load_math_samples(str(root), "train", ROOT)
        test = load_math_samples(str(root), "test", ROOT)
    assert len(train) == 1
    assert train[0].gold_answer == "6"
    assert train[0].metadata["source_path"].endswith("train.jsonl")
    assert len(test) == 1
    assert test[0].gold_answer == "8"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
