import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.dataset_humaneval import load_humaneval_samples, split_calibration_and_eval  # noqa: E402


def _sample(task_id):
    return {
        "task_id": task_id,
        "prompt": f"def f_{task_id.replace('/', '_')}():\n",
        "canonical_solution": "    return 1\n",
        "test": "def check(candidate):\n    assert candidate() == 1\n",
        "entry_point": f"f_{task_id.replace('/', '_')}",
    }


def test_loads_jsonl_and_json_subset_split():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        full = root / "test.jsonl"
        subset = root / "human_eval_dev.json"
        rows = [_sample("HumanEval/0"), _sample("HumanEval/1"), _sample("HumanEval/2")]
        full.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        subset.write_text(json.dumps([rows[2], rows[0]]), encoding="utf-8")

        loaded = load_humaneval_samples(full)
        calibration, eval_samples = split_calibration_and_eval(full, subset)

    assert [sample.task_id for sample in loaded] == ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
    assert [sample.task_id for sample in calibration] == ["HumanEval/1"]
    assert [sample.task_id for sample in eval_samples] == ["HumanEval/2", "HumanEval/0"]


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
