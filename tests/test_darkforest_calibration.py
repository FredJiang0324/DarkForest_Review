import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.bayes import compute_darkforest_belief  # noqa: E402
from darkforest.calibration import (  # noqa: E402
    apply_calibration_to_config,
    estimate_calibration_from_records,
    load_calibration,
    save_calibration,
)
from darkforest.schemas import DarkForestConfig  # noqa: E402


def agent(answer, malformed=False, invalid=False, confidence=0.5):
    return {
        "raw_response": str(answer),
        "parsed_reasoning": "r",
        "parsed_answer": None if invalid else answer,
        "normalized_answer": None if invalid else answer,
        "confidence": confidence,
        "malformed_json": malformed,
        "invalid_parse": invalid,
        "parse_method": "strict_json",
        "error": None,
        "latency_sec": 0.0,
        "usage": {},
    }


def record(idx, gold, qwen, m1, m2):
    return {
        "idx": idx,
        "gold_answer": gold,
        "normalized_gold_answer": gold,
        "metadata": {"subject": "algebra", "level": "1"},
        "agents": {
            "qwen": agent(qwen),
            "mathstral_1": agent(m1, malformed=(idx == 1)),
            "mathstral_2": agent(m2),
        },
    }


def test_calibration_estimates_priors_and_support_patterns():
    config = DarkForestConfig(min_support_pattern_count=1)
    calibration = estimate_calibration_from_records(
        [record(0, "1", "1", "0", "0"), record(1, "1", "1", "0", "0")],
        config,
        seed=123,
        root_dir="",
        command="test",
    )
    assert calibration["agent_reliability"]["qwen"]["smoothed_accuracy"] == 0.75
    assert calibration["agent_reliability"]["mathstral_1"]["smoothed_accuracy"] == 0.25
    assert calibration["learned_darkforest_params"]["agent_priors"]["qwen"] > calibration["learned_darkforest_params"]["agent_priors"]["mathstral_1"]
    assert calibration["support_pattern_reliability"]["qwen"]["num"] == 2
    assert calibration["support_pattern_reliability"]["mathstral_1+mathstral_2"]["num"] == 2


def test_malformed_penalty_and_correlation_are_clipped():
    config = DarkForestConfig(min_support_pattern_count=1)
    records = [record(0, "1", "1", "0", "0"), record(1, "1", "1", "0", "0")]
    calibration = estimate_calibration_from_records(records, config)
    penalty = calibration["learned_darkforest_params"]["malformed_output_penalty"]
    discount = calibration["learned_darkforest_params"]["same_model_correlation_discount"]
    assert 0.1 <= penalty <= 1.0
    assert 0.2 <= discount <= 1.0


def test_calibration_file_loads_and_overrides_default_priors():
    config = DarkForestConfig()
    calibration = estimate_calibration_from_records([record(0, "1", "1", "0", "0")], config)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calibration.json"
        save_calibration(calibration, path)
        loaded = load_calibration(path)
    new_config = apply_calibration_to_config(DarkForestConfig(), loaded)
    assert new_config.params_source == "calibrated"
    assert new_config.agent_priors["qwen"] == loaded["learned_darkforest_params"]["agent_priors"]["qwen"]


def test_frozen_belief_does_not_update_loaded_priors():
    calibration = {
        "learned_darkforest_params": {
            "agent_priors": {"qwen": 0.9, "mathstral_1": 0.4, "mathstral_2": 0.4},
            "same_model_correlation_discount": 0.5,
            "missing_confidence_default": 0.5,
            "malformed_output_penalty": 0.5,
            "accept_threshold": 0.75,
            "uncertainty_threshold": 0.6,
            "support_pattern_reliability": {},
        },
        "confidence_calibration": {},
    }
    config = apply_calibration_to_config(DarkForestConfig(freeze_calibration=True), calibration)
    before = dict(config.agent_priors)
    outputs = {
        "qwen": agent("2"),
        "mathstral_1": agent("3"),
        "mathstral_2": agent("3"),
    }
    compute_darkforest_belief(outputs, config)
    assert config.agent_priors == before
    assert config.freeze_calibration is True


def test_calibration_skips_missing_gold_for_reliability_stats():
    config = DarkForestConfig()
    records = [
        {
            "idx": 0,
            "gold_answer": "1",
            "normalized_gold_answer": "1",
            "metadata": {},
            "agents": {
                "qwen": agent("1"),
                "mathstral_1": agent("2"),
                "mathstral_2": agent("2"),
            },
        },
        {
            "idx": 1,
            "gold_answer": None,
            "normalized_gold_answer": None,
            "metadata": {},
            "agents": {
                "qwen": agent("bad"),
                "mathstral_1": agent("bad"),
                "mathstral_2": agent("bad"),
            },
        },
    ]
    calibration = estimate_calibration_from_records(records, config)
    assert calibration["num_calibration_samples"] == 2
    assert calibration["num_scored_calibration_samples"] == 1
    assert calibration["num_unscored_calibration_samples"] == 1
    assert calibration["agent_reliability"]["qwen"]["num_samples"] == 1
    assert calibration["agent_reliability"]["qwen"]["num_correct"] == 1


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
