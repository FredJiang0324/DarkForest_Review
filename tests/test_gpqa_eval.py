import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.eval_gpqa import (  # noqa: E402
    GPQADarkForestConfig,
    apply_gpqa_calibration,
    apply_gpqa_belief_guardrail,
    choose_gpqa_majority_vote,
    choose_gpqa_pair_agreement,
    compute_gpqa_belief,
    score_gpqa_prediction,
)
from darkforest.parsing_gpqa import ParsedGPQAOutput  # noqa: E402


def _out(agent, answer, invalid=False):
    return ParsedGPQAOutput(
        agent_key=agent,
        raw_response=f"the answer is ({answer})" if answer else "",
        parsed_answer=answer,
        invalid_parse=invalid,
        parse_method="test",
        error=None,
        latency_sec=0.0,
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


def test_belief_clusters_identical_answers():
    config = GPQADarkForestConfig()
    outputs = {
        "qwen": _out("qwen", "A"),
        "qwen_coder": _out("qwen_coder", "A"),
        "mathstral": _out("mathstral", "B"),
    }
    belief = compute_gpqa_belief(outputs, config)
    assert belief["top_answer"] == "A"
    assert belief["answer_clusters"][0]["supporting_agents"] == ["qwen", "qwen_coder"]
    assert belief["disagreement"] is True


def test_calibration_overrides_priors():
    config = GPQADarkForestConfig()
    calibration = {
        "learned_darkforest_params": {
            "agent_priors": {"qwen": 0.2, "qwen_coder": 0.8, "mathstral": 0.3},
            "support_pattern_reliability": {},
        }
    }
    apply_gpqa_calibration(config, calibration, "calibration.json")
    assert config.params_source == "calibrated"
    assert config.agent_priors["qwen_coder"] == 0.8


def test_score_random_invalid_is_deterministic():
    first = score_gpqa_prediction(None, "A", 4, "random", seed=7)
    second = score_gpqa_prediction(None, "A", 4, "random", seed=7)
    assert first == second
    assert first[2] is True


def test_guardrail_trusts_calibrated_qwen_coder_agreement():
    config = GPQADarkForestConfig(
        belief_guardrail="trust_supported_cluster",
        agent_priors={"qwen": 0.6, "qwen_coder": 0.6, "mathstral": 0.2},
        support_pattern_reliability={
            "qwen+qwen_coder": {"num": 10, "smoothed_accuracy": 0.9},
        },
        accept_threshold=0.75,
        min_support_pattern_count=5,
        belief_guardrail_min_posterior=0.5,
        belief_guardrail_min_margin=0.1,
    )
    outputs = {
        "qwen": _out("qwen", "A"),
        "qwen_coder": _out("qwen_coder", "A"),
        "mathstral": _out("mathstral", "B"),
    }
    belief = compute_gpqa_belief(outputs, config)
    final_answer, info = apply_gpqa_belief_guardrail("B", belief, config)
    assert final_answer == "A"
    assert info["applied"] is True


def test_majority_vote_uses_priority_tie_break():
    config = GPQADarkForestConfig(decision_priority=["qwen_coder", "qwen", "mathstral"])
    outputs = {
        "qwen": _out("qwen", "A"),
        "qwen_coder": _out("qwen_coder", "B"),
        "mathstral": _out("mathstral", "C"),
    }
    assert choose_gpqa_majority_vote(outputs, config) == "B"


def test_pair_agreement_prefers_qwen_coder_pair_before_fallback():
    config = GPQADarkForestConfig(decision_priority=["qwen_coder", "qwen", "mathstral"])
    outputs = {
        "qwen": _out("qwen", "A"),
        "qwen_coder": _out("qwen_coder", "B"),
        "mathstral": _out("mathstral", "B"),
    }
    assert choose_gpqa_pair_agreement(outputs, config) == "B"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
