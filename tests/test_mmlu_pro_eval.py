import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.eval_mmlu_pro import (  # noqa: E402
    MMLUProDarkForestConfig,
    apply_mmlu_pro_calibration,
    apply_mmlu_pro_belief_guardrail,
    compute_mmlu_pro_belief,
    score_mmlu_pro_prediction,
)
from darkforest.parsing_mmlu_pro import ParsedMMLUProOutput  # noqa: E402


def _out(agent, answer, invalid=False):
    return ParsedMMLUProOutput(
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
    config = MMLUProDarkForestConfig()
    outputs = {
        "qwen": _out("qwen", "A"),
        "qwen_coder": _out("qwen_coder", "A"),
        "mathstral": _out("mathstral", "B"),
    }
    belief = compute_mmlu_pro_belief(outputs, config)
    assert belief["top_answer"] == "A"
    assert belief["answer_clusters"][0]["supporting_agents"] == ["qwen", "qwen_coder"]
    assert belief["disagreement"] is True


def test_calibration_overrides_priors():
    config = MMLUProDarkForestConfig()
    calibration = {
        "learned_darkforest_params": {
            "agent_priors": {"qwen": 0.2, "qwen_coder": 0.8, "mathstral": 0.3},
            "support_pattern_reliability": {},
        }
    }
    apply_mmlu_pro_calibration(config, calibration, "calibration.json")
    assert config.params_source == "calibrated"
    assert config.agent_priors["qwen_coder"] == 0.8


def test_score_random_invalid_is_deterministic():
    first = score_mmlu_pro_prediction(None, "A", 4, "random", seed=7)
    second = score_mmlu_pro_prediction(None, "A", 4, "random", seed=7)
    assert first == second
    assert first[2] is True


def test_guardrail_trusts_calibrated_qwen_coder_agreement():
    config = MMLUProDarkForestConfig(
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
    belief = compute_mmlu_pro_belief(outputs, config)
    final_answer, info = apply_mmlu_pro_belief_guardrail("B", belief, config)
    assert final_answer == "A"
    assert info["applied"] is True


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
