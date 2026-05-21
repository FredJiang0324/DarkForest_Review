import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.bayes import compute_darkforest_belief  # noqa: E402
from darkforest.schemas import DarkForestConfig, ParsedAgentOutput  # noqa: E402


def out(agent, answer, confidence=0.5, malformed=False, invalid=False):
    return ParsedAgentOutput(
        agent_key=agent,
        raw_response=str(answer),
        parsed_reasoning="reason",
        parsed_answer=answer if not invalid else None,
        normalized_answer=answer if not invalid else None,
        confidence=confidence,
        malformed_json=malformed,
        invalid_parse=invalid,
        parse_method="strict_json",
        error=None,
        latency_sec=0.0,
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


def test_identical_answers_cluster_and_posteriors_sum():
    config = DarkForestConfig()
    belief = compute_darkforest_belief(
        {"qwen": out("qwen", "7"), "mathstral_1": out("mathstral_1", "7"), "mathstral_2": out("mathstral_2", "7")},
        config,
    )
    assert belief["num_distinct_answers"] == 1
    assert belief["top_answer"] == "7"
    assert abs(sum(c["posterior"] for c in belief["answer_clusters"]) - 1.0) < 1e-9


def test_distinct_answers_disagree():
    config = DarkForestConfig()
    belief = compute_darkforest_belief(
        {"qwen": out("qwen", "7"), "mathstral_1": out("mathstral_1", "8"), "mathstral_2": out("mathstral_2", "8")},
        config,
    )
    assert belief["disagreement"] is True
    assert belief["num_distinct_answers"] == 2


def test_mathstral_2_discount_when_mathstral_1_same_answer():
    config = DarkForestConfig(same_model_correlation_discount=0.25)
    belief = compute_darkforest_belief(
        {"qwen": out("qwen", "1"), "mathstral_1": out("mathstral_1", "2"), "mathstral_2": out("mathstral_2", "2")},
        config,
    )
    cluster = [c for c in belief["answer_clusters"] if c["normalized_answer"] == "2"][0]
    assert cluster["independence_weights"]["mathstral_2"] == 0.25
    assert cluster["agent_contributions"]["mathstral_2"] == 0.25


def test_malformed_penalty_is_applied():
    config = DarkForestConfig(malformed_output_penalty=0.4)
    belief = compute_darkforest_belief(
        {"qwen": out("qwen", "1", malformed=True), "mathstral_1": out("mathstral_1", "2"), "mathstral_2": out("mathstral_2", "3")},
        config,
    )
    cluster = [c for c in belief["answer_clusters"] if c["normalized_answer"] == "1"][0]
    assert cluster["agent_contributions"]["qwen"] == 0.4


def test_calibrated_support_pattern_overrides_default():
    config = DarkForestConfig(
        min_support_pattern_count=1,
        support_pattern_reliability={"qwen+mathstral_1": {"num": 5, "smoothed_accuracy": 0.9}},
    )
    belief = compute_darkforest_belief(
        {"qwen": out("qwen", "9"), "mathstral_1": out("mathstral_1", "9"), "mathstral_2": out("mathstral_2", "8")},
        config,
    )
    cluster = [c for c in belief["answer_clusters"] if c["normalized_answer"] == "9"][0]
    assert cluster["support_prior"] == 0.9
    assert cluster["support_prior_source"] == "calibrated"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
