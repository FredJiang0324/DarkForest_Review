import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.guardrail import apply_belief_guardrail  # noqa: E402
from darkforest.schemas import DarkForestConfig  # noqa: E402


def belief(top_answer="7", posterior=2 / 3, margin=1 / 3, supporters=None):
    supporters = supporters or ["qwen", "qwen_coder"]
    return {
        "posterior_margin": margin,
        "answer_clusters": [
            {
                "normalized_answer": top_answer,
                "posterior": posterior,
                "supporting_agents": supporters,
            }
        ],
    }


def outputs(qwen="5", qwen_coder="7", mathstral="8"):
    return {
        "qwen": {
            "agent_key": "qwen",
            "parsed_answer": qwen,
            "normalized_answer": qwen,
            "invalid_parse": qwen is None,
            "parse_method": "boxed",
        },
        "qwen_coder": {
            "agent_key": "qwen_coder",
            "parsed_answer": qwen_coder,
            "normalized_answer": qwen_coder,
            "invalid_parse": qwen_coder is None,
            "parse_method": "boxed",
        },
        "mathstral": {
            "agent_key": "mathstral",
            "parsed_answer": mathstral,
            "normalized_answer": mathstral,
            "invalid_parse": mathstral is None,
            "parse_method": "boxed",
        },
    }


def test_guardrail_replaces_with_supported_top_cluster():
    config = DarkForestConfig(belief_guardrail="trust_supported_cluster")
    final = {"parsed_answer": "5", "normalized_answer": "5", "invalid_parse": False, "parse_method": "boxed"}
    guarded, report = apply_belief_guardrail(final, belief(), config)
    assert guarded["parsed_answer"] == "7"
    assert guarded["normalized_answer"] == "7"
    assert report["applied"] is True
    assert report["reason"] == "trusted_supported_top_cluster"


def test_guardrail_does_not_replace_singleton_cluster():
    config = DarkForestConfig(belief_guardrail="trust_supported_cluster")
    final = {"parsed_answer": "5", "normalized_answer": "5", "invalid_parse": False, "parse_method": "boxed"}
    guarded, report = apply_belief_guardrail(final, belief(supporters=["qwen"]), config)
    assert guarded["parsed_answer"] == "5"
    assert report["applied"] is False
    assert report["reason"] == "top_cluster_not_supported_by_multiple_agents"


def test_guardrail_does_not_replace_when_disabled():
    config = DarkForestConfig(belief_guardrail="none")
    final = {"parsed_answer": "5", "normalized_answer": "5", "invalid_parse": False, "parse_method": "boxed"}
    guarded, report = apply_belief_guardrail(final, belief(), config)
    assert guarded["parsed_answer"] == "5"
    assert report["applied"] is False
    assert report["reason"] == "disabled"


def test_qwen_anchor_keeps_anchor_when_top_cluster_is_singleton():
    config = DarkForestConfig(
        fixed_agents=["qwen", "qwen_coder", "mathstral"],
        belief_guardrail="qwen_anchor_supported_cluster",
        belief_guardrail_anchor_agent="qwen",
    )
    final = {"parsed_answer": "9", "normalized_answer": "9", "invalid_parse": False, "parse_method": "boxed"}
    guarded, report = apply_belief_guardrail(
        final,
        belief(top_answer="5", supporters=["qwen"], posterior=0.59, margin=0.37),
        config,
        outputs(qwen="5"),
    )
    assert guarded["parsed_answer"] == "5"
    assert guarded["normalized_answer"] == "5"
    assert report["applied"] is True
    assert report["selected_source"] == "anchor_agent"
    assert report["reason"] == "anchor_agent_overrode_coordinator"


def test_qwen_anchor_allows_trusted_multi_agent_cluster():
    config = DarkForestConfig(
        fixed_agents=["qwen", "qwen_coder", "mathstral"],
        belief_guardrail="qwen_anchor_supported_cluster",
        belief_guardrail_anchor_agent="qwen",
        belief_guardrail_min_posterior=0.66,
        belief_guardrail_min_margin=0.25,
    )
    final = {"parsed_answer": "5", "normalized_answer": "5", "invalid_parse": False, "parse_method": "boxed"}
    guarded, report = apply_belief_guardrail(
        final,
        belief(top_answer="7", supporters=["qwen_coder", "mathstral"], posterior=0.74, margin=0.47),
        config,
        outputs(qwen="5", qwen_coder="7", mathstral="7"),
    )
    assert guarded["parsed_answer"] == "7"
    assert guarded["normalized_answer"] == "7"
    assert report["applied"] is True
    assert report["selected_source"] == "belief_top_cluster"
    assert report["reason"] == "trusted_supported_top_cluster_over_anchor"


def test_qwen_anchor_keeps_coordinator_when_anchor_missing_and_no_trusted_cluster():
    config = DarkForestConfig(
        fixed_agents=["qwen", "qwen_coder", "mathstral"],
        belief_guardrail="qwen_anchor_supported_cluster",
        belief_guardrail_anchor_agent="qwen",
    )
    final = {"parsed_answer": "9", "normalized_answer": "9", "invalid_parse": False, "parse_method": "boxed"}
    guarded, report = apply_belief_guardrail(
        final,
        belief(top_answer="7", supporters=["qwen_coder"], posterior=0.55, margin=0.10),
        config,
        outputs(qwen=None, qwen_coder="7", mathstral="8"),
    )
    assert guarded["parsed_answer"] == "9"
    assert report["applied"] is False
    assert report["reason"] == "anchor_unavailable_and_no_trusted_cluster"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
