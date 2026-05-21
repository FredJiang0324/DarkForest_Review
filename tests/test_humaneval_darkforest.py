import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.eval_humaneval import (  # noqa: E402
    HumanEvalDarkForestConfig,
    apply_humaneval_calibration,
    apply_humaneval_guardrail,
    build_humaneval_initial_prompt,
    compute_humaneval_belief,
)
from darkforest.dataset_humaneval import HumanEvalSample  # noqa: E402
from darkforest.parsing_humaneval import ParsedHumanEvalOutput  # noqa: E402


def _out(agent, completion, invalid=False):
    return ParsedHumanEvalOutput(
        agent_key=agent,
        raw_response=completion or "",
        parsed_completion=completion,
        normalized_completion=completion,
        invalid_parse=invalid,
        parse_method="test",
        error=None,
        latency_sec=0.0,
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


def test_humaneval_belief_clusters_identical_code():
    config = HumanEvalDarkForestConfig()
    outputs = {
        "qwen": _out("qwen", "    return x + 1"),
        "qwen_coder": _out("qwen_coder", "    return x + 1"),
        "mathstral": _out("mathstral", "    return x - 1"),
    }
    belief = compute_humaneval_belief(outputs, config)
    assert belief["num_distinct_completions"] == 2
    assert belief["disagreement"] is True
    assert belief["code_clusters"][0]["supporting_agents"] == ["qwen", "qwen_coder"]


def test_humaneval_calibration_overrides_priors():
    config = HumanEvalDarkForestConfig()
    calibration = {
        "learned_darkforest_params": {
            "agent_priors": {"qwen": 0.4, "qwen_coder": 0.9, "mathstral": 0.5},
            "support_pattern_reliability": {},
        }
    }
    apply_humaneval_calibration(config, calibration, "calibration.json")
    assert config.params_source == "calibrated"
    assert config.agent_priors["qwen_coder"] == 0.9


def test_humaneval_guardrail_trusts_supported_cluster():
    config = HumanEvalDarkForestConfig(
        belief_guardrail="coder_anchor_supported_cluster",
        belief_guardrail_min_posterior=0.6,
        belief_guardrail_min_margin=0.1,
    )
    outputs = {
        "qwen": _out("qwen", "    return x + 1"),
        "qwen_coder": _out("qwen_coder", "    return x + 1"),
        "mathstral": _out("mathstral", "    return x - 1"),
    }
    belief = compute_humaneval_belief(outputs, config)
    coordinator = _out("coordinator", "    return x - 1")
    selected, report = apply_humaneval_guardrail(coordinator, belief, outputs, config)
    assert selected == "    return x + 1"
    assert report["selected_source"] == "belief_top_cluster"


def test_humaneval_guardrail_keeps_valid_coordinator_when_anchor_not_trusted():
    config = HumanEvalDarkForestConfig(
        belief_guardrail="coder_anchor_supported_cluster",
        anchor_fallback_min_posterior=0.8,
    )
    outputs = {
        "qwen": _out("qwen", "    return x + 1"),
        "qwen_coder": _out("qwen_coder", "    return x - 1"),
        "mathstral": _out("mathstral", "    return x + 2"),
    }
    belief = compute_humaneval_belief(outputs, config)
    coordinator = _out("coordinator", "    return x + 1")
    selected, report = apply_humaneval_guardrail(coordinator, belief, outputs, config)
    assert selected == "    return x + 1"
    assert report["selected_source"] == "coordinator"
    assert report["reason"] == "valid_coordinator_kept_no_trusted_cluster"


def test_humaneval_guardrail_uses_confident_anchor_top_cluster():
    config = HumanEvalDarkForestConfig(
        belief_guardrail="coder_anchor_supported_cluster",
        anchor_fallback_min_posterior=0.8,
        agent_priors={"qwen": 0.05, "qwen_coder": 0.9, "mathstral": 0.05},
    )
    outputs = {
        "qwen": _out("qwen", "    return x + 1"),
        "qwen_coder": _out("qwen_coder", "    return x - 1"),
        "mathstral": _out("mathstral", "    return x + 2"),
    }
    belief = compute_humaneval_belief(outputs, config)
    coordinator = _out("coordinator", "    return x + 1")
    selected, report = apply_humaneval_guardrail(coordinator, belief, outputs, config)
    assert selected == "    return x - 1"
    assert report["selected_source"] == "qwen_coder"
    assert report["reason"] == "high_confidence_anchor_top_cluster"


def test_humaneval_prompt_requires_self_contained_completion():
    sample = HumanEvalSample(
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        canonical_solution="    return x\n",
        test="def check(candidate):\n    assert candidate(1) == 1\n",
        entry_point="f",
    )
    prompt = build_humaneval_initial_prompt(sample)
    assert "self-contained" in prompt
    assert "define/import" in prompt


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
