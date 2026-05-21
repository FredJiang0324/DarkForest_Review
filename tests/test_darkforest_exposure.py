import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.bayes import compute_darkforest_belief  # noqa: E402
from darkforest.coordination import (  # noqa: E402
    build_initial_agent_prompt,
    build_exposed_agent_content,
    build_static_coordinator_prompt_with_exposure,
)
from darkforest.schemas import DarkForestConfig, ExposurePolicy, ParsedAgentOutput  # noqa: E402


def out(agent):
    return ParsedAgentOutput(
        agent_key=agent,
        raw_response="SECRET_RAW_RESPONSE",
        parsed_reasoning="SECRET_REASONING",
        parsed_answer="42",
        normalized_answer="42",
        confidence=0.7,
        malformed_json=False,
        invalid_parse=False,
        parse_method="strict_json",
        error=None,
        latency_sec=0.0,
        usage={},
    )


def outputs():
    return {"qwen": out("qwen"), "mathstral_1": out("mathstral_1"), "mathstral_2": out("mathstral_2")}


def test_exposure_policy_hides_raw_reasoning_and_confidence():
    policy = ExposurePolicy(expose_reasoning=False, expose_confidence=False, expose_full_responses=False, expose_belief_summary=False)
    exposed, metrics = build_exposed_agent_content(outputs(), policy)
    assert "SECRET_RAW_RESPONSE" not in exposed
    assert "SECRET_REASONING" not in exposed
    assert "confidence" not in exposed
    assert "DarkForest belief summary" not in exposed
    assert metrics["cross_agent_input_chars"] == len(exposed)
    assert metrics["cross_agent_input_tokens"] > 0


def test_belief_summary_policy():
    config = DarkForestConfig()
    belief = compute_darkforest_belief(outputs(), config)
    exposed, metrics = build_exposed_agent_content(outputs(), config.exposure_policy, belief)
    assert "DarkForest belief summary" in exposed
    assert "confidence:" in exposed
    assert metrics["belief_summary_exposed"] is True


def test_exposed_content_exactly_appears_in_coordinator_prompt():
    config = DarkForestConfig()
    belief = compute_darkforest_belief(outputs(), config)
    prompt, exposed, metrics = build_static_coordinator_prompt_with_exposure("What is 40+2?", outputs(), config, belief)
    assert exposed in prompt
    assert metrics["num_agent_outputs_exposed_to_coordinator"] == 3


def test_darkforest_audit_coordinator_prompt_instructs_independent_audit():
    config = DarkForestConfig(coordinator_prompt_style="darkforest_audit")
    belief = compute_darkforest_belief(outputs(), config)
    prompt, exposed, _ = build_static_coordinator_prompt_with_exposure("What is 40+2?", outputs(), config, belief)
    assert exposed in prompt
    assert "Your job is not to vote" in prompt
    assert "solve from scratch" in prompt


def test_darkforest_belief_audit_prompt_uses_belief_directive():
    config = DarkForestConfig(coordinator_prompt_style="darkforest_belief_audit")
    belief = compute_darkforest_belief(outputs(), config)
    prompt, exposed, _ = build_static_coordinator_prompt_with_exposure("What is 40+2?", outputs(), config, belief)
    assert exposed in prompt
    assert "Belief-conditioned procedure" in prompt
    assert "coordinator_directive:" in prompt
    assert "singleton candidate" in prompt


def test_initial_prompt_contains_literal_boxed_placeholder():
    prompt = build_initial_agent_prompt("What is 1+1?")
    assert "Question: What is 1+1?" in prompt
    assert 'The answer is \\\\boxed{X}' in prompt
    assert '"confidence_level": "<a float between 0.0 and 1.0>"' not in prompt


def test_legacy_json_initial_prompt_is_available():
    prompt = build_initial_agent_prompt("What is 1+1?", initial_prompt_style="json")
    assert "Question: What is 1+1?" in prompt
    assert 'The answer is \\\\boxed{X}' in prompt
    assert '"confidence_level": "<a float between 0.0 and 1.0>"' in prompt


def test_full_response_token_limit_truncates_exposure():
    long_raw = "x" * 100
    agent_outputs = outputs()
    for output in agent_outputs.values():
        output.raw_response = long_raw
    policy = ExposurePolicy(
        expose_full_responses=True,
        max_peer_response_chars=3000,
        max_peer_response_tokens=2,
    )
    exposed, _ = build_exposed_agent_content(agent_outputs, policy)
    assert "xxxxxxxx\n[truncated]" in exposed
    assert "x" * 20 not in exposed


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
