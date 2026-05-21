from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


FIXED_AGENTS = ["qwen", "mathstral_1", "mathstral_2"]
AGENT_PROFILES = {
    "fixed_mathstral_pair": ["qwen", "mathstral_1", "mathstral_2"],
    "qwen_coder_mathstral": ["qwen", "qwen_coder", "mathstral"],
}


def agents_for_profile(profile: str) -> List[str]:
    if profile not in AGENT_PROFILES:
        raise ValueError(f"Unknown agent profile: {profile}")
    return list(AGENT_PROFILES[profile])


@dataclass
class ParsedAgentOutput:
    agent_key: str
    raw_response: str
    parsed_reasoning: Optional[str]
    parsed_answer: Optional[str]
    normalized_answer: Optional[str]
    confidence: Optional[float]
    malformed_json: bool
    invalid_parse: bool
    parse_method: str
    error: Optional[str]
    latency_sec: float
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExposurePolicy:
    expose_reasoning: bool = False
    expose_confidence: bool = True
    expose_full_responses: bool = False
    expose_belief_summary: bool = True
    max_peer_response_chars: int = 3000
    max_peer_response_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DarkForestConfig:
    coordination_method: str = "darkforest"
    design_name: str = "DarkForest"
    fixed_agents: List[str] = field(default_factory=lambda: list(FIXED_AGENTS))
    coordinator_model: str = "qwen"
    coordination_rounds: int = 1
    agent_priors: Dict[str, float] = field(
        default_factory=lambda: {agent: 1.0 for agent in FIXED_AGENTS}
    )
    support_pattern_reliability: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence_calibration: Dict[str, Any] = field(default_factory=dict)
    same_model_correlation_discount: float = 0.5
    missing_confidence_default: float = 0.5
    malformed_output_penalty: float = 0.5
    accept_threshold: float = 0.75
    uncertainty_threshold: float = 0.60
    min_support_pattern_count: int = 10
    answer_match_backend: str = "exact"
    belief_guardrail: str = "none"
    belief_guardrail_anchor_agent: str = "qwen"
    belief_guardrail_min_posterior: float = 0.66
    belief_guardrail_min_margin: float = 0.25
    exposure_policy: ExposurePolicy = field(default_factory=ExposurePolicy)
    api_style: str = "completions"
    initial_prompt_style: str = "freeform"
    coordinator_prompt_style: str = "darkforest"
    prompt_template_mode: str = "raw"
    prompt_templates: Dict[str, str] = field(default_factory=dict)
    calibration_source: Optional[str] = None
    freeze_calibration: bool = True
    params_source: str = "default"
    parameter_sources: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["exposure_policy"] = self.exposure_policy.to_dict()
        return data

    def darkforest_summary_dict(self) -> Dict[str, Any]:
        return {
            "expose_reasoning": self.exposure_policy.expose_reasoning,
            "expose_confidence": self.exposure_policy.expose_confidence,
            "expose_full_responses": self.exposure_policy.expose_full_responses,
            "expose_belief_summary": self.exposure_policy.expose_belief_summary,
            "same_model_correlation_discount": self.same_model_correlation_discount,
            "missing_confidence_default": self.missing_confidence_default,
            "malformed_output_penalty": self.malformed_output_penalty,
            "agent_priors": dict(self.agent_priors),
            "accept_threshold": self.accept_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "min_support_pattern_count": self.min_support_pattern_count,
            "answer_match_backend": self.answer_match_backend,
            "belief_guardrail": self.belief_guardrail,
            "belief_guardrail_anchor_agent": self.belief_guardrail_anchor_agent,
            "belief_guardrail_min_posterior": self.belief_guardrail_min_posterior,
            "belief_guardrail_min_margin": self.belief_guardrail_min_margin,
            "support_pattern_reliability": self.support_pattern_reliability,
            "confidence_calibration": self.confidence_calibration,
            "api_style": self.api_style,
            "initial_prompt_style": self.initial_prompt_style,
            "coordinator_prompt_style": self.coordinator_prompt_style,
            "prompt_template_mode": self.prompt_template_mode,
            "prompt_templates": dict(self.prompt_templates),
            "parameter_sources": dict(self.parameter_sources),
            "params_source": self.params_source,
            "calibration_source": self.calibration_source,
        }


@dataclass
class MathSample:
    idx: int
    question: str
    solution: Optional[str]
    gold_answer: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResponse:
    text: str
    latency_sec: float
    usage: Dict[str, Any]
    error: Optional[str] = None
    raw_json: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
