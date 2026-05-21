from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping

from .schemas import FIXED_AGENTS, DarkForestConfig, ParsedAgentOutput


def _as_output(value: Any, agent_key: str | None = None) -> ParsedAgentOutput:
    if isinstance(value, ParsedAgentOutput):
        return value
    resolved_agent_key = value.get("agent_key") or agent_key
    return ParsedAgentOutput(
        agent_key=resolved_agent_key,
        raw_response=value.get("raw_response", ""),
        parsed_reasoning=value.get("parsed_reasoning"),
        parsed_answer=value.get("parsed_answer"),
        normalized_answer=value.get("normalized_answer"),
        confidence=value.get("confidence"),
        malformed_json=bool(value.get("malformed_json", False)),
        invalid_parse=bool(value.get("invalid_parse", False)),
        parse_method=value.get("parse_method", "unknown"),
        error=value.get("error"),
        latency_sec=float(value.get("latency_sec", 0.0) or 0.0),
        usage=value.get("usage") or {},
    )


def _agent_order(config: DarkForestConfig | None = None) -> List[str]:
    return list(config.fixed_agents if config is not None else FIXED_AGENTS)


def _ordered_outputs(
    agent_outputs: Mapping[str, Any] | Iterable[Any],
    config: DarkForestConfig | None = None,
) -> List[ParsedAgentOutput]:
    order = _agent_order(config)
    if isinstance(agent_outputs, Mapping):
        return [_as_output(agent_outputs[key], key) for key in order if key in agent_outputs]
    outputs = [_as_output(item) for item in agent_outputs]
    return sorted(outputs, key=lambda item: order.index(item.agent_key) if item.agent_key in order else 99)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _support_prior(pattern: str, config: DarkForestConfig) -> tuple[float, str]:
    entry = config.support_pattern_reliability.get(pattern)
    if not entry:
        return 1.0, "default"
    count = int(entry.get("num", entry.get("count", 0)) or 0)
    if count < config.min_support_pattern_count:
        return 1.0, "default"
    prior = entry.get("smoothed_accuracy", entry.get("accuracy"))
    if prior is None:
        return 1.0, "default"
    return max(0.0, float(prior)), "calibrated"


def compute_darkforest_belief(
    agent_outputs: Mapping[str, Any] | Iterable[Any],
    config: DarkForestConfig,
) -> Dict[str, Any]:
    agent_order = _agent_order(config)
    outputs = _ordered_outputs(agent_outputs, config)
    valid_outputs = [item for item in outputs if not item.invalid_parse and item.normalized_answer]
    clusters: Dict[str, List[ParsedAgentOutput]] = defaultdict(list)
    for output in valid_outputs:
        clusters[str(output.normalized_answer)].append(output)

    raw_cluster_rows = []
    for normalized_answer, members in clusters.items():
        supporting_agents = sorted([member.agent_key for member in members], key=agent_order.index)
        support_pattern = "+".join(supporting_agents)
        support_prior, support_prior_source = _support_prior(support_pattern, config)
        contributions: Dict[str, float] = {}
        independence_weights: Dict[str, float] = {}
        confidences: List[float] = []
        for member in members:
            confidence = (
                float(member.confidence)
                if member.confidence is not None
                else float(config.missing_confidence_default)
            )
            confidence = _clip(confidence)
            confidences.append(confidence)
            confidence_factor = 1.0 + (confidence - 0.5)
            agent_prior = float(config.agent_priors.get(member.agent_key, 1.0))
            parse_penalty = (
                float(config.malformed_output_penalty)
                if (member.malformed_json or member.invalid_parse)
                else 1.0
            )
            independence_weight = 1.0
            if (
                member.agent_key == "mathstral_2"
                and "mathstral_1" in supporting_agents
            ):
                independence_weight = float(config.same_model_correlation_discount)
            independence_weights[member.agent_key] = independence_weight
            contributions[member.agent_key] = (
                agent_prior * parse_penalty * independence_weight * confidence_factor
            )

        score = support_prior * sum(contributions.values())
        raw_cluster_rows.append(
            {
                "normalized_answer": normalized_answer,
                "raw_answers": [member.parsed_answer for member in members],
                "supporting_agents": supporting_agents,
                "support_pattern": support_pattern,
                "score": score,
                "posterior": 0.0,
                "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
                "independence_weights": independence_weights,
                "agent_contributions": contributions,
                "support_prior": support_prior,
                "support_prior_source": support_prior_source,
                "parameter_sources": {
                    "agent_priors": config.parameter_sources.get("agent_priors", config.params_source),
                    "support_prior": support_prior_source,
                    "same_model_correlation_discount": config.parameter_sources.get(
                        "same_model_correlation_discount", config.params_source
                    ),
                    "missing_confidence_default": config.parameter_sources.get(
                        "missing_confidence_default", config.params_source
                    ),
                    "malformed_output_penalty": config.parameter_sources.get(
                        "malformed_output_penalty", config.params_source
                    ),
                },
            }
        )

    total_score = sum(max(0.0, row["score"]) for row in raw_cluster_rows)
    if raw_cluster_rows and total_score <= 0.0:
        uniform = 1.0 / len(raw_cluster_rows)
        for row in raw_cluster_rows:
            row["posterior"] = uniform
    elif raw_cluster_rows:
        for row in raw_cluster_rows:
            row["posterior"] = max(0.0, row["score"]) / total_score

    raw_cluster_rows.sort(key=lambda row: (-row["posterior"], row["support_pattern"], row["normalized_answer"]))
    top = raw_cluster_rows[0] if raw_cluster_rows else None
    second = raw_cluster_rows[1] if len(raw_cluster_rows) > 1 else None
    top_posterior = float(top["posterior"]) if top else 0.0
    posterior_margin = top_posterior - float(second["posterior"]) if second else top_posterior
    disagreement = len(raw_cluster_rows) > 1
    high_uncertainty = bool(
        raw_cluster_rows
        and (top_posterior < config.uncertainty_threshold or posterior_margin < 0.15)
    )
    calibrated_used = any(row["support_prior_source"] == "calibrated" for row in raw_cluster_rows)

    return {
        "answer_clusters": raw_cluster_rows,
        "top_answer": top["normalized_answer"] if top else None,
        "top_posterior": top_posterior,
        "posterior_margin": posterior_margin,
        "num_distinct_answers": len(raw_cluster_rows),
        "num_invalid_agent_parses": sum(1 for item in outputs if item.invalid_parse),
        "num_malformed_json": sum(1 for item in outputs if item.malformed_json),
        "disagreement": disagreement,
        "high_uncertainty": high_uncertainty,
        "same_model_correlation_discount": config.same_model_correlation_discount,
        "params_source": "calibrated" if (config.params_source == "calibrated" or calibrated_used) else "default",
    }
