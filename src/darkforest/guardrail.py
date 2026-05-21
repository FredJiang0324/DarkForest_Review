from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, Optional, Tuple

from .parsing import math_exact_match, normalize_math_answer
from .schemas import DarkForestConfig, ParsedAgentOutput


def _agent_answer(
    agent_outputs: Optional[Mapping[str, Any] | Iterable[Any]],
    agent_key: str,
) -> Dict[str, Any]:
    if agent_outputs is None:
        return {"found": False}
    value: Any = None
    if isinstance(agent_outputs, Mapping):
        value = agent_outputs.get(agent_key)
    else:
        for item in agent_outputs:
            item_key = getattr(item, "agent_key", None)
            if item_key is None and isinstance(item, Mapping):
                item_key = item.get("agent_key")
            if item_key == agent_key:
                value = item
                break
    if value is None:
        return {"found": False}
    if isinstance(value, ParsedAgentOutput):
        parsed = value.parsed_answer
        normalized = value.normalized_answer
        invalid = value.invalid_parse
        parse_method = value.parse_method
    elif isinstance(value, Mapping):
        parsed = value.get("parsed_answer")
        normalized = value.get("normalized_answer")
        invalid = bool(value.get("invalid_parse"))
        parse_method = value.get("parse_method")
    else:
        parsed = getattr(value, "parsed_answer", None)
        normalized = getattr(value, "normalized_answer", None)
        invalid = bool(getattr(value, "invalid_parse", True))
        parse_method = getattr(value, "parse_method", None)
    return {
        "found": True,
        "parsed_answer": parsed,
        "normalized_answer": normalize_math_answer(normalized or parsed),
        "invalid_parse": invalid,
        "parse_method": parse_method,
    }


def _select_answer(
    final_parse: Dict[str, Any],
    selected_answer: str,
    parse_method_suffix: str,
) -> Dict[str, Any]:
    guarded = dict(final_parse)
    guarded["parsed_answer"] = selected_answer
    guarded["normalized_answer"] = normalize_math_answer(selected_answer)
    guarded["parse_method"] = f"{final_parse.get('parse_method', 'unknown')}+{parse_method_suffix}"
    guarded["invalid_parse"] = not bool(guarded["normalized_answer"])
    return guarded


def apply_belief_guardrail(
    final_parse: Dict[str, Any],
    belief_state: Dict[str, Any],
    config: DarkForestConfig,
    agent_outputs: Optional[Mapping[str, Any] | Iterable[Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report: Dict[str, Any] = {
        "mode": config.belief_guardrail,
        "applied": False,
        "reason": "disabled" if config.belief_guardrail == "none" else None,
        "selected_source": "coordinator",
        "original_parsed_answer": final_parse.get("parsed_answer"),
        "original_normalized_answer": final_parse.get("normalized_answer"),
        "replacement_answer": None,
        "replacement_normalized_answer": None,
        "replacement_supporting_agents": [],
        "replacement_posterior": None,
        "replacement_margin": belief_state.get("posterior_margin"),
        "anchor_agent": config.belief_guardrail_anchor_agent,
        "anchor_parsed_answer": None,
        "anchor_normalized_answer": None,
        "anchor_invalid_parse": None,
        "min_posterior": config.belief_guardrail_min_posterior,
        "min_margin": config.belief_guardrail_min_margin,
    }
    if config.belief_guardrail == "none":
        return final_parse, report
    if config.belief_guardrail not in {"trust_supported_cluster", "qwen_anchor_supported_cluster"}:
        report["reason"] = f"unknown_mode:{config.belief_guardrail}"
        return final_parse, report

    clusters = belief_state.get("answer_clusters") or []
    if not clusters and config.belief_guardrail == "trust_supported_cluster":
        report["reason"] = "no_clusters"
        return final_parse, report
    top = clusters[0] if clusters else {}
    top_answer = top.get("normalized_answer")
    supporting_agents = list(top.get("supporting_agents") or [])
    posterior = float(top.get("posterior") or 0.0)
    margin = float(belief_state.get("posterior_margin") or 0.0)
    if top:
        report.update(
            {
                "replacement_answer": top_answer,
                "replacement_normalized_answer": normalize_math_answer(top_answer),
                "replacement_supporting_agents": supporting_agents,
                "replacement_posterior": posterior,
                "replacement_margin": margin,
            }
        )

    if config.belief_guardrail == "qwen_anchor_supported_cluster":
        anchor = _agent_answer(agent_outputs, config.belief_guardrail_anchor_agent)
        anchor_answer = anchor.get("normalized_answer")
        anchor_valid = bool(anchor.get("found") and anchor_answer and not anchor.get("invalid_parse"))
        report.update(
            {
                "anchor_parsed_answer": anchor.get("parsed_answer"),
                "anchor_normalized_answer": anchor_answer,
                "anchor_invalid_parse": anchor.get("invalid_parse"),
            }
        )
        trusted_top = bool(
            top_answer
            and len(supporting_agents) >= 2
            and posterior >= config.belief_guardrail_min_posterior
            and margin >= config.belief_guardrail_min_margin
        )
        if trusted_top:
            report["selected_source"] = "belief_top_cluster"
            if math_exact_match(final_parse.get("parsed_answer") or final_parse.get("normalized_answer"), top_answer):
                report["reason"] = "coordinator_already_matches_supported_top_cluster"
                return final_parse, report
            guarded = _select_answer(final_parse, str(top_answer), "qwen_anchor_supported_cluster")
            report["applied"] = True
            report["reason"] = (
                "trusted_supported_top_cluster_over_anchor"
                if anchor_valid and not math_exact_match(anchor_answer, top_answer)
                else "trusted_supported_top_cluster"
            )
            return guarded, report

        if anchor_valid:
            report["selected_source"] = "anchor_agent"
            report["replacement_answer"] = anchor.get("parsed_answer") or anchor_answer
            report["replacement_normalized_answer"] = anchor_answer
            if math_exact_match(final_parse.get("parsed_answer") or final_parse.get("normalized_answer"), anchor_answer):
                report["reason"] = "coordinator_already_matches_anchor"
                return final_parse, report
            guarded = _select_answer(
                final_parse,
                str(anchor.get("parsed_answer") or anchor_answer),
                "qwen_anchor",
            )
            report["applied"] = True
            report["reason"] = "anchor_agent_overrode_coordinator"
            return guarded, report

        report["reason"] = "anchor_unavailable_and_no_trusted_cluster"
        return final_parse, report

    if len(supporting_agents) < 2:
        report["reason"] = "top_cluster_not_supported_by_multiple_agents"
        return final_parse, report
    if posterior < config.belief_guardrail_min_posterior:
        report["reason"] = "posterior_below_threshold"
        return final_parse, report
    if margin < config.belief_guardrail_min_margin:
        report["reason"] = "margin_below_threshold"
        return final_parse, report
    if math_exact_match(final_parse.get("parsed_answer") or final_parse.get("normalized_answer"), top_answer):
        report["reason"] = "coordinator_already_matches_top_cluster"
        return final_parse, report

    guarded = _select_answer(final_parse, str(top_answer), "belief_guardrail")
    report["applied"] = True
    report["selected_source"] = "belief_top_cluster"
    report["reason"] = "trusted_supported_top_cluster"
    return guarded, report
