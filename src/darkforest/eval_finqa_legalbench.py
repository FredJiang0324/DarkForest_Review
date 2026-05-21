from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .dataset_finqa_legalbench import DomainQASample, build_domain_initial_prompt, strip_trailing_final_answer_marker
from .llm_client import VLLMClient
from .metrics import build_sample_metrics
from .parsing_finqa_legalbench import ParsedDomainOutput, normalize_label, parse_domain_agent_output
from .utils import estimate_tokens, median, percentile, read_json, utc_now_iso, write_json


DOMAIN_AGENTS = ["qwen", "finance_llama", "saul"]
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DomainDarkForestConfig:
    benchmark: str
    coordination_method: str = "darkforest"
    design_name: str = "DarkForest"
    fixed_agents: List[str] = field(default_factory=lambda: list(DOMAIN_AGENTS))
    coordinator_model: str = "qwen"
    agent_priors: Dict[str, float] = field(default_factory=lambda: {agent: 1.0 for agent in DOMAIN_AGENTS})
    agent_priors_by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    support_pattern_reliability: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accept_threshold: float = 0.75
    uncertainty_threshold: float = 0.60
    min_support_pattern_count: int = 3
    expose_belief_summary: bool = True
    expose_full_responses: bool = False
    expose_reasoning: bool = False
    max_peer_response_chars: int = 3000
    max_reasoning_chars: int = 900
    decision_policy: str = "coordinator"
    decision_priority: List[str] = field(default_factory=lambda: ["qwen", "finance_llama", "saul"])
    belief_guardrail: str = "trust_supported_cluster"
    belief_guardrail_min_posterior: float = 0.66
    belief_guardrail_min_margin: float = 0.20
    params_source: str = "default"
    calibration_source: Optional[str] = None
    freeze_calibration: bool = True
    parameter_sources: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_finqa_metric_module() -> Any:
    return _load_module(ROOT / "data/scripts/eval_finqa_metrics.py", "darkforest_finqa_metrics")


def build_prediction_record(sample: DomainQASample, output: ParsedDomainOutput) -> Dict[str, Any]:
    prediction_text = output.raw_response
    if sample.benchmark == "legalbench" and output.parsed_answer is not None:
        prediction_text = f"Final Answer: {output.parsed_answer}"
    record = {
        "id": sample.sample_id,
        "prediction": prediction_text,
    }
    if output.parsed_answer is not None:
        record["final_answer"] = output.parsed_answer
    if output.parsed_program is not None:
        record["program"] = output.parsed_program
    return record


def _finqa_execution_key(value: Any) -> Optional[str]:
    if value in (None, "n/a"):
        return None
    if isinstance(value, str) and value.lower() == "n/a":
        return None
    try:
        number = float(value)
        return f"exec:{number:.8g}"
    except (TypeError, ValueError):
        text = normalize_label(value)
        return f"exec:{text}" if text else None


def score_and_enrich_output(
    sample: DomainQASample,
    output: ParsedDomainOutput,
    benchmark: str,
    finqa_module: Optional[Any] = None,
    finqa_evaluator: Optional[Any] = None,
) -> Tuple[bool, Dict[str, Any]]:
    if benchmark == "legalbench":
        normalized_gold = normalize_label(sample.normalized_gold_answer or sample.gold_answer)
        normalized_pred = normalize_label(output.parsed_answer)
        output.normalized_answer = normalized_pred or None
        output.cluster_key = normalized_pred if normalized_pred and not output.invalid_parse else None
        correct = bool(normalized_pred and normalized_pred == normalized_gold)
        return correct, {
            "normalized_gold": normalized_gold,
            "normalized_prediction": normalized_pred,
            "exact_match_correct": correct,
        }

    if finqa_module is None:
        finqa_module = load_finqa_metric_module()
    if finqa_evaluator is None:
        official_path = ROOT / "data/FinQA/code/evaluate/evaluate.py"
        finqa_evaluator = finqa_module.load_official_evaluator(official_path)

    metrics, details = finqa_module.evaluate(
        [sample.raw_sample],
        [build_prediction_record(sample, output)],
        finqa_evaluator,
    )
    detail = details[0] if details else {}
    output.parsed_program = detail.get("predicted_program") or output.parsed_program
    output.parsed_answer = detail.get("predicted_final_answer") or output.parsed_answer
    output.normalized_answer = normalize_label(output.parsed_answer)
    output.finqa_predicted_execution = detail.get("predicted_execution")
    output.finqa_program_valid = not bool(detail.get("invalid_program_parse"))
    output.finqa_execution_valid = not bool(detail.get("invalid_execution")) and output.finqa_program_valid
    output.invalid_parse = not bool(output.finqa_execution_valid)
    exec_key = _finqa_execution_key(detail.get("predicted_execution"))
    output.cluster_key = exec_key if output.finqa_execution_valid else None
    if output.invalid_parse and not output.error:
        output.error = "no valid executable FinQA program parsed"
    return bool(detail.get("execution_correct")), {
        **detail,
        "metrics": metrics,
    }


def compute_domain_belief(
    agent_outputs: Mapping[str, ParsedDomainOutput],
    config: DomainDarkForestConfig,
    sample: Optional[DomainQASample] = None,
) -> Dict[str, Any]:
    clusters: Dict[str, List[ParsedDomainOutput]] = defaultdict(list)
    for agent in config.fixed_agents:
        output = agent_outputs.get(agent)
        if output and not output.invalid_parse and output.cluster_key:
            clusters[str(output.cluster_key)].append(output)

    rows: List[Dict[str, Any]] = []
    for key, members in clusters.items():
        supporting_agents = [agent for agent in config.fixed_agents if any(m.agent_key == agent for m in members)]
        support_pattern = "+".join(supporting_agents)
        support_entry = config.support_pattern_reliability.get(support_pattern) or {}
        count = int(support_entry.get("num", 0) or 0)
        if count >= config.min_support_pattern_count:
            support_prior = float(support_entry.get("smoothed_accuracy", support_entry.get("accuracy", 1.0)))
            support_prior_source = "calibrated"
        else:
            support_prior = 1.0
            support_prior_source = "default"
        category_priors = config.agent_priors_by_category.get(sample.category, {}) if sample is not None else {}
        contributions = {}
        contribution_sources = {}
        for member in members:
            if member.agent_key in category_priors:
                contributions[member.agent_key] = float(category_priors[member.agent_key])
                contribution_sources[member.agent_key] = "category_calibrated"
            else:
                contributions[member.agent_key] = float(config.agent_priors.get(member.agent_key, 1.0))
                contribution_sources[member.agent_key] = (
                    "global_calibrated" if config.params_source == "calibrated" else "default"
                )
        score = support_prior * sum(contributions.values())
        exemplar = members[0]
        rows.append(
            {
                "cluster_key": key,
                "parsed_answer": exemplar.parsed_answer,
                "parsed_program": exemplar.parsed_program,
                "finqa_predicted_execution": exemplar.finqa_predicted_execution,
                "supporting_agents": supporting_agents,
                "support_pattern": support_pattern,
                "score": score,
                "posterior": 0.0,
                "agent_contributions": contributions,
                "agent_contribution_sources": contribution_sources,
                "support_prior": support_prior,
                "support_prior_source": support_prior_source,
                "priority_rank": min(
                    [
                        config.decision_priority.index(agent)
                        if agent in config.decision_priority
                        else len(config.decision_priority)
                        for agent in supporting_agents
                    ]
                    or [len(config.decision_priority)]
                ),
            }
        )
    total = sum(max(0.0, row["score"]) for row in rows)
    if rows and total <= 0:
        for row in rows:
            row["posterior"] = 1.0 / len(rows)
    elif rows:
        for row in rows:
            row["posterior"] = max(0.0, row["score"]) / total
    rows.sort(
        key=lambda row: (
            -row["posterior"],
            -len(row["supporting_agents"]),
            int(row.get("priority_rank", len(config.decision_priority))),
            row["cluster_key"],
        )
    )
    top = rows[0] if rows else None
    second = rows[1] if len(rows) > 1 else None
    top_posterior = float(top["posterior"]) if top else 0.0
    margin = top_posterior - float(second["posterior"]) if second else top_posterior
    return {
        "answer_clusters": rows,
        "top_cluster_key": top["cluster_key"] if top else None,
        "top_answer": top.get("parsed_answer") if top else None,
        "top_program": top.get("parsed_program") if top else None,
        "top_execution": top.get("finqa_predicted_execution") if top else None,
        "top_posterior": top_posterior,
        "posterior_margin": margin,
        "num_distinct_answers": len(rows),
        "num_invalid_agent_parses": sum(1 for output in agent_outputs.values() if output.invalid_parse),
        "disagreement": len(rows) > 1,
        "high_uncertainty": bool(rows and (top_posterior < config.uncertainty_threshold or margin < 0.15)),
        "params_source": config.params_source,
    }


def _format_belief_summary(belief_state: Dict[str, Any]) -> str:
    lines = [
        "DarkForest belief summary:",
        f"- top_cluster_key: {belief_state.get('top_cluster_key')}",
        f"- top_answer: {belief_state.get('top_answer')}",
        f"- top_execution: {belief_state.get('top_execution')}",
        f"- top_posterior: {float(belief_state.get('top_posterior') or 0.0):.4f}",
        f"- posterior_margin: {float(belief_state.get('posterior_margin') or 0.0):.4f}",
        f"- disagreement: {belief_state.get('disagreement')}",
        f"- high_uncertainty: {belief_state.get('high_uncertainty')}",
    ]
    for idx, cluster in enumerate((belief_state.get("answer_clusters") or [])[:4], start=1):
        lines.append(
            f"- cluster_{idx}: key={cluster.get('cluster_key')} support={cluster.get('supporting_agents')} "
            f"posterior={float(cluster.get('posterior') or 0.0):.4f} "
            f"support_prior={float(cluster.get('support_prior') or 0.0):.4f} "
            f"source={cluster.get('support_prior_source')}"
        )
    return "\n".join(lines)


def build_exposed_domain_content(
    sample: DomainQASample,
    agent_outputs: Mapping[str, ParsedDomainOutput],
    config: DomainDarkForestConfig,
    belief_state: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    parts: List[str] = []
    for agent in config.fixed_agents:
        output = agent_outputs[agent]
        parts.append(f"Agent: {agent}")
        parts.append(f"parsed_answer: {output.parsed_answer}")
        parts.append(f"normalized_answer: {output.normalized_answer}")
        if sample.benchmark == "finqa":
            parts.append(f"parsed_program: {output.parsed_program}")
            parts.append(f"predicted_execution: {output.finqa_predicted_execution}")
            parts.append(f"program_valid: {str(bool(output.finqa_program_valid)).lower()}")
            parts.append(f"execution_valid: {str(bool(output.finqa_execution_valid)).lower()}")
        parts.append(f"cluster_key: {output.cluster_key}")
        parts.append(f"parse_method: {output.parse_method}")
        parts.append(f"invalid_parse: {str(output.invalid_parse).lower()}")
        if output.error:
            parts.append(f"parse_error: {output.error}")
        if config.expose_reasoning and output.reasoning_excerpt:
            parts.append("rationale_excerpt:")
            parts.append(output.reasoning_excerpt[: config.max_reasoning_chars])
        if config.expose_full_responses:
            parts.append("raw_response:")
            parts.append(output.raw_response[: config.max_peer_response_chars])
        parts.append("")
    if config.expose_belief_summary and belief_state is not None:
        parts.append(_format_belief_summary(belief_state))
    text = "\n".join(parts).rstrip()
    return text, {
        "num_agents": len(config.fixed_agents),
        "num_agent_outputs_exposed_to_coordinator": len(config.fixed_agents),
        "cross_agent_input_chars": len(text),
        "cross_agent_input_tokens": estimate_tokens(text),
        "raw_full_response_exposed": bool(config.expose_full_responses),
        "reasoning_exposed": bool(config.expose_reasoning),
        "confidence_exposed": False,
        "belief_summary_exposed": bool(config.expose_belief_summary and belief_state is not None),
    }


def build_domain_coordinator_prompt(
    sample: DomainQASample,
    agent_outputs: Mapping[str, ParsedDomainOutput],
    config: DomainDarkForestConfig,
    belief_state: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    exposed_content, exposure_metrics = build_exposed_domain_content(sample, agent_outputs, config, belief_state)
    if sample.benchmark == "finqa":
        prompt = (
            "You are coordinating fixed agents on a FinQA text-only financial reasoning problem.\n"
            "Solve the problem yourself from the provided text evidence. Candidate programs are fallible.\n"
            "Use only the text evidence; do not use table operations because no table is provided.\n"
            "The final Program must be a comma-separated sequence of FinQA operation calls only. "
            "Do not write infix arithmetic outside operation calls; use #0, #1, etc. for prior steps. "
            "For percentage-change questions, compute the ratio in the Program and express the percent only in Final Answer if needed.\n"
            "If the top DarkForest cluster has a valid executable program, copy that program exactly unless you find a concrete arithmetic or evidence error.\n"
            "Prefer a valid executable program with strong DarkForest support unless you find a concrete contradiction.\n"
            "Finish exactly with two lines:\n"
            "Program: <FinQA-style program>\n"
            "Final Answer: <number>\n\n"
            "Problem:\n"
            f"{sample.prompt.rstrip()}\n\n"
            "Candidate outputs exposed by the orchestrator:\n"
            f"{exposed_content}\n\n"
            "Final solution:"
        )
    elif sample.benchmark == "legalbench":
        choices = ", ".join(sample.answer_choices)
        problem_prompt = strip_trailing_final_answer_marker(sample.prompt.rstrip())
        prompt = (
            "You are coordinating fixed agents on a LegalBench exact-match task.\n"
            "First determine the answer from the legal text and question. Candidate answers are fallible.\n"
            "Use the DarkForest belief summary as compact evidence, but do not average blindly.\n"
            f"Allowed final answers: {choices}\n"
            "Finish exactly with one line:\n"
            "Final Answer: <answer>\n\n"
            "Problem:\n"
            f"{problem_prompt}\n\n"
            "Candidate outputs exposed by the orchestrator:\n"
            f"{exposed_content}\n\n"
            "Final answer:"
        )
    else:
        raise ValueError(f"Unsupported benchmark: {sample.benchmark}")
    return prompt, exposed_content, exposure_metrics


def query_domain_initial_agents(
    sample: DomainQASample,
    clients: Mapping[str, VLLMClient],
    config: DomainDarkForestConfig,
    temperature: float,
    max_tokens: int,
    seed: int,
    parallel_agents: bool,
    finqa_module: Optional[Any] = None,
    finqa_evaluator: Optional[Any] = None,
) -> Tuple[Dict[str, ParsedDomainOutput], str, float, Dict[str, bool], Dict[str, Dict[str, Any]]]:
    prompt = build_domain_initial_prompt(sample)

    def call(agent: str) -> Tuple[str, ParsedDomainOutput, bool, Dict[str, Any]]:
        response = clients[agent].complete(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed)
        parsed = parse_domain_agent_output(
            agent,
            response.text,
            sample.benchmark,
            answer_choices=sample.answer_choices,
            latency_sec=response.latency_sec,
            usage=response.usage,
            error=response.error,
        )
        correct, score_detail = score_and_enrich_output(
            sample,
            parsed,
            sample.benchmark,
            finqa_module=finqa_module,
            finqa_evaluator=finqa_evaluator,
        )
        return agent, parsed, correct, score_detail

    start = time.perf_counter()
    outputs: Dict[str, ParsedDomainOutput] = {}
    correctness: Dict[str, bool] = {}
    score_details: Dict[str, Dict[str, Any]] = {}
    if parallel_agents:
        with ThreadPoolExecutor(max_workers=len(config.fixed_agents)) as executor:
            futures = {executor.submit(call, agent): agent for agent in config.fixed_agents}
            for future in as_completed(futures):
                agent, parsed, correct, detail = future.result()
                outputs[agent] = parsed
                correctness[agent] = correct
                score_details[agent] = detail
    else:
        for agent in config.fixed_agents:
            agent, parsed, correct, detail = call(agent)
            outputs[agent] = parsed
            correctness[agent] = correct
            score_details[agent] = detail
    ordered = {agent: outputs[agent] for agent in config.fixed_agents}
    return ordered, prompt, time.perf_counter() - start, correctness, score_details


def query_domain_coordinator(
    sample: DomainQASample,
    clients: Mapping[str, VLLMClient],
    config: DomainDarkForestConfig,
    agent_outputs: Mapping[str, ParsedDomainOutput],
    belief_state: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    seed: int,
    finqa_module: Optional[Any] = None,
    finqa_evaluator: Optional[Any] = None,
) -> Tuple[ParsedDomainOutput, str, str, Dict[str, Any], bool, Dict[str, Any]]:
    prompt, exposed_content, exposure_metrics = build_domain_coordinator_prompt(
        sample, agent_outputs, config, belief_state
    )
    response = clients[config.coordinator_model].complete(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed)
    parsed = parse_domain_agent_output(
        "coordinator",
        response.text,
        sample.benchmark,
        answer_choices=sample.answer_choices,
        latency_sec=response.latency_sec,
        usage=response.usage,
        error=response.error,
    )
    correct, detail = score_and_enrich_output(
        sample,
        parsed,
        sample.benchmark,
        finqa_module=finqa_module,
        finqa_evaluator=finqa_evaluator,
    )
    return parsed, prompt, exposed_content, exposure_metrics, correct, detail


def _support_count(cluster: Mapping[str, Any]) -> int:
    return len(cluster.get("supporting_agents") or [])


def apply_domain_belief_guardrail(
    coordinator_output: ParsedDomainOutput,
    belief_state: Dict[str, Any],
    config: DomainDarkForestConfig,
    sample: DomainQASample,
    finqa_module: Optional[Any] = None,
    finqa_evaluator: Optional[Any] = None,
) -> Tuple[ParsedDomainOutput, Dict[str, Any], bool, Dict[str, Any]]:
    info = {
        "policy": config.belief_guardrail,
        "applied": False,
        "original_cluster_key": coordinator_output.cluster_key,
        "final_cluster_key": coordinator_output.cluster_key,
        "reason": "disabled",
    }
    correct, detail = score_and_enrich_output(sample, coordinator_output, sample.benchmark, finqa_module, finqa_evaluator)
    if config.belief_guardrail == "none":
        return coordinator_output, info, correct, detail
    if config.belief_guardrail != "trust_supported_cluster":
        raise ValueError(f"Unknown belief_guardrail: {config.belief_guardrail}")
    clusters = belief_state.get("answer_clusters") or []
    if not clusters:
        info["reason"] = "no_clusters"
        return coordinator_output, info, correct, detail
    top = clusters[0]
    top_posterior = float(belief_state.get("top_posterior") or 0.0)
    margin = float(belief_state.get("posterior_margin") or 0.0)
    support_prior = float(top.get("support_prior") or 0.0)
    enough_support = _support_count(top) >= 2
    calibrated_trust = top.get("support_prior_source") == "calibrated" and support_prior >= config.accept_threshold
    single_valid_cluster = (
        len(clusters) == 1
        and (sample.benchmark == "legalbench" or top.get("parsed_program"))
        and float(top.get("posterior") or 0.0) >= 0.99
    )
    if (
        (enough_support or calibrated_trust or single_valid_cluster)
        and top_posterior >= config.belief_guardrail_min_posterior
        and margin >= config.belief_guardrail_min_margin
        and top.get("cluster_key")
        and top.get("cluster_key") != coordinator_output.cluster_key
    ):
        guarded = deepcopy(coordinator_output)
        guarded.cluster_key = str(top.get("cluster_key"))
        guarded.parsed_answer = top.get("parsed_answer")
        guarded.parsed_program = top.get("parsed_program")
        guarded.finqa_predicted_execution = top.get("finqa_predicted_execution")
        guarded.normalized_answer = normalize_label(guarded.parsed_answer)
        correct, detail = score_and_enrich_output(sample, guarded, sample.benchmark, finqa_module, finqa_evaluator)
        info.update(
            {
                "applied": True,
                "final_cluster_key": guarded.cluster_key,
                "reason": "trusted_supported_cluster",
                "top_posterior": top_posterior,
                "posterior_margin": margin,
                "support_pattern": top.get("support_pattern"),
                "support_prior": support_prior,
            }
        )
        return guarded, info, correct, detail
    info.update(
        {
            "reason": "threshold_not_met",
            "top_posterior": top_posterior,
            "posterior_margin": margin,
            "support_pattern": top.get("support_pattern"),
            "support_prior": support_prior,
        }
    )
    return coordinator_output, info, correct, detail


def choose_domain_majority_vote(
    agent_outputs: Mapping[str, ParsedDomainOutput],
    config: DomainDarkForestConfig,
) -> Optional[ParsedDomainOutput]:
    groups: Dict[str, List[ParsedDomainOutput]] = defaultdict(list)
    for agent in config.fixed_agents:
        output = agent_outputs.get(agent)
        if output and output.cluster_key:
            groups[str(output.cluster_key)].append(output)
    if not groups:
        return None
    max_count = max(len(v) for v in groups.values())
    tied_keys = {key for key, values in groups.items() if len(values) == max_count}
    for agent in config.decision_priority:
        output = agent_outputs.get(agent)
        if output and output.cluster_key in tied_keys:
            return output
    return groups[sorted(tied_keys)[0]][0]


def make_domain_metrics(
    correct: bool,
    invalid_parse: bool,
    agent_outputs: Mapping[str, ParsedDomainOutput],
    coordinator_output: Optional[ParsedDomainOutput],
    initial_latency_sec: float,
    exposure_metrics: Dict[str, Any],
    config: DomainDarkForestConfig,
) -> Dict[str, Any]:
    return build_sample_metrics(
        correct=correct,
        invalid_parse=invalid_parse,
        initial_usages=[agent_outputs[agent].usage for agent in config.fixed_agents if agent in agent_outputs],
        coordination_usage=coordinator_output.usage if coordinator_output is not None else None,
        verification_usage=None,
        initial_latency_sec=initial_latency_sec,
        coordination_latency_sec=coordinator_output.latency_sec if coordinator_output is not None else 0.0,
        verification_latency_sec=0.0,
        individual_initial_latencies={
            agent: agent_outputs[agent].latency_sec for agent in config.fixed_agents if agent in agent_outputs
        },
        exposure_metrics=exposure_metrics,
        coordination_rounds=1 if coordinator_output is not None else 0,
        scored=True,
    )


def estimate_domain_calibration(
    records: Sequence[Dict[str, Any]],
    config: DomainDarkForestConfig,
    seed: int,
    command: str,
    root_dir: str,
    calibration_split: str = "calibration",
) -> Dict[str, Any]:
    agent_stats = {
        agent: {"num_samples": 0, "num_valid": 0, "num_correct": 0, "num_invalid_parse": 0}
        for agent in config.fixed_agents
    }
    support_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"num": 0, "num_correct": 0})
    category_stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: {agent: {"num": 0, "num_correct": 0} for agent in config.fixed_agents}
    )
    for record in records:
        category = record.get("category") or config.benchmark
        for agent in config.fixed_agents:
            output = record["agents"][agent]
            stats = agent_stats[agent]
            stats["num_samples"] += 1
            category_stats[category][agent]["num"] += 1
            if output.get("invalid_parse"):
                stats["num_invalid_parse"] += 1
            else:
                stats["num_valid"] += 1
            if record.get("agent_correct", {}).get(agent) is True:
                stats["num_correct"] += 1
                category_stats[category][agent]["num_correct"] += 1
        for cluster in record.get("darkforest_belief", {}).get("answer_clusters") or []:
            pattern = cluster.get("support_pattern")
            key = cluster.get("cluster_key")
            if not pattern or not key:
                continue
            support_stats[pattern]["num"] += 1
            if key in set(record.get("correct_cluster_keys") or []):
                support_stats[pattern]["num_correct"] += 1

    reliability: Dict[str, Dict[str, Any]] = {}
    priors: Dict[str, float] = {}
    for agent, stats in agent_stats.items():
        num = int(stats["num_samples"])
        valid = int(stats["num_valid"])
        correct = int(stats["num_correct"])
        denom = valid if valid else num
        smoothed = (correct + 1.0) / (denom + 2.0) if denom else 0.5
        priors[agent] = smoothed
        reliability[agent] = {
            **stats,
            "accuracy": correct / denom if denom else 0.0,
            "smoothed_accuracy": smoothed,
            "invalid_parse_rate": stats["num_invalid_parse"] / num if num else 0.0,
        }
    support_reliability = {}
    for pattern, stats in support_stats.items():
        num = int(stats["num"])
        correct = int(stats["num_correct"])
        support_reliability[pattern] = {
            "num": num,
            "num_correct": correct,
            "accuracy": correct / num if num else 0.0,
            "smoothed_accuracy": (correct + 1.0) / (num + 2.0) if num else 0.5,
        }
    by_category = {}
    for category, per_agent in category_stats.items():
        by_category[category] = {}
        for agent, stats in per_agent.items():
            num = stats["num"]
            correct = stats["num_correct"]
            by_category[category][agent] = {
                "num": num,
                "num_correct": correct,
                "accuracy": correct / num if num else 0.0,
                "smoothed_accuracy": (correct + 1.0) / (num + 2.0) if num else 0.5,
            }
    return {
        "dataset": "FinQA" if config.benchmark == "finqa" else "LegalBench",
        "calibration_split": calibration_split,
        "num_calibration_samples": len(records),
        "fixed_agents": list(config.fixed_agents),
        "agent_reliability": reliability,
        "agent_reliability_by_category": by_category,
        "support_pattern_reliability": support_reliability,
        "learned_darkforest_params": {
            "agent_priors": priors,
            "agent_priors_by_category": {
                category: {
                    agent: stats["smoothed_accuracy"]
                    for agent, stats in per_agent.items()
                }
                for category, per_agent in by_category.items()
            },
            "support_pattern_reliability": support_reliability,
            "accept_threshold": config.accept_threshold,
            "uncertainty_threshold": config.uncertainty_threshold,
            "min_support_pattern_count": config.min_support_pattern_count,
        },
        "created_at": utc_now_iso(),
        "seed": seed,
        "command": command,
        "root_dir": root_dir,
    }


def apply_domain_calibration(
    config: DomainDarkForestConfig,
    calibration: Dict[str, Any],
    source_path: str,
) -> DomainDarkForestConfig:
    learned = calibration.get("learned_darkforest_params") or {}
    if learned.get("agent_priors"):
        config.agent_priors = {
            agent: float(learned["agent_priors"].get(agent, 1.0)) for agent in config.fixed_agents
        }
    if learned.get("agent_priors_by_category"):
        config.agent_priors_by_category = {
            str(category): {
                agent: float(per_agent.get(agent, config.agent_priors.get(agent, 1.0)))
                for agent in config.fixed_agents
            }
            for category, per_agent in learned["agent_priors_by_category"].items()
            if isinstance(per_agent, dict)
        }
    if learned.get("support_pattern_reliability"):
        config.support_pattern_reliability = dict(learned["support_pattern_reliability"])
    for attr in ["accept_threshold", "uncertainty_threshold", "min_support_pattern_count"]:
        if attr in learned:
            setattr(config, attr, learned[attr])
    config.params_source = "calibrated"
    config.calibration_source = source_path
    config.parameter_sources = {
        "agent_priors": "calibrated",
        "agent_priors_by_category": "calibrated" if config.agent_priors_by_category else "default",
        "support_pattern_reliability": "calibrated",
        "thresholds": "calibrated",
    }
    return config


def write_records_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n")


def run_external_evaluator(
    benchmark: str,
    data_file: Path,
    pred_file: Path,
    details_file: Path,
) -> Dict[str, Any]:
    if benchmark == "finqa":
        cmd = [
            sys.executable,
            str(ROOT / "data/scripts/eval_finqa_metrics.py"),
            "--data",
            str(data_file),
            "--pred",
            str(pred_file),
            "--output",
            str(details_file),
            "--finqa_root",
            str(ROOT / "data/FinQA"),
        ]
    else:
        cmd = [
            sys.executable,
            str(ROOT / "data/scripts/eval_legalbench_exact_match.py"),
            "--data",
            str(data_file),
            "--pred",
            str(pred_file),
            "--output",
            str(details_file),
        ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def aggregate_domain_summary(
    records: Sequence[Dict[str, Any]],
    benchmark: str,
    mode: str,
    config: DomainDarkForestConfig,
    temperature: float,
    max_tokens: int,
    seed: int,
    calibration_file: Optional[str],
    calibration_num_samples: Optional[int],
    total_wall_time_sec: Optional[float],
    evaluator_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evaluator_metrics = evaluator_metrics or {}
    num_samples = len(records)
    if benchmark == "finqa" and evaluator_metrics:
        num_correct = int(evaluator_metrics.get("execution_correct", 0) or 0)
        accuracy = float(evaluator_metrics.get("execution_accuracy", 0.0) or 0.0)
    elif benchmark == "legalbench" and evaluator_metrics:
        num_correct = int(evaluator_metrics.get("correct", 0) or 0)
        accuracy = float(evaluator_metrics.get("exact_match_accuracy", 0.0) or 0.0)
    else:
        num_correct = sum(1 for record in records if record.get("correct") is True)
        accuracy = num_correct / num_samples if num_samples else 0.0
    metrics = [record.get("metrics", {}) for record in records]
    total_tokens = [int(item.get("total_tokens", 0) or 0) for item in metrics]
    input_tokens = [int(item.get("input_tokens_total", 0) or 0) for item in metrics]
    output_tokens = [int(item.get("output_tokens_total", 0) or 0) for item in metrics]
    latencies = [float(item.get("latency_sec_total", 0.0) or 0.0) for item in metrics]
    exposure = [item.get("exposure_metrics", {}) for item in metrics]
    wall = total_wall_time_sec if total_wall_time_sec is not None else sum(latencies)

    def avg(values: Iterable[float | int]) -> float:
        values = list(values)
        return float(sum(values) / num_samples) if num_samples else 0.0

    summary: Dict[str, Any] = {
        "dataset": "FinQA" if benchmark == "finqa" else "LegalBench",
        "split": "finqa_text_only_sample" if benchmark == "finqa" else "legalbench_eval_500",
        "mode": mode,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "num_invalid_parse": sum(1 for record in records if record.get("invalid_parse") is True),
        "invalid_parse_rate_percent": (
            100.0 * sum(1 for record in records if record.get("invalid_parse") is True) / num_samples
            if num_samples
            else 0.0
        ),
        "avg_llm_calls_per_sample": avg(item.get("llm_calls_total", 0) for item in metrics),
        "avg_input_tokens_per_sample": avg(input_tokens),
        "avg_output_tokens_per_sample": avg(output_tokens),
        "avg_total_tokens_per_sample": avg(total_tokens),
        "avg_latency_sec_per_sample": avg(latencies),
        "median_latency_sec_per_sample": median(latencies),
        "p90_latency_sec_per_sample": percentile(latencies, 90),
        "p95_latency_sec_per_sample": percentile(latencies, 95),
        "samples_per_second": num_samples / wall if wall and wall > 0 else 0.0,
        "avg_cross_agent_input_tokens": avg(item.get("cross_agent_input_tokens", 0) for item in exposure),
        "avg_cross_agent_input_chars": avg(item.get("cross_agent_input_chars", 0) for item in exposure),
        "accuracy_per_1k_tokens": num_correct / (sum(total_tokens) / 1000.0) if sum(total_tokens) else 0.0,
        "accuracy_per_second": num_correct / sum(latencies) if sum(latencies) else 0.0,
        "coordination_method": config.coordination_method,
        "design_name": config.design_name,
        "coordinator_model": config.coordinator_model,
        "fixed_agents": list(config.fixed_agents),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "calibration_file": calibration_file,
        "freeze_calibration": config.freeze_calibration,
        "calibration_num_samples": calibration_num_samples,
        "darkforest_params_source": config.params_source,
        "darkforest_config": config.summary_dict(),
        "evaluator_metrics": evaluator_metrics,
    }
    if benchmark == "finqa" and evaluator_metrics:
        summary.update(
            {
                "execution_correct": evaluator_metrics.get("execution_correct"),
                "execution_accuracy": evaluator_metrics.get("execution_accuracy"),
                "execution_accuracy_percent": 100.0 * float(evaluator_metrics.get("execution_accuracy", 0.0)),
                "program_correct": evaluator_metrics.get("program_correct"),
                "program_accuracy": evaluator_metrics.get("program_accuracy"),
                "program_accuracy_percent": 100.0 * float(evaluator_metrics.get("program_accuracy", 0.0)),
                "answer_correct_diagnostic": evaluator_metrics.get("answer_correct_diagnostic"),
                "answer_accuracy_diagnostic": evaluator_metrics.get("answer_accuracy_diagnostic"),
                "answer_accuracy_diagnostic_percent": 100.0
                * float(evaluator_metrics.get("answer_accuracy_diagnostic", 0.0)),
                "primary_metric": "execution_accuracy",
                "secondary_metric": "program_accuracy",
            }
        )
    elif benchmark == "legalbench" and evaluator_metrics:
        summary.update(
            {
                "exact_match_accuracy": evaluator_metrics.get("exact_match_accuracy"),
                "exact_match_accuracy_percent": 100.0
                * float(evaluator_metrics.get("exact_match_accuracy", 0.0)),
                "primary_metric": "exact_match_accuracy",
                "per_task_accuracy": evaluator_metrics.get("per_task_accuracy"),
            }
        )
    return summary


def read_calibration(path: str | Path) -> Dict[str, Any]:
    return read_json(Path(path))
