from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .dataset_gpqa import CHOICES, GPQASample, build_gpqa_cot_prompt
from .llm_client import VLLMClient
from .metrics import build_sample_metrics
from .parsing_gpqa import ParsedGPQAOutput, parse_gpqa_agent_output
from .utils import estimate_tokens, median, percentile, read_json, utc_now_iso, write_json


GPQA_AGENTS = ["qwen", "qwen_coder", "mathstral"]


@dataclass
class GPQADarkForestConfig:
    coordination_method: str = "darkforest"
    design_name: str = "DarkForest"
    fixed_agents: List[str] = field(default_factory=lambda: list(GPQA_AGENTS))
    coordinator_model: str = "qwen_coder"
    agent_priors: Dict[str, float] = field(
        default_factory=lambda: {agent: 1.0 for agent in GPQA_AGENTS}
    )
    support_pattern_reliability: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accept_threshold: float = 0.75
    uncertainty_threshold: float = 0.60
    min_support_pattern_count: int = 5
    expose_belief_summary: bool = True
    expose_full_responses: bool = False
    expose_reasoning: bool = False
    max_peer_response_chars: int = 3000
    max_reasoning_chars: int = 900
    decision_policy: str = "coordinator"
    decision_priority: List[str] = field(default_factory=lambda: ["qwen_coder", "qwen", "mathstral"])
    verifier_model: str = "qwen"
    verifier_trigger: str = "high_uncertainty_or_qwen_coder_disagree"
    anchor_agent: str = "qwen"
    belief_guardrail: str = "none"
    belief_guardrail_min_posterior: float = 0.66
    belief_guardrail_min_margin: float = 0.25
    params_source: str = "default"
    calibration_source: Optional[str] = None
    freeze_calibration: bool = True
    parameter_sources: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_gpqa_belief(
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    config: GPQADarkForestConfig,
) -> Dict[str, Any]:
    clusters: Dict[str, List[ParsedGPQAOutput]] = defaultdict(list)
    for agent in config.fixed_agents:
        output = agent_outputs.get(agent)
        if output and not output.invalid_parse and output.parsed_answer:
            clusters[output.parsed_answer].append(output)

    rows: List[Dict[str, Any]] = []
    for answer, members in clusters.items():
        supporting_agents = [agent for agent in config.fixed_agents if any(m.agent_key == agent for m in members)]
        support_pattern = "+".join(supporting_agents)
        entry = config.support_pattern_reliability.get(support_pattern) or {}
        count = int(entry.get("num", 0) or 0)
        if count >= config.min_support_pattern_count:
            support_prior = float(entry.get("smoothed_accuracy", entry.get("accuracy", 1.0)))
            support_prior_source = "calibrated"
        else:
            support_prior = 1.0
            support_prior_source = "default"
        contributions = {
            member.agent_key: float(config.agent_priors.get(member.agent_key, 1.0))
            for member in members
        }
        score = support_prior * sum(contributions.values())
        rows.append(
            {
                "answer": answer,
                "supporting_agents": supporting_agents,
                "support_pattern": support_pattern,
                "score": score,
                "posterior": 0.0,
                "agent_contributions": contributions,
                "support_prior": support_prior,
                "support_prior_source": support_prior_source,
            }
        )

    total_score = sum(max(0.0, row["score"]) for row in rows)
    if rows and total_score <= 0.0:
        uniform = 1.0 / len(rows)
        for row in rows:
            row["posterior"] = uniform
    elif rows:
        for row in rows:
            row["posterior"] = max(0.0, row["score"]) / total_score
    rows.sort(key=lambda row: (-row["posterior"], -len(row["supporting_agents"]), row["answer"]))
    top = rows[0] if rows else None
    second = rows[1] if len(rows) > 1 else None
    top_posterior = float(top["posterior"]) if top else 0.0
    posterior_margin = top_posterior - float(second["posterior"]) if second else top_posterior
    return {
        "answer_clusters": rows,
        "top_answer": top["answer"] if top else None,
        "top_posterior": top_posterior,
        "posterior_margin": posterior_margin,
        "num_distinct_answers": len(rows),
        "num_invalid_agent_parses": sum(1 for output in agent_outputs.values() if output.invalid_parse),
        "disagreement": len(rows) > 1,
        "high_uncertainty": bool(rows and (top_posterior < config.uncertainty_threshold or posterior_margin < 0.15)),
        "params_source": config.params_source,
    }


def _format_options(sample: GPQASample) -> str:
    return "\n".join(f"{CHOICES[idx]}. {option}" for idx, option in enumerate(sample.options))


def _option_text(sample: GPQASample, answer: Optional[str]) -> Optional[str]:
    if answer not in CHOICES:
        return None
    idx = CHOICES.index(str(answer))
    if idx >= len(sample.options):
        return None
    return sample.options[idx]


def _format_belief_summary(belief_state: Dict[str, Any]) -> str:
    lines = [
        "DarkForest belief summary:",
        f"- top_answer: {belief_state.get('top_answer')}",
        f"- top_posterior: {belief_state.get('top_posterior', 0.0):.4f}",
        f"- posterior_margin: {belief_state.get('posterior_margin', 0.0):.4f}",
        f"- disagreement: {belief_state.get('disagreement')}",
        f"- high_uncertainty: {belief_state.get('high_uncertainty')}",
    ]
    for idx, cluster in enumerate((belief_state.get("answer_clusters") or [])[:3], start=1):
        lines.append(
            f"- cluster_{idx}: answer={cluster.get('answer')} support={cluster.get('supporting_agents')} "
            f"posterior={float(cluster.get('posterior') or 0.0):.4f} "
            f"support_prior={float(cluster.get('support_prior') or 0.0):.4f} "
            f"support_prior_source={cluster.get('support_prior_source')}"
        )
    return "\n".join(lines)


def build_exposed_gpqa_content(
    sample: GPQASample,
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    config: GPQADarkForestConfig,
    belief_state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    parts: List[str] = []
    for agent in config.fixed_agents:
        output = agent_outputs[agent]
        parts.append(f"Agent: {agent}")
        parts.append(f"parsed_answer: {output.parsed_answer}")
        chosen_text = _option_text(sample, output.parsed_answer)
        if chosen_text is not None:
            parts.append(f"chosen_option_text: {chosen_text}")
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


def build_gpqa_coordinator_prompt(
    sample: GPQASample,
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    config: GPQADarkForestConfig,
    belief_state: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    exposed_content, exposure_metrics = build_exposed_gpqa_content(sample, agent_outputs, config, belief_state)
    prompt = (
        "You are coordinating fixed agents on a GPQA multiple-choice question.\n"
        "Think step by step, then finish exactly with \"the answer is (X)\" where X is the correct letter choice.\n"
        "First solve the question yourself from the option text. Then use candidate answers as fallible evidence.\n"
        "Prefer a high-posterior supported cluster unless your reasoning finds a concrete contradiction.\n"
        "When the candidates disagree, compare the option meanings directly instead of averaging blindly.\n\n"
        f"Category: {sample.category}\n"
        "Question:\n"
        f"{sample.question}\n"
        "Options:\n"
        f"{_format_options(sample)}\n\n"
        "Candidate answers:\n"
        f"{exposed_content}\n\n"
        "Answer: Let's think step by step."
    )
    return prompt, exposed_content, exposure_metrics


def should_trigger_gpqa_verifier(
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    belief_state: Dict[str, Any],
    config: GPQADarkForestConfig,
) -> Tuple[bool, str]:
    if config.verifier_trigger == "always":
        return True, "always"
    if config.verifier_trigger == "disagreement":
        triggered = bool(belief_state.get("disagreement"))
        return triggered, "disagreement" if triggered else "no_disagreement"
    if config.verifier_trigger != "high_uncertainty_or_qwen_coder_disagree":
        raise ValueError(f"Unknown GPQA verifier_trigger: {config.verifier_trigger}")

    qwen_answer = agent_outputs.get("qwen").parsed_answer if agent_outputs.get("qwen") else None
    coder_answer = agent_outputs.get("qwen_coder").parsed_answer if agent_outputs.get("qwen_coder") else None
    qwen_coder_disagree = bool(qwen_answer and coder_answer and qwen_answer != coder_answer)
    qwen_or_coder_invalid = not bool(qwen_answer and coder_answer)
    high_uncertainty = bool(belief_state.get("high_uncertainty"))
    if high_uncertainty:
        return True, "high_uncertainty"
    if qwen_coder_disagree:
        return True, "qwen_coder_disagree"
    if qwen_or_coder_invalid:
        return True, "qwen_or_coder_invalid"
    return False, "not_triggered"


def choose_gpqa_majority_vote(
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    config: GPQADarkForestConfig,
) -> Optional[str]:
    counts: Dict[str, int] = defaultdict(int)
    for agent in config.fixed_agents:
        answer = agent_outputs.get(agent).parsed_answer if agent_outputs.get(agent) else None
        if answer:
            counts[answer] += 1
    if not counts:
        return None
    max_count = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == max_count}
    for agent in config.decision_priority:
        answer = agent_outputs.get(agent).parsed_answer if agent_outputs.get(agent) else None
        if answer in tied:
            return answer
    return sorted(tied)[0]


def choose_gpqa_pair_agreement(
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    config: GPQADarkForestConfig,
) -> Optional[str]:
    pairs = [
        ("qwen", "qwen_coder"),
        ("qwen", "mathstral"),
        ("qwen_coder", "mathstral"),
    ]
    for left, right in pairs:
        left_answer = agent_outputs.get(left).parsed_answer if agent_outputs.get(left) else None
        right_answer = agent_outputs.get(right).parsed_answer if agent_outputs.get(right) else None
        if left_answer and left_answer == right_answer:
            return left_answer
    for agent in config.decision_priority:
        answer = agent_outputs.get(agent).parsed_answer if agent_outputs.get(agent) else None
        if answer:
            return answer
    return None


def build_gpqa_verifier_prompt(
    sample: GPQASample,
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    config: GPQADarkForestConfig,
    belief_state: Dict[str, Any],
    coordinator_answer: Optional[str] = None,
    mode: str = "targeted_verifier",
) -> Tuple[str, str, Dict[str, Any]]:
    exposed_content, exposure_metrics = build_exposed_gpqa_content(sample, agent_outputs, config, belief_state)
    anchor_output = agent_outputs.get(config.anchor_agent)
    anchor_answer = anchor_output.parsed_answer if anchor_output else None
    anchor_text = _option_text(sample, anchor_answer)
    if mode == "qwen_anchor_verifier":
        task = (
            "You are a conservative verifier for a GPQA multiple-choice question.\n"
            f"The anchor agent is {config.anchor_agent}. Keep the anchor answer unless another candidate or your own reasoning gives a concrete reason to change it.\n"
            "Do not average votes blindly. Check the option meanings against the question.\n"
        )
    else:
        task = (
            "You are a strict verifier for a GPQA multiple-choice question.\n"
            "A coordinator or agents may be wrong. Re-check the option meanings against the question and select one final letter.\n"
        )
    coordinator_line = ""
    if coordinator_answer:
        coordinator_line = (
            f"Current coordinator answer: {coordinator_answer}"
            f" ({_option_text(sample, coordinator_answer)})\n"
        )
    anchor_line = ""
    if anchor_answer:
        anchor_line = f"Anchor answer: {anchor_answer} ({anchor_text})\n"
    prompt = (
        f"{task}"
        "Think step by step, then finish exactly with \"the answer is (X)\" where X is the correct letter choice.\n\n"
        f"Category: {sample.category}\n"
        "Question:\n"
        f"{sample.question}\n"
        "Options:\n"
        f"{_format_options(sample)}\n\n"
        f"{anchor_line}"
        f"{coordinator_line}"
        "Candidate evidence:\n"
        f"{exposed_content}\n\n"
        "Verified answer: Let's think step by step."
    )
    return prompt, exposed_content, exposure_metrics


def query_gpqa_verifier(
    sample: GPQASample,
    clients: Mapping[str, VLLMClient],
    config: GPQADarkForestConfig,
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    belief_state: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    seed: int,
    coordinator_answer: Optional[str] = None,
    mode: str = "targeted_verifier",
) -> Tuple[ParsedGPQAOutput, str, str, Dict[str, Any]]:
    prompt, exposed_content, exposure_metrics = build_gpqa_verifier_prompt(
        sample,
        agent_outputs,
        config,
        belief_state,
        coordinator_answer=coordinator_answer,
        mode=mode,
    )
    response = clients[config.verifier_model].complete(
        prompt, temperature=temperature, max_tokens=max_tokens, seed=seed
    )
    parsed = parse_gpqa_agent_output(
        "verifier",
        response.text,
        num_choices=len(sample.options),
        latency_sec=response.latency_sec,
        usage=response.usage,
        error=response.error,
    )
    return parsed, prompt, exposed_content, exposure_metrics


def apply_gpqa_belief_guardrail(
    coordinator_answer: Optional[str],
    belief_state: Dict[str, Any],
    config: GPQADarkForestConfig,
) -> Tuple[Optional[str], Dict[str, Any]]:
    info = {
        "policy": config.belief_guardrail,
        "applied": False,
        "original_answer": coordinator_answer,
        "final_answer": coordinator_answer,
        "reason": "disabled",
    }
    if config.belief_guardrail == "none":
        return coordinator_answer, info
    if config.belief_guardrail != "trust_supported_cluster":
        raise ValueError(f"Unknown GPQA belief_guardrail: {config.belief_guardrail}")

    clusters = belief_state.get("answer_clusters") or []
    if not clusters:
        info["reason"] = "no_clusters"
        return coordinator_answer, info
    top = clusters[0]
    top_answer = top.get("answer")
    top_posterior = float(belief_state.get("top_posterior") or 0.0)
    margin = float(belief_state.get("posterior_margin") or 0.0)
    support_pattern = str(top.get("support_pattern") or "")
    support_prior = float(top.get("support_prior") or 0.0)
    trusted_support = support_pattern in {"qwen+qwen_coder", "qwen+qwen_coder+mathstral"}
    trusted_calibration = (
        top.get("support_prior_source") == "calibrated"
        and support_prior >= config.accept_threshold
    )
    if (
        top_answer
        and trusted_support
        and trusted_calibration
        and top_posterior >= config.belief_guardrail_min_posterior
        and margin >= config.belief_guardrail_min_margin
    ):
        info.update(
            {
                "applied": top_answer != coordinator_answer,
                "final_answer": top_answer,
                "reason": "trusted_supported_cluster",
                "top_posterior": top_posterior,
                "posterior_margin": margin,
                "support_pattern": support_pattern,
                "support_prior": support_prior,
            }
        )
        return str(top_answer), info

    info.update(
        {
            "reason": "threshold_not_met",
            "top_posterior": top_posterior,
            "posterior_margin": margin,
            "support_pattern": support_pattern,
            "support_prior": support_prior,
        }
    )
    return coordinator_answer, info


def query_gpqa_initial_agents(
    sample: GPQASample,
    validation_by_category: Mapping[str, List[GPQASample]],
    clients: Mapping[str, VLLMClient],
    config: GPQADarkForestConfig,
    ntrain: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    parallel_agents: bool,
    exclude_validation_self: bool = False,
) -> Tuple[Dict[str, ParsedGPQAOutput], str, float]:
    exclude_id = sample.question_id if exclude_validation_self else None
    prompt = build_gpqa_cot_prompt(sample, dict(validation_by_category), ntrain=ntrain, exclude_question_id=exclude_id)
    num_choices = len(sample.options)

    def call(agent: str) -> Tuple[str, ParsedGPQAOutput]:
        response = clients[agent].complete(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed)
        parsed = parse_gpqa_agent_output(
            agent,
            response.text,
            num_choices=num_choices,
            latency_sec=response.latency_sec,
            usage=response.usage,
            error=response.error,
        )
        return agent, parsed

    start = time.perf_counter()
    outputs: Dict[str, ParsedGPQAOutput] = {}
    if parallel_agents:
        with ThreadPoolExecutor(max_workers=len(config.fixed_agents)) as executor:
            futures = {executor.submit(call, agent): agent for agent in config.fixed_agents}
            for future in as_completed(futures):
                agent, parsed = future.result()
                outputs[agent] = parsed
    else:
        for agent in config.fixed_agents:
            agent, parsed = call(agent)
            outputs[agent] = parsed
    return {agent: outputs[agent] for agent in config.fixed_agents}, prompt, time.perf_counter() - start


def query_gpqa_coordinator(
    sample: GPQASample,
    clients: Mapping[str, VLLMClient],
    config: GPQADarkForestConfig,
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    belief_state: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    seed: int,
) -> Tuple[ParsedGPQAOutput, str, str, Dict[str, Any]]:
    prompt, exposed_content, exposure_metrics = build_gpqa_coordinator_prompt(
        sample, agent_outputs, config, belief_state
    )
    response = clients[config.coordinator_model].complete(
        prompt, temperature=temperature, max_tokens=max_tokens, seed=seed
    )
    parsed = parse_gpqa_agent_output(
        "coordinator",
        response.text,
        num_choices=len(sample.options),
        latency_sec=response.latency_sec,
        usage=response.usage,
        error=response.error,
    )
    return parsed, prompt, exposed_content, exposure_metrics


def score_gpqa_prediction(
    parsed_answer: Optional[str],
    gold_answer: Optional[str],
    num_choices: int,
    invalid_answer_policy: str,
    seed: int,
) -> Tuple[bool, Optional[str], bool]:
    invalid = not bool(parsed_answer)
    answer = parsed_answer
    if invalid and invalid_answer_policy == "random":
        rng = random.Random(seed)
        answer = rng.choice(CHOICES[:num_choices])
    return bool(answer and gold_answer and answer == gold_answer), answer, invalid


def make_gpqa_metrics(
    correct: bool,
    invalid_parse: bool,
    agent_outputs: Mapping[str, ParsedGPQAOutput],
    coordinator_output: Optional[ParsedGPQAOutput],
    initial_latency_sec: float,
    exposure_metrics: Dict[str, Any],
    config: GPQADarkForestConfig,
    verification_output: Optional[ParsedGPQAOutput] = None,
) -> Dict[str, Any]:
    metrics = build_sample_metrics(
        correct=correct,
        invalid_parse=invalid_parse,
        initial_usages=[agent_outputs[agent].usage for agent in config.fixed_agents if agent in agent_outputs],
        coordination_usage=coordinator_output.usage if coordinator_output is not None else None,
        verification_usage=verification_output.usage if verification_output is not None else None,
        initial_latency_sec=initial_latency_sec,
        coordination_latency_sec=coordinator_output.latency_sec if coordinator_output is not None else 0.0,
        verification_latency_sec=verification_output.latency_sec if verification_output is not None else 0.0,
        individual_initial_latencies={
            agent: agent_outputs[agent].latency_sec for agent in config.fixed_agents if agent in agent_outputs
        },
        exposure_metrics=exposure_metrics,
        coordination_rounds=1 if coordinator_output is not None else 0,
        scored=True,
    )
    metrics["llm_calls_by_phase"]["verification"] = 1 if verification_output is not None else 0
    metrics["llm_calls_total"] = sum(metrics["llm_calls_by_phase"].values())
    return metrics


def estimate_gpqa_calibration(
    records: Sequence[Dict[str, Any]],
    config: GPQADarkForestConfig,
    seed: int,
    command: str,
    root_dir: str,
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
        category = record["category"]
        agent_correct = record.get("agent_correct", {})
        for agent in config.fixed_agents:
            output = record["agents"][agent]
            stats = agent_stats[agent]
            stats["num_samples"] += 1
            category_stats[category][agent]["num"] += 1
            if output.get("invalid_parse"):
                stats["num_invalid_parse"] += 1
            else:
                stats["num_valid"] += 1
            if agent_correct.get(agent) is True:
                stats["num_correct"] += 1
                category_stats[category][agent]["num_correct"] += 1
        for cluster in record.get("darkforest_belief", {}).get("answer_clusters") or []:
            pattern = cluster.get("support_pattern")
            answer = cluster.get("answer")
            if not pattern or not answer:
                continue
            support_stats[pattern]["num"] += 1
            if answer == record.get("gold_answer"):
                support_stats[pattern]["num_correct"] += 1

    reliability: Dict[str, Dict[str, Any]] = {}
    priors: Dict[str, float] = {}
    for agent, stats in agent_stats.items():
        num = int(stats["num_samples"])
        correct = int(stats["num_correct"])
        smoothed = (correct + 1.0) / (num + 2.0) if num else 0.5
        priors[agent] = smoothed
        reliability[agent] = {
            **stats,
            "accuracy": correct / num if num else 0.0,
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
            }
    return {
        "dataset": "GPQA",
        "calibration_split": "dev",
        "num_calibration_samples": len(records),
        "fixed_agents": list(config.fixed_agents),
        "agent_reliability": reliability,
        "agent_reliability_by_category": by_category,
        "support_pattern_reliability": support_reliability,
        "learned_darkforest_params": {
            "agent_priors": priors,
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


def apply_gpqa_calibration(
    config: GPQADarkForestConfig,
    calibration: Dict[str, Any],
    source_path: str,
) -> GPQADarkForestConfig:
    learned = calibration.get("learned_darkforest_params") or {}
    if learned.get("agent_priors"):
        config.agent_priors = {agent: float(learned["agent_priors"].get(agent, 1.0)) for agent in config.fixed_agents}
    if learned.get("support_pattern_reliability"):
        config.support_pattern_reliability = dict(learned["support_pattern_reliability"])
    for attr in ["accept_threshold", "uncertainty_threshold", "min_support_pattern_count"]:
        if attr in learned:
            setattr(config, attr, learned[attr])
    config.params_source = "calibrated"
    config.calibration_source = source_path
    config.parameter_sources = {
        "agent_priors": "calibrated",
        "support_pattern_reliability": "calibrated",
        "thresholds": "calibrated",
    }
    return config


def aggregate_gpqa_summary(
    records: Sequence[Dict[str, Any]],
    mode: str,
    config: GPQADarkForestConfig,
    temperature: float,
    max_tokens: int,
    seed: int,
    calibration_file: Optional[str],
    calibration_num_samples: Optional[int],
    total_wall_time_sec: Optional[float],
    invalid_answer_policy: str,
) -> Dict[str, Any]:
    num_samples = len(records)
    num_correct = sum(1 for record in records if record.get("correct") is True)
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

    return {
        "dataset": "GPQA",
        "split": "GoA_GPQA_test",
        "mode": mode,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "accuracy": num_correct / num_samples if num_samples else 0.0,
        "accuracy_percent": 100.0 * num_correct / num_samples if num_samples else 0.0,
        "num_invalid_parse": sum(1 for record in records if record.get("invalid_parse") is True),
        "invalid_parse_rate_percent": (
            100.0 * sum(1 for record in records if record.get("invalid_parse") is True) / num_samples
            if num_samples
            else 0.0
        ),
        "invalid_answer_policy": invalid_answer_policy,
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
    }


def read_calibration(path: str | Path) -> Dict[str, Any]:
    return read_json(Path(path))


def write_records_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n")
