from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .dataset_humaneval import HumanEvalSample, write_humaneval_problem_file
from .llm_client import VLLMClient
from .metrics import build_sample_metrics
from .parsing_humaneval import ParsedHumanEvalOutput, parse_humaneval_agent_output
from .utils import estimate_tokens, median, percentile, read_jsonl, utc_now_iso, write_json


HUMANEVAL_AGENTS = ["qwen", "qwen_coder", "mathstral"]


@dataclass
class HumanEvalDarkForestConfig:
    coordination_method: str = "darkforest"
    design_name: str = "DarkForest"
    fixed_agents: List[str] = field(default_factory=lambda: list(HUMANEVAL_AGENTS))
    coordinator_model: str = "qwen_coder"
    agent_priors: Dict[str, float] = field(
        default_factory=lambda: {agent: 1.0 for agent in HUMANEVAL_AGENTS}
    )
    support_pattern_reliability: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accept_threshold: float = 0.75
    uncertainty_threshold: float = 0.60
    min_support_pattern_count: int = 5
    expose_belief_summary: bool = True
    expose_full_responses: bool = False
    max_peer_response_chars: int = 4000
    belief_guardrail: str = "coder_anchor_supported_cluster"
    anchor_agent: str = "qwen_coder"
    belief_guardrail_min_posterior: float = 0.66
    belief_guardrail_min_margin: float = 0.20
    anchor_fallback_min_posterior: float = 0.80
    params_source: str = "default"
    calibration_source: Optional[str] = None
    freeze_calibration: bool = True
    parameter_sources: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "coordination_method": self.coordination_method,
            "design_name": self.design_name,
            "fixed_agents": list(self.fixed_agents),
            "coordinator_model": self.coordinator_model,
            "agent_priors": dict(self.agent_priors),
            "support_pattern_reliability": self.support_pattern_reliability,
            "accept_threshold": self.accept_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "min_support_pattern_count": self.min_support_pattern_count,
            "expose_belief_summary": self.expose_belief_summary,
            "expose_full_responses": self.expose_full_responses,
            "max_peer_response_chars": self.max_peer_response_chars,
            "belief_guardrail": self.belief_guardrail,
            "anchor_agent": self.anchor_agent,
            "belief_guardrail_min_posterior": self.belief_guardrail_min_posterior,
            "belief_guardrail_min_margin": self.belief_guardrail_min_margin,
            "anchor_fallback_min_posterior": self.anchor_fallback_min_posterior,
            "params_source": self.params_source,
            "calibration_source": self.calibration_source,
            "freeze_calibration": self.freeze_calibration,
            "parameter_sources": dict(self.parameter_sources),
        }


def build_humaneval_initial_prompt(sample: HumanEvalSample) -> str:
    return (
        "Complete the following Python function.\n"
        "Return only the code completion that should be appended after the prompt.\n"
        "Do not include markdown fences, explanations, tests, or the original prompt.\n\n"
        "Your completion must be self-contained inside the function body.\n"
        "Do not call helper functions, classes, or modules unless they are already defined in the prompt "
        "or you define/import them inside the completion.\n\n"
        f"{sample.prompt}"
    )


def _completion_preview(code: Optional[str], limit: int = 500) -> Optional[str]:
    if code is None:
        return None
    return code if len(code) <= limit else code[:limit] + "...<truncated>"


def compute_humaneval_belief(
    agent_outputs: Mapping[str, ParsedHumanEvalOutput],
    config: HumanEvalDarkForestConfig,
) -> Dict[str, Any]:
    clusters: Dict[str, List[ParsedHumanEvalOutput]] = defaultdict(list)
    for agent in config.fixed_agents:
        output = agent_outputs.get(agent)
        if output and not output.invalid_parse and output.normalized_completion:
            clusters[output.normalized_completion].append(output)

    rows: List[Dict[str, Any]] = []
    for normalized_completion, members in clusters.items():
        supporting_agents = [agent for agent in config.fixed_agents if any(m.agent_key == agent for m in members)]
        support_pattern = "+".join(supporting_agents)
        support_entry = config.support_pattern_reliability.get(support_pattern) or {}
        support_count = int(support_entry.get("num", 0) or 0)
        if support_count >= config.min_support_pattern_count:
            support_prior = float(support_entry.get("smoothed_accuracy", support_entry.get("accuracy", 1.0)))
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
                "normalized_completion": normalized_completion,
                "completion_preview": _completion_preview(normalized_completion),
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

    rows.sort(key=lambda row: (-row["posterior"], -len(row["supporting_agents"]), row["support_pattern"]))
    top = rows[0] if rows else None
    second = rows[1] if len(rows) > 1 else None
    top_posterior = float(top["posterior"]) if top else 0.0
    posterior_margin = top_posterior - float(second["posterior"]) if second else top_posterior
    high_uncertainty = bool(rows and (top_posterior < config.uncertainty_threshold or posterior_margin < 0.15))
    return {
        "code_clusters": rows,
        "top_completion": top["normalized_completion"] if top else None,
        "top_completion_preview": top["completion_preview"] if top else None,
        "top_posterior": top_posterior,
        "posterior_margin": posterior_margin,
        "num_distinct_completions": len(rows),
        "num_invalid_agent_parses": sum(1 for item in agent_outputs.values() if item.invalid_parse),
        "disagreement": len(rows) > 1,
        "high_uncertainty": high_uncertainty,
        "params_source": config.params_source,
    }


def _format_belief_summary(belief_state: Dict[str, Any]) -> str:
    lines = [
        "DarkForest belief summary:",
        f"- top_posterior: {belief_state.get('top_posterior', 0.0):.4f}",
        f"- posterior_margin: {belief_state.get('posterior_margin', 0.0):.4f}",
        f"- num_distinct_completions: {belief_state.get('num_distinct_completions', 0)}",
        f"- disagreement: {belief_state.get('disagreement')}",
        f"- high_uncertainty: {belief_state.get('high_uncertainty')}",
    ]
    clusters = belief_state.get("code_clusters") or []
    for idx, cluster in enumerate(clusters[:3], start=1):
        lines.append(
            f"- cluster_{idx}: support={cluster.get('supporting_agents')} "
            f"posterior={float(cluster.get('posterior') or 0.0):.4f}"
        )
    return "\n".join(lines)


def build_exposed_humaneval_content(
    agent_outputs: Mapping[str, ParsedHumanEvalOutput],
    config: HumanEvalDarkForestConfig,
    belief_state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    parts: List[str] = []
    for agent in config.fixed_agents:
        output = agent_outputs[agent]
        parts.append(f"Agent: {agent}")
        parts.append(f"parse_method: {output.parse_method}")
        parts.append(f"invalid_parse: {str(output.invalid_parse).lower()}")
        if output.error:
            parts.append(f"parse_error: {output.error}")
        parts.append("parsed_completion:")
        parts.append("```python")
        parts.append(output.parsed_completion or "")
        parts.append("```")
        if config.expose_full_responses:
            raw = output.raw_response[: config.max_peer_response_chars]
            parts.append("raw_response:")
            parts.append(raw)
        parts.append("")

    if config.expose_belief_summary and belief_state is not None:
        parts.append(_format_belief_summary(belief_state))

    text = "\n".join(parts).rstrip()
    metrics = {
        "num_agents": len(config.fixed_agents),
        "num_agent_outputs_exposed_to_coordinator": len(config.fixed_agents),
        "cross_agent_input_chars": len(text),
        "cross_agent_input_tokens": estimate_tokens(text),
        "raw_full_response_exposed": bool(config.expose_full_responses),
        "reasoning_exposed": False,
        "confidence_exposed": False,
        "belief_summary_exposed": bool(config.expose_belief_summary and belief_state is not None),
    }
    return text, metrics


def build_humaneval_coordinator_prompt(
    sample: HumanEvalSample,
    agent_outputs: Mapping[str, ParsedHumanEvalOutput],
    config: HumanEvalDarkForestConfig,
    belief_state: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    exposed_content, exposure_metrics = build_exposed_humaneval_content(agent_outputs, config, belief_state)
    prompt = (
        "You are coordinating fixed code agents on a HumanEval problem.\n"
        "Produce exactly one Python code completion that should be appended after the prompt.\n"
        "Do not include markdown fences, explanations, tests, or the original prompt.\n"
        "The completion must be syntactically valid in the target function body.\n"
        "The completion must be self-contained: if it needs a helper function or standard-library import, "
        "define/import it inside the completion.\n"
        "Use the candidate completions as fallible evidence; repair or synthesize when needed.\n\n"
        "HumanEval prompt:\n"
        f"{sample.prompt}\n"
        "Candidate completions:\n"
        f"{exposed_content}\n\n"
        "Final completion:\n"
    )
    return prompt, exposed_content, exposure_metrics


def query_humaneval_initial_agents(
    sample: HumanEvalSample,
    clients: Mapping[str, VLLMClient],
    config: HumanEvalDarkForestConfig,
    temperature: float,
    max_tokens: int,
    seed: int,
    parallel_agents: bool,
) -> Tuple[Dict[str, ParsedHumanEvalOutput], str, float]:
    prompt = build_humaneval_initial_prompt(sample)

    def call(agent: str) -> Tuple[str, ParsedHumanEvalOutput]:
        response = clients[agent].complete(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed)
        parsed = parse_humaneval_agent_output(
            agent,
            response.text,
            prompt=sample.prompt,
            entry_point=sample.entry_point,
            latency_sec=response.latency_sec,
            usage=response.usage,
            error=response.error,
        )
        return agent, parsed

    start = time.perf_counter()
    outputs: Dict[str, ParsedHumanEvalOutput] = {}
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
    elapsed = time.perf_counter() - start
    return {agent: outputs[agent] for agent in config.fixed_agents}, prompt, elapsed


def query_humaneval_coordinator(
    sample: HumanEvalSample,
    clients: Mapping[str, VLLMClient],
    config: HumanEvalDarkForestConfig,
    agent_outputs: Mapping[str, ParsedHumanEvalOutput],
    belief_state: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    seed: int,
) -> Tuple[ParsedHumanEvalOutput, str, str, Dict[str, Any]]:
    prompt, exposed_content, exposure_metrics = build_humaneval_coordinator_prompt(
        sample, agent_outputs, config, belief_state
    )
    response = clients[config.coordinator_model].complete(
        prompt, temperature=temperature, max_tokens=max_tokens, seed=seed
    )
    parsed = parse_humaneval_agent_output(
        "coordinator",
        response.text,
        prompt=sample.prompt,
        entry_point=sample.entry_point,
        latency_sec=response.latency_sec,
        usage=response.usage,
        error=response.error,
    )
    return parsed, prompt, exposed_content, exposure_metrics


def apply_humaneval_guardrail(
    coordinator_output: ParsedHumanEvalOutput,
    belief_state: Dict[str, Any],
    agent_outputs: Mapping[str, ParsedHumanEvalOutput],
    config: HumanEvalDarkForestConfig,
) -> Tuple[str, Dict[str, Any]]:
    selected = coordinator_output.parsed_completion or ""
    report: Dict[str, Any] = {
        "mode": config.belief_guardrail,
        "applied": False,
        "selected_source": "coordinator",
        "reason": "disabled" if config.belief_guardrail == "none" else None,
        "anchor_agent": config.anchor_agent,
        "min_posterior": config.belief_guardrail_min_posterior,
        "min_margin": config.belief_guardrail_min_margin,
        "anchor_fallback_min_posterior": config.anchor_fallback_min_posterior,
    }
    if config.belief_guardrail == "none":
        return selected, report
    if config.belief_guardrail != "coder_anchor_supported_cluster":
        report["reason"] = f"unknown_mode:{config.belief_guardrail}"
        return selected, report

    clusters = belief_state.get("code_clusters") or []
    top = clusters[0] if clusters else {}
    trusted_top = bool(
        top
        and len(top.get("supporting_agents") or []) >= 2
        and float(top.get("posterior") or 0.0) >= config.belief_guardrail_min_posterior
        and float(belief_state.get("posterior_margin") or 0.0) >= config.belief_guardrail_min_margin
    )
    if trusted_top:
        replacement = str(top.get("normalized_completion") or "")
        report.update(
            {
                "applied": replacement != (coordinator_output.normalized_completion or ""),
                "selected_source": "belief_top_cluster",
                "reason": "trusted_supported_top_cluster",
                "supporting_agents": top.get("supporting_agents") or [],
                "posterior": top.get("posterior"),
                "posterior_margin": belief_state.get("posterior_margin"),
            }
        )
        return replacement, report

    anchor = agent_outputs.get(config.anchor_agent)
    anchor_valid = bool(anchor and not anchor.invalid_parse and anchor.parsed_completion)
    coordinator_valid = bool(
        coordinator_output.parsed_completion
        and coordinator_output.normalized_completion
        and not coordinator_output.invalid_parse
    )
    anchor_top = bool(
        anchor_valid
        and top
        and top.get("normalized_completion") == anchor.normalized_completion
        and float(top.get("posterior") or 0.0) >= config.anchor_fallback_min_posterior
        and float(belief_state.get("posterior_margin") or 0.0) >= config.belief_guardrail_min_margin
    )
    if anchor_valid and (not coordinator_valid or anchor_top):
        report.update(
            {
                "applied": anchor.normalized_completion != coordinator_output.normalized_completion,
                "selected_source": config.anchor_agent,
                "reason": (
                    "anchor_agent_overrode_invalid_coordinator"
                    if not coordinator_valid
                    else "high_confidence_anchor_top_cluster"
                ),
                "posterior": top.get("posterior") if top else None,
                "posterior_margin": belief_state.get("posterior_margin"),
            }
        )
        return anchor.parsed_completion, report

    report["reason"] = (
        "valid_coordinator_kept_no_trusted_cluster"
        if coordinator_valid
        else "no_trusted_cluster_or_anchor"
    )
    return selected, report


def run_humaneval_evaluator(
    samples: Sequence[HumanEvalSample],
    completion_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    basename: str,
    n_workers: int = 4,
    timeout: float = 3.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Path, Path]:
    from human_eval.evaluation import evaluate_functional_correctness

    output_dir.mkdir(parents=True, exist_ok=True)
    problem_file = output_dir / f"{basename}_problems.jsonl"
    sample_file = output_dir / f"{basename}_samples.jsonl"
    write_humaneval_problem_file(samples, problem_file)
    with sample_file.open("w", encoding="utf-8") as handle:
        for row in completion_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=False) + "\n")
    result = evaluate_functional_correctness(
        str(sample_file),
        k=[1],
        n_workers=n_workers,
        timeout=timeout,
        problem_file=str(problem_file),
        ignore_incomplete=False,
    )
    result_rows = read_jsonl(Path(str(sample_file) + "_results.jsonl"))
    return {key: float(value) for key, value in result.items()}, result_rows, sample_file, problem_file


def estimate_humaneval_calibration(
    records: Sequence[Dict[str, Any]],
    config: HumanEvalDarkForestConfig,
    seed: int,
    command: str,
    root_dir: str,
) -> Dict[str, Any]:
    agent_stats: Dict[str, Dict[str, Any]] = {
        agent: {"num_samples": 0, "num_valid": 0, "num_passed": 0, "num_invalid_parse": 0}
        for agent in config.fixed_agents
    }
    support_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"num": 0, "num_passed": 0})
    for record in records:
        agent_passed = record.get("agent_passed", {})
        for agent in config.fixed_agents:
            output = record["agents"][agent]
            stats = agent_stats[agent]
            stats["num_samples"] += 1
            if output.get("invalid_parse"):
                stats["num_invalid_parse"] += 1
            else:
                stats["num_valid"] += 1
            if agent_passed.get(agent) is True:
                stats["num_passed"] += 1

        clusters = record.get("darkforest_belief", {}).get("code_clusters") or []
        for cluster in clusters:
            pattern = cluster.get("support_pattern")
            if not pattern:
                continue
            supporters = cluster.get("supporting_agents") or []
            passed = any(agent_passed.get(agent) is True for agent in supporters)
            support_stats[pattern]["num"] += 1
            if passed:
                support_stats[pattern]["num_passed"] += 1

    reliability: Dict[str, Dict[str, Any]] = {}
    learned_priors: Dict[str, float] = {}
    for agent, stats in agent_stats.items():
        num = int(stats["num_samples"])
        passed = int(stats["num_passed"])
        smoothed = (passed + 1.0) / (num + 2.0) if num else 0.5
        learned_priors[agent] = smoothed
        reliability[agent] = {
            **stats,
            "pass_rate": passed / num if num else 0.0,
            "smoothed_pass_rate": smoothed,
            "invalid_parse_rate": stats["num_invalid_parse"] / num if num else 0.0,
        }

    support_reliability: Dict[str, Dict[str, Any]] = {}
    for pattern, stats in support_stats.items():
        num = int(stats["num"])
        passed = int(stats["num_passed"])
        support_reliability[pattern] = {
            "num": num,
            "num_passed": passed,
            "pass_rate": passed / num if num else 0.0,
            "smoothed_accuracy": (passed + 1.0) / (num + 2.0) if num else 0.5,
        }

    return {
        "dataset": "HumanEval",
        "calibration_split": "full_minus_goa_dev_subset",
        "num_calibration_samples": len(records),
        "fixed_agents": list(config.fixed_agents),
        "agent_reliability": reliability,
        "support_pattern_reliability": support_reliability,
        "learned_darkforest_params": {
            "agent_priors": learned_priors,
            "support_pattern_reliability": support_reliability,
            "accept_threshold": config.accept_threshold,
            "uncertainty_threshold": config.uncertainty_threshold,
            "min_support_pattern_count": config.min_support_pattern_count,
            "belief_guardrail": config.belief_guardrail,
            "anchor_agent": config.anchor_agent,
            "belief_guardrail_min_posterior": config.belief_guardrail_min_posterior,
            "belief_guardrail_min_margin": config.belief_guardrail_min_margin,
            "anchor_fallback_min_posterior": config.anchor_fallback_min_posterior,
        },
        "created_at": utc_now_iso(),
        "seed": seed,
        "command": command,
        "root_dir": root_dir,
    }


def apply_humaneval_calibration(
    config: HumanEvalDarkForestConfig,
    calibration: Dict[str, Any],
    source_path: str,
) -> HumanEvalDarkForestConfig:
    learned = calibration.get("learned_darkforest_params") or {}
    if learned.get("agent_priors"):
        config.agent_priors = {agent: float(learned["agent_priors"].get(agent, 1.0)) for agent in config.fixed_agents}
    if learned.get("support_pattern_reliability"):
        config.support_pattern_reliability = dict(learned["support_pattern_reliability"])
    for attr in [
        "accept_threshold",
        "uncertainty_threshold",
        "min_support_pattern_count",
        "belief_guardrail",
        "anchor_agent",
        "belief_guardrail_min_posterior",
        "belief_guardrail_min_margin",
        "anchor_fallback_min_posterior",
    ]:
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


def aggregate_humaneval_summary(
    records: Sequence[Dict[str, Any]],
    mode: str,
    config: HumanEvalDarkForestConfig,
    temperature: float,
    max_tokens: int,
    seed: int,
    calibration_file: Optional[str],
    calibration_num_samples: Optional[int],
    total_wall_time_sec: Optional[float],
    evaluator_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    num_samples = len(records)
    num_passed = sum(1 for record in records if record.get("passed") is True)
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

    pass_at_1 = float((evaluator_result or {}).get("pass@1", num_passed / num_samples if num_samples else 0.0))
    return {
        "dataset": "HumanEval",
        "split": "GoA_dev_subset",
        "mode": mode,
        "num_samples": num_samples,
        "num_passed": num_passed,
        "pass@1": pass_at_1,
        "pass@1_percent": 100.0 * pass_at_1,
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
        "avg_num_agent_outputs_exposed_to_coordinator": avg(
            item.get("num_agent_outputs_exposed_to_coordinator", 0) for item in exposure
        ),
        "pass_per_1k_tokens": num_passed / (sum(total_tokens) / 1000.0) if sum(total_tokens) else 0.0,
        "pass_per_second": num_passed / sum(latencies) if sum(latencies) else 0.0,
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
        "evaluator": "openai_humaneval.evaluate_functional_correctness",
        "evaluator_result": evaluator_result or {},
    }


def write_records_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n")


def make_sample_metrics(
    passed: bool,
    invalid_parse: bool,
    agent_outputs: Mapping[str, ParsedHumanEvalOutput],
    coordinator_output: Optional[ParsedHumanEvalOutput],
    initial_latency_sec: float,
    exposure_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return build_sample_metrics(
        correct=passed,
        invalid_parse=invalid_parse,
        initial_usages=[agent_outputs[agent].usage for agent in HUMANEVAL_AGENTS if agent in agent_outputs],
        coordination_usage=coordinator_output.usage if coordinator_output is not None else None,
        initial_latency_sec=initial_latency_sec,
        coordination_latency_sec=coordinator_output.latency_sec if coordinator_output is not None else 0.0,
        individual_initial_latencies={
            agent: agent_outputs[agent].latency_sec for agent in HUMANEVAL_AGENTS if agent in agent_outputs
        },
        exposure_metrics=exposure_metrics,
        coordination_rounds=1 if coordinator_output is not None else 0,
        scored=True,
    )
