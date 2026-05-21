from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schemas import DarkForestConfig
from .utils import median, percentile


def llm_call_count_for_method(method: str) -> int:
    if method in {"single_qwen", "single_mathstral_1", "single_mathstral_2"}:
        return 1
    if method == "majority_vote":
        return 3
    return 4


def build_sample_metrics(
    correct: bool,
    invalid_parse: bool,
    initial_usages: List[Dict[str, Any]],
    coordination_usage: Optional[Dict[str, Any]],
    initial_latency_sec: float,
    coordination_latency_sec: float,
    individual_initial_latencies: Optional[Dict[str, float]],
    exposure_metrics: Dict[str, Any],
    coordination_rounds: int = 1,
    verification_usage: Optional[Dict[str, Any]] = None,
    verification_latency_sec: float = 0.0,
    scored: bool = True,
) -> Dict[str, Any]:
    has_coordination = coordination_usage is not None
    verification_usage = verification_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    coordination_usage = coordination_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    initial_input = sum(int(usage.get("input_tokens", 0) or 0) for usage in initial_usages)
    initial_output = sum(int(usage.get("output_tokens", 0) or 0) for usage in initial_usages)
    initial_total = sum(
        int(usage.get("total_tokens", (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)) or 0)
        for usage in initial_usages
    )
    verification_input = int(verification_usage.get("input_tokens", 0) or 0)
    verification_output = int(verification_usage.get("output_tokens", 0) or 0)
    verification_total = int(
        verification_usage.get("total_tokens", verification_input + verification_output) or 0
    )
    coordination_input = int(coordination_usage.get("input_tokens", 0) or 0)
    coordination_output = int(coordination_usage.get("output_tokens", 0) or 0)
    coordination_total = int(
        coordination_usage.get("total_tokens", coordination_input + coordination_output) or 0
    )
    llm_calls_by_phase = {
        "initial_agents": len(initial_usages),
        "verification": 0,
        "coordination": 1 if coordination_rounds == 1 and has_coordination else 0,
    }
    total_latency = initial_latency_sec + verification_latency_sec + coordination_latency_sec
    return {
        "correct": correct,
        "scored": scored,
        "invalid_parse": invalid_parse,
        "llm_calls_total": sum(llm_calls_by_phase.values()),
        "llm_calls_by_phase": llm_calls_by_phase,
        "input_tokens_total": initial_input + verification_input + coordination_input,
        "output_tokens_total": initial_output + verification_output + coordination_output,
        "total_tokens": initial_total + verification_total + coordination_total,
        "tokens_by_phase": {
            "initial_agents_input": initial_input,
            "initial_agents_output": initial_output,
            "verification_input": verification_input,
            "verification_output": verification_output,
            "coordination_input": coordination_input,
            "coordination_output": coordination_output,
        },
        "latency_sec_total": total_latency,
        "latency_by_phase_sec": {
            "initial_agents": initial_latency_sec,
            "verification": verification_latency_sec,
            "coordination": coordination_latency_sec,
        },
        "individual_initial_agent_latency_sec": individual_initial_latencies or {},
        "exposure_metrics": exposure_metrics,
    }


def aggregate_evaluation_summary(
    records: List[Dict[str, Any]],
    split: str,
    mode: str,
    config: DarkForestConfig,
    temperature: float,
    max_tokens: int,
    seed: int,
    calibration_file: Optional[str],
    calibration_num_samples: Optional[int],
    total_wall_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    scored_records = [record for record in records if record.get("scored", True) is not False]
    num_records_processed = len(records)
    num_unscored = num_records_processed - len(scored_records)
    num_samples = len(scored_records)
    num_correct = sum(1 for record in scored_records if record.get("correct") is True)
    num_invalid = sum(1 for record in scored_records if record.get("invalid_parse") is True)
    metrics = [record.get("metrics", {}) for record in scored_records]
    input_tokens = [int(item.get("input_tokens_total", 0) or 0) for item in metrics]
    output_tokens = [int(item.get("output_tokens_total", 0) or 0) for item in metrics]
    total_tokens = [int(item.get("total_tokens", 0) or 0) for item in metrics]
    latencies = [float(item.get("latency_sec_total", 0.0) or 0.0) for item in metrics]
    initial_latencies = [
        float(item.get("latency_by_phase_sec", {}).get("initial_agents", 0.0) or 0.0)
        for item in metrics
    ]
    verification_latencies = [
        float(item.get("latency_by_phase_sec", {}).get("verification", 0.0) or 0.0)
        for item in metrics
    ]
    coordination_latencies = [
        float(item.get("latency_by_phase_sec", {}).get("coordination", 0.0) or 0.0)
        for item in metrics
    ]
    exposure = [item.get("exposure_metrics", {}) for item in metrics]
    total_tokens_sum = sum(total_tokens)
    total_latency_sum = sum(latencies)
    wall = total_wall_time_sec if total_wall_time_sec is not None else total_latency_sum

    def avg(values: List[float | int]) -> float:
        return float(sum(values) / num_samples) if num_samples else 0.0

    return {
        "dataset": "MATH",
        "split": split,
        "mode": mode,
        "num_samples": num_samples,
        "num_records_processed": num_records_processed,
        "num_unscored": num_unscored,
        "num_correct": num_correct,
        "em_percent": 100.0 * num_correct / num_samples if num_samples else 0.0,
        "num_invalid_parse": num_invalid,
        "invalid_parse_rate_percent": 100.0 * num_invalid / num_samples if num_samples else 0.0,
        "avg_llm_calls_per_sample": avg([item.get("llm_calls_total", 0) for item in metrics]),
        "avg_input_tokens_per_sample": avg(input_tokens),
        "avg_output_tokens_per_sample": avg(output_tokens),
        "avg_total_tokens_per_sample": avg(total_tokens),
        "avg_latency_sec_per_sample": avg(latencies),
        "median_latency_sec_per_sample": median(latencies),
        "p90_latency_sec_per_sample": percentile(latencies, 90),
        "p95_latency_sec_per_sample": percentile(latencies, 95),
        "samples_per_second": num_samples / wall if wall and wall > 0 else 0.0,
        "avg_initial_agent_latency_sec": avg(initial_latencies),
        "avg_verification_latency_sec": avg(verification_latencies),
        "avg_coordination_latency_sec": avg(coordination_latencies),
        "avg_cross_agent_input_tokens": avg(
            [item.get("cross_agent_input_tokens", 0) for item in exposure]
        ),
        "avg_cross_agent_input_chars": avg(
            [item.get("cross_agent_input_chars", 0) for item in exposure]
        ),
        "avg_num_agent_outputs_exposed_to_coordinator": avg(
            [item.get("num_agent_outputs_exposed_to_coordinator", 0) for item in exposure]
        ),
        "em_per_1k_tokens": num_correct / (total_tokens_sum / 1000.0) if total_tokens_sum else 0.0,
        "em_per_second": num_correct / total_latency_sum if total_latency_sum else 0.0,
        "coordination_method": config.coordination_method,
        "design_name": config.design_name,
        "coordinator_model": config.coordinator_model,
        "fixed_agents": list(config.fixed_agents),
        "answer_match_backend": config.answer_match_backend,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "calibration_file": calibration_file,
        "freeze_calibration": config.freeze_calibration,
        "calibration_num_samples": calibration_num_samples,
        "darkforest_params_source": config.params_source,
        "darkforest_config": config.darkforest_summary_dict(),
        "learned_darkforest_params_snapshot": {
            "agent_priors": dict(config.agent_priors),
            "same_model_correlation_discount": config.same_model_correlation_discount,
            "missing_confidence_default": config.missing_confidence_default,
            "malformed_output_penalty": config.malformed_output_penalty,
            "accept_threshold": config.accept_threshold,
            "uncertainty_threshold": config.uncertainty_threshold,
            "min_support_pattern_count": config.min_support_pattern_count,
            "belief_guardrail": config.belief_guardrail,
            "belief_guardrail_anchor_agent": config.belief_guardrail_anchor_agent,
            "belief_guardrail_min_posterior": config.belief_guardrail_min_posterior,
            "belief_guardrail_min_margin": config.belief_guardrail_min_margin,
            "support_pattern_reliability": config.support_pattern_reliability,
            "confidence_calibration": config.confidence_calibration,
            "parameter_sources": dict(config.parameter_sources),
            "params_source": config.params_source,
            "calibration_source": config.calibration_source,
        },
    }
