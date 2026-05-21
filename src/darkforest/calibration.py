from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .bayes import compute_darkforest_belief
from .parsing import math_answers_match, normalize_math_answer
from .schemas import FIXED_AGENTS, DarkForestConfig
from .utils import command_line, read_json, utc_now_iso, write_json


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _agent_stats() -> Dict[str, Any]:
    return {
        "num_samples": 0,
        "num_valid": 0,
        "num_correct": 0,
        "num_invalid_parse": 0,
        "num_malformed_json": 0,
    }


def _support_stats() -> Dict[str, Any]:
    return {"num": 0, "num_correct": 0}


def _finalize_agent_stats(stats: Dict[str, Any], alpha: float = 1.0, beta: float = 1.0) -> Dict[str, Any]:
    num_samples = int(stats.get("num_samples", 0))
    num_valid = int(stats.get("num_valid", 0))
    num_correct = int(stats.get("num_correct", 0))
    num_invalid = int(stats.get("num_invalid_parse", 0))
    num_malformed = int(stats.get("num_malformed_json", 0))
    accuracy = num_correct / num_valid if num_valid else 0.0
    smoothed = (num_correct + alpha) / (num_valid + alpha + beta) if num_valid else 0.5
    return {
        "num_samples": num_samples,
        "num_valid": num_valid,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "smoothed_accuracy": smoothed,
        "invalid_parse_rate": num_invalid / num_samples if num_samples else 0.0,
        "malformed_json_rate": num_malformed / num_samples if num_samples else 0.0,
    }


def _finalize_support(stats: Dict[str, Any], alpha: float = 1.0, beta: float = 1.0) -> Dict[str, Any]:
    num = int(stats.get("num", 0))
    correct = int(stats.get("num_correct", 0))
    accuracy = correct / num if num else 0.0
    smoothed = (correct + alpha) / (num + alpha + beta) if num else alpha / (alpha + beta)
    return {
        "num": num,
        "num_correct": correct,
        "accuracy": accuracy,
        "smoothed_accuracy": smoothed,
    }


def _record_gold(record: Mapping[str, Any]) -> Optional[str]:
    return record.get("normalized_gold_answer") or normalize_math_answer(record.get("gold_answer"))


def _agent_outputs(
    record: Mapping[str, Any],
    fixed_agents: Iterable[str] = FIXED_AGENTS,
) -> Dict[str, Dict[str, Any]]:
    return {agent: dict(record.get("agents", {}).get(agent, {})) for agent in fixed_agents}


def _support_patterns_for_record(
    record: Mapping[str, Any],
    fixed_agents: Iterable[str] = FIXED_AGENTS,
    answer_match_backend: str = "exact",
) -> List[Dict[str, Any]]:
    fixed_agents = list(fixed_agents)
    normalized_gold = _record_gold(record)
    scored = bool(normalized_gold)
    clusters: Dict[str, List[str]] = defaultdict(list)
    raw_answers: Dict[str, List[str]] = defaultdict(list)
    for agent, output in _agent_outputs(record, fixed_agents).items():
        if output.get("invalid_parse") or not output.get("normalized_answer"):
            continue
        normalized = str(output.get("normalized_answer"))
        clusters[normalized].append(agent)
        raw_answers[normalized].append(output.get("parsed_answer"))
    rows = []
    for normalized_answer, agents in clusters.items():
        agents = sorted(agents, key=fixed_agents.index)
        pattern = "+".join(agents)
        rows.append(
            {
                "normalized_answer": normalized_answer,
                "raw_answers": raw_answers[normalized_answer],
                "supporting_agents": agents,
                "support_pattern": pattern,
                "cluster_correct": (
                    math_answers_match(normalized_answer, normalized_gold, answer_match_backend)
                    if scored
                    else None
                ),
            }
        )
    return rows


def estimate_calibration_from_records(
    records: Iterable[Mapping[str, Any]],
    config: DarkForestConfig,
    *,
    dataset: str = "MATH",
    calibration_split: str = "train",
    seed: int = 0,
    root_dir: Optional[str] = None,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    records = list(records)
    fixed_agents = list(config.fixed_agents)
    agent_stats = {agent: _agent_stats() for agent in fixed_agents}
    by_subject: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: {agent: _agent_stats() for agent in fixed_agents}
    )
    by_level: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: {agent: _agent_stats() for agent in fixed_agents}
    )
    support_stats: Dict[str, Dict[str, Any]] = defaultdict(_support_stats)
    confidence_bins = [
        {"bin_start": i / 10.0, "bin_end": (i + 1) / 10.0, "num": 0, "num_correct": 0}
        for i in range(10)
    ]
    missing_conf = {"num": 0, "num_correct": 0}
    malformed_total = {"num": 0, "num_correct": 0}
    wellformed_total = {"num": 0, "num_correct": 0}
    invalid_parse_count = 0
    unscored_records = 0

    mathstral_pair_valid = 0
    mathstral_pair_same = 0
    mathstral_pair_same_correct = 0
    qwen_mathstral_same = 0
    qwen_mathstral_same_correct = 0

    for record in records:
        normalized_gold = _record_gold(record)
        scored = bool(normalized_gold)
        if not scored:
            unscored_records += 1
            continue
        metadata = record.get("metadata") or {}
        subject = str(metadata.get("subject") or metadata.get("type") or "unknown")
        level = str(metadata.get("level") or "unknown")
        outputs = _agent_outputs(record, fixed_agents)

        for agent in fixed_agents:
            output = outputs[agent]
            for stats in (agent_stats[agent], by_subject[subject][agent], by_level[level][agent]):
                stats["num_samples"] += 1
                if output.get("invalid_parse"):
                    stats["num_invalid_parse"] += 1
                else:
                    stats["num_valid"] += 1
                if output.get("malformed_json"):
                    stats["num_malformed_json"] += 1
            correct = (
                math_answers_match(
                    output.get("parsed_answer") or output.get("normalized_answer"),
                    normalized_gold,
                    config.answer_match_backend,
                )
                if normalized_gold
                else False
            )
            if correct:
                for stats in (agent_stats[agent], by_subject[subject][agent], by_level[level][agent]):
                    stats["num_correct"] += 1

            if output.get("invalid_parse"):
                invalid_parse_count += 1
            bucket = malformed_total if output.get("malformed_json") else wellformed_total
            bucket["num"] += 1
            if correct:
                bucket["num_correct"] += 1

            confidence = output.get("confidence")
            if confidence is None:
                missing_conf["num"] += 1
                if correct:
                    missing_conf["num_correct"] += 1
            else:
                conf = _clip(float(confidence), 0.0, 1.0)
                idx = min(9, int(conf * 10))
                confidence_bins[idx]["num"] += 1
                if correct:
                    confidence_bins[idx]["num_correct"] += 1

        for cluster in _support_patterns_for_record(record, fixed_agents, config.answer_match_backend):
            stats = support_stats[cluster["support_pattern"]]
            stats["num"] += 1
            if cluster["cluster_correct"]:
                stats["num_correct"] += 1

        if "mathstral_1" in fixed_agents and "mathstral_2" in fixed_agents:
            m1 = outputs["mathstral_1"]
            m2 = outputs["mathstral_2"]
            if (
                not m1.get("invalid_parse")
                and not m2.get("invalid_parse")
                and m1.get("normalized_answer")
                and m2.get("normalized_answer")
            ):
                mathstral_pair_valid += 1
                if m1.get("normalized_answer") == m2.get("normalized_answer"):
                    mathstral_pair_same += 1
                    if normalized_gold and math_answers_match(
                        m1.get("parsed_answer") or m1.get("normalized_answer"),
                        normalized_gold,
                        config.answer_match_backend,
                    ):
                        mathstral_pair_same_correct += 1

        if "qwen" in fixed_agents:
            qwen = outputs["qwen"]
            for other_key in [agent for agent in fixed_agents if agent != "qwen"]:
                other = outputs[other_key]
                if (
                    not qwen.get("invalid_parse")
                    and not other.get("invalid_parse")
                    and qwen.get("normalized_answer")
                    and qwen.get("normalized_answer") == other.get("normalized_answer")
                ):
                    qwen_mathstral_same += 1
                    if normalized_gold and math_answers_match(
                        qwen.get("parsed_answer") or qwen.get("normalized_answer"),
                        normalized_gold,
                        config.answer_match_backend,
                    ):
                        qwen_mathstral_same_correct += 1

    finalized_agents = {agent: _finalize_agent_stats(agent_stats[agent]) for agent in fixed_agents}
    finalized_subject = {
        subject: {agent: _finalize_agent_stats(stats[agent]) for agent in fixed_agents}
        for subject, stats in by_subject.items()
    }
    finalized_level = {
        level: {agent: _finalize_agent_stats(stats[agent]) for agent in fixed_agents}
        for level, stats in by_level.items()
    }

    all_patterns = [
        "+".join(combo)
        for size in range(1, len(fixed_agents) + 1)
        for combo in combinations(fixed_agents, size)
    ]
    finalized_support = {
        pattern: _finalize_support(support_stats.get(pattern, _support_stats()))
        for pattern in all_patterns
    }
    for pattern, stats in sorted(support_stats.items()):
        finalized_support[pattern] = _finalize_support(stats)

    bins_payload = []
    for item in confidence_bins:
        empirical = item["num_correct"] / item["num"] if item["num"] else None
        bins_payload.append(
            {
                "bin_start": item["bin_start"],
                "bin_end": item["bin_end"],
                "num": item["num"],
                "empirical_accuracy": empirical,
            }
        )
    missing_empirical = (
        missing_conf["num_correct"] / missing_conf["num"] if missing_conf["num"] else None
    )
    missing_default = _clip(missing_empirical if missing_empirical is not None else 0.5, 0.1, 0.9)

    malformed_acc = (
        malformed_total["num_correct"] / malformed_total["num"] if malformed_total["num"] else None
    )
    wellformed_acc = (
        wellformed_total["num_correct"] / wellformed_total["num"] if wellformed_total["num"] else None
    )
    if malformed_acc is None or wellformed_acc is None or wellformed_acc <= 0:
        malformed_penalty = 0.5
    else:
        malformed_penalty = _clip(malformed_acc / wellformed_acc, 0.1, 1.0)

    if "mathstral_1" in fixed_agents and "mathstral_2" in fixed_agents:
        base = max(
            finalized_agents["mathstral_1"]["smoothed_accuracy"],
            finalized_agents["mathstral_2"]["smoothed_accuracy"],
        )
        both_entry = finalized_support.get("mathstral_1+mathstral_2", _finalize_support(_support_stats()))
    else:
        base = None
        both_entry = _finalize_support(_support_stats())
    if base is not None and both_entry["num"] >= config.min_support_pattern_count and base < 1.0:
        incremental = (both_entry["smoothed_accuracy"] - base) / max(1e-6, 1.0 - base)
        same_model_discount = _clip(incremental, 0.2, 1.0)
    else:
        same_model_discount = config.same_model_correlation_discount

    learned_params = {
        "agent_priors": {
            agent: finalized_agents[agent]["smoothed_accuracy"] for agent in fixed_agents
        },
        "same_model_correlation_discount": same_model_discount,
        "missing_confidence_default": missing_default,
        "malformed_output_penalty": malformed_penalty,
        "accept_threshold": config.accept_threshold,
        "uncertainty_threshold": config.uncertainty_threshold,
        "support_pattern_reliability": finalized_support,
    }

    return {
        "dataset": dataset,
        "calibration_split": calibration_split,
        "num_calibration_samples": len(records),
        "num_scored_calibration_samples": len(records) - unscored_records,
        "num_unscored_calibration_samples": unscored_records,
        "answer_match_backend": config.answer_match_backend,
        "fixed_agents": list(fixed_agents),
        "agent_reliability": finalized_agents,
        "agent_reliability_by_subject": finalized_subject,
        "agent_reliability_by_level": finalized_level,
        "support_pattern_reliability": finalized_support,
        "confidence_calibration": {
            "bins": bins_payload,
            "missing_confidence": {
                "num": missing_conf["num"],
                "empirical_accuracy": missing_empirical,
            },
        },
        "malformed_output_stats": {
            "num_malformed_json": malformed_total["num"],
            "num_invalid_parse": invalid_parse_count,
            "malformed_json_accuracy": malformed_acc,
            "wellformed_json_accuracy": wellformed_acc,
            "invalid_parse_rate": invalid_parse_count / (len(records) * len(fixed_agents)) if records else 0.0,
            "malformed_output_penalty": malformed_penalty,
        },
        "correlation_stats": {
            "same_model_pair": (
                ["mathstral_1", "mathstral_2"]
                if "mathstral_1" in fixed_agents and "mathstral_2" in fixed_agents
                else None
            ),
            "mathstral_pair_same_answer_rate": (
                mathstral_pair_same / mathstral_pair_valid if mathstral_pair_valid else None
            ),
            "mathstral_pair_same_answer_correct_rate": (
                mathstral_pair_same_correct / mathstral_pair_same if mathstral_pair_same else None
            ),
            "mathstral_1_accuracy": (
                finalized_agents["mathstral_1"]["accuracy"] if "mathstral_1" in finalized_agents else None
            ),
            "mathstral_2_accuracy": (
                finalized_agents["mathstral_2"]["accuracy"] if "mathstral_2" in finalized_agents else None
            ),
            "qwen_mathstral_same_answer_correct_rate": (
                qwen_mathstral_same_correct / qwen_mathstral_same if qwen_mathstral_same else None
            ),
            "estimated_same_model_correlation_discount": same_model_discount,
        },
        "learned_darkforest_params": learned_params,
        "created_at": utc_now_iso(),
        "seed": seed,
        "command": command if command is not None else command_line(),
        "root_dir": root_dir,
    }


def run_calibration(
    samples: Iterable[Any],
    agent_outputs: Iterable[Mapping[str, Any]],
    config: DarkForestConfig,
) -> Dict[str, Any]:
    records = []
    for sample, outputs in zip(samples, agent_outputs):
        record = {
            "idx": getattr(sample, "idx", None),
            "question": getattr(sample, "question", None),
            "gold_answer": getattr(sample, "gold_answer", None),
            "normalized_gold_answer": normalize_math_answer(getattr(sample, "gold_answer", None)),
            "metadata": getattr(sample, "metadata", {}),
            "agents": outputs,
        }
        record["support_patterns"] = _support_patterns_for_record(record, config.fixed_agents)
        record["darkforest_belief"] = compute_darkforest_belief(outputs, config)
        records.append(record)
    return estimate_calibration_from_records(records, config)


def save_calibration(calibration_dict: Dict[str, Any], path: str | Path) -> None:
    write_json(Path(path), calibration_dict)


def load_calibration(path: str | Path) -> Dict[str, Any]:
    return read_json(Path(path))


def apply_calibration_to_config(
    config: DarkForestConfig,
    calibration_dict: Dict[str, Any],
) -> DarkForestConfig:
    calibration_agents = calibration_dict.get("fixed_agents")
    if calibration_agents and list(calibration_agents) != list(config.fixed_agents):
        raise ValueError(
            "Calibration fixed_agents do not match current config: "
            f"calibration={calibration_agents}, current={config.fixed_agents}"
        )
    params = calibration_dict.get("learned_darkforest_params", {})
    if "agent_priors" in params:
        config.agent_priors = {
            agent: float(params["agent_priors"].get(agent, 1.0))
            for agent in config.fixed_agents
        }
        config.parameter_sources["agent_priors"] = "calibrated"
    if "support_pattern_reliability" in params:
        config.support_pattern_reliability = params["support_pattern_reliability"]
        config.parameter_sources["support_pattern_reliability"] = "calibrated"
    for attr in (
        "same_model_correlation_discount",
        "missing_confidence_default",
        "malformed_output_penalty",
        "accept_threshold",
        "uncertainty_threshold",
    ):
        if attr in params:
            setattr(config, attr, float(params[attr]))
            config.parameter_sources[attr] = "calibrated"
    config.confidence_calibration = calibration_dict.get("confidence_calibration", {})
    config.calibration_source = "calibrated"
    config.params_source = "calibrated"
    return config
