from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .bayes import compute_darkforest_belief
from .calibration import estimate_calibration_from_records, save_calibration
from .coordination import (
    build_initial_agent_prompt,
    build_static_coordinator_prompt_with_exposure,
    prompt_for_agent,
)
from .guardrail import apply_belief_guardrail
from .metrics import aggregate_evaluation_summary, build_sample_metrics
from .parsing import extract_final_answer, math_answers_match, normalize_math_answer, parse_agent_response
from .schemas import FIXED_AGENTS, DarkForestConfig, LLMResponse, MathSample, ParsedAgentOutput
from .utils import append_jsonl, completed_indices, read_jsonl, write_json


def _as_dict_outputs(
    outputs: Mapping[str, ParsedAgentOutput],
    fixed_agents: Iterable[str] = FIXED_AGENTS,
) -> Dict[str, Dict[str, Any]]:
    return {agent: outputs[agent].to_dict() for agent in fixed_agents}


def _call_client(
    client: Any,
    prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> LLMResponse:
    return client.complete(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )


def query_initial_agents(
    sample: MathSample,
    clients: Mapping[str, Any],
    config: DarkForestConfig,
    *,
    temperature: float,
    max_tokens: int,
    seed: int,
    parallel_agents: bool,
    fixed_agents: Iterable[str] = FIXED_AGENTS,
) -> Tuple[Dict[str, ParsedAgentOutput], str, Dict[str, str], float]:
    base_prompt = build_initial_agent_prompt(sample.question, config.initial_prompt_style)
    fixed_agents = list(fixed_agents)
    agent_prompts = {agent: prompt_for_agent(base_prompt, agent, config) for agent in fixed_agents}

    def run_one(agent_key: str) -> ParsedAgentOutput:
        response = _call_client(clients[agent_key], agent_prompts[agent_key], temperature, max_tokens, seed)
        return parse_agent_response(
            agent_key=agent_key,
            raw_response=response.text,
            latency_sec=response.latency_sec,
            usage=response.usage,
            error=response.error,
        )

    start = time.perf_counter()
    outputs: Dict[str, ParsedAgentOutput] = {}
    if parallel_agents:
        with ThreadPoolExecutor(max_workers=len(fixed_agents)) as executor:
            futures = {executor.submit(run_one, agent): agent for agent in fixed_agents}
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    outputs[agent] = future.result()
                except Exception as exc:  # noqa: BLE001 - record per-agent failure
                    outputs[agent] = parse_agent_response(
                        agent,
                        "",
                        latency_sec=0.0,
                        usage={},
                        error=str(exc),
                    )
    else:
        for agent in fixed_agents:
            outputs[agent] = run_one(agent)
    phase_latency = time.perf_counter() - start
    return outputs, base_prompt, agent_prompts, phase_latency


def _prepare_jsonl(path: Path, resume: bool, overwrite: bool) -> List[Dict[str, Any]]:
    if path.exists() and overwrite:
        path.write_text("", encoding="utf-8")
        return []
    if path.exists() and resume:
        return read_jsonl(path)
    if path.exists() and not resume:
        raise FileExistsError(f"Output file exists; pass --resume true or --overwrite true: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return []


def _sample_common_record(sample: MathSample, initial_prompt: str) -> Dict[str, Any]:
    normalized_gold = normalize_math_answer(sample.gold_answer) if sample.gold_answer else None
    return {
        "idx": sample.idx,
        "split": sample.metadata.get("split"),
        "question": sample.question,
        "instruction": initial_prompt,
        "gold_answer": sample.gold_answer,
        "normalized_gold_answer": normalized_gold,
        "metadata": sample.metadata,
    }


def _clusters_for_calibration_record(
    record: Dict[str, Any],
    fixed_agents: Iterable[str] = FIXED_AGENTS,
    config: Optional[DarkForestConfig] = None,
) -> List[Dict[str, Any]]:
    fixed_agents = list(fixed_agents)
    has_gold = bool(record.get("normalized_gold_answer"))
    backend = config.answer_match_backend if config is not None else "exact"
    clusters: Dict[str, List[str]] = {}
    raw_answers: Dict[str, List[str]] = {}
    for agent in fixed_agents:
        output = record["agents"][agent]
        if output.get("invalid_parse") or not output.get("normalized_answer"):
            continue
        answer = output["normalized_answer"]
        clusters.setdefault(answer, []).append(agent)
        raw_answers.setdefault(answer, []).append(output.get("parsed_answer"))
    rows = []
    for answer, agents in clusters.items():
        agents = sorted(agents, key=fixed_agents.index)
        rows.append(
            {
                "normalized_answer": answer,
                "raw_answers": raw_answers[answer],
                "supporting_agents": agents,
                "support_pattern": "+".join(agents),
                "cluster_correct": (
                    math_answers_match(answer, record["normalized_gold_answer"], backend)
                    if has_gold
                    else None
                ),
            }
        )
    return rows


def run_calibration_samples(
    samples: Iterable[MathSample],
    clients: Mapping[str, Any],
    config: DarkForestConfig,
    *,
    output_file: Path,
    calibration_output: Path,
    temperature: float,
    max_tokens: int,
    seed: int,
    parallel_agents: bool,
    resume: bool,
    overwrite: bool,
    save_prompts: bool,
    calibration_query_coordinator: bool = False,
    root_dir: Optional[str] = None,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    existing_records = _prepare_jsonl(output_file, resume, overwrite)
    done = completed_indices(output_file) if resume else set()
    records = list(existing_records)
    fixed_agents = list(config.fixed_agents)

    for sample in samples:
        if sample.idx in done:
            continue
        agent_outputs, initial_prompt, agent_prompts, initial_phase_latency = query_initial_agents(
            sample,
            clients,
            config,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            parallel_agents=parallel_agents,
            fixed_agents=fixed_agents,
        )
        agent_dict = _as_dict_outputs(agent_outputs, fixed_agents)
        record = _sample_common_record(sample, initial_prompt)
        scored = bool(record["normalized_gold_answer"])
        record["scored"] = scored
        record["agents"] = agent_dict
        record["agent_correct"] = {
            agent: (
                math_answers_match(
                    agent_dict[agent].get("parsed_answer") or agent_dict[agent].get("normalized_answer"),
                    record.get("gold_answer") or record["normalized_gold_answer"],
                    config.answer_match_backend,
                )
                if scored
                else None
            )
            for agent in fixed_agents
        }
        record["support_patterns"] = _clusters_for_calibration_record(record, fixed_agents, config)
        record["darkforest_belief"] = compute_darkforest_belief(agent_outputs, config)
        initial_usages = [agent_outputs[agent].usage for agent in fixed_agents]
        individual_latencies = {agent: agent_outputs[agent].latency_sec for agent in fixed_agents}
        record["metrics"] = build_sample_metrics(
            correct=any(record["agent_correct"].values()),
            invalid_parse=all(agent_dict[agent].get("invalid_parse") for agent in fixed_agents),
            initial_usages=initial_usages,
            coordination_usage=None,
            initial_latency_sec=initial_phase_latency,
            coordination_latency_sec=0.0,
            individual_initial_latencies=individual_latencies,
            exposure_metrics={
                "num_agents": len(fixed_agents),
                "num_agent_outputs_exposed_to_coordinator": 0,
                "cross_agent_input_chars": 0,
                "cross_agent_input_tokens": 0,
                "raw_full_response_exposed": False,
                "reasoning_exposed": False,
                "confidence_exposed": False,
                "belief_summary_exposed": False,
            },
            scored=scored,
        )
        if save_prompts:
            record["prompts"] = {"initial_agents": agent_prompts}

        if calibration_query_coordinator:
            belief = record["darkforest_belief"]
            raw_prompt, exposed, exposure_metrics = build_static_coordinator_prompt_with_exposure(
                sample.question,
                agent_outputs,
                config,
                belief,
            )
            coordinator = config.coordinator_model
            prompt = prompt_for_agent(raw_prompt, coordinator, config)
            response = _call_client(clients[coordinator], prompt, temperature, max_tokens, seed)
            final = extract_final_answer(response.text)
            record["coordination"] = {
                "method": config.coordination_method,
                "design_name": config.design_name,
                "coordinator_model": coordinator,
                "coordination_rounds": config.coordination_rounds,
                "fixed_agents": list(config.fixed_agents),
                "exposure_policy": config.exposure_policy.to_dict(),
                "darkforest_belief": belief,
                "coordinator_prompt": prompt,
                "exposed_content": exposed,
                "coordinator_response": response.text,
                "coordinator_error": response.error,
                "final_parse": final,
            }
            record["metrics"] = build_sample_metrics(
                correct=(
                    math_answers_match(
                        final.get("parsed_answer") or final.get("normalized_answer"),
                        record.get("gold_answer") or record["normalized_gold_answer"],
                        config.answer_match_backend,
                    )
                    if scored
                    else None
                ),
                invalid_parse=bool(final.get("invalid_parse")),
                initial_usages=initial_usages,
                coordination_usage=response.usage,
                initial_latency_sec=initial_phase_latency,
                coordination_latency_sec=response.latency_sec,
                individual_initial_latencies=individual_latencies,
                exposure_metrics=exposure_metrics,
            )
            if save_prompts:
                record["prompts"]["coordinator"] = prompt

        append_jsonl(output_file, record)
        records.append(record)

    calibration = estimate_calibration_from_records(
        records,
        config,
        dataset="MATH",
        calibration_split="train",
        seed=seed,
        root_dir=root_dir,
        command=command,
    )
    save_calibration(calibration, calibration_output)
    return calibration


def run_evaluation_samples(
    samples: Iterable[MathSample],
    clients: Mapping[str, Any],
    config: DarkForestConfig,
    *,
    output_file: Path,
    summary_file: Path,
    temperature: float,
    max_tokens: int,
    seed: int,
    parallel_agents: bool,
    resume: bool,
    overwrite: bool,
    save_prompts: bool,
    split: str,
    mode: str,
    calibration_file: Optional[str],
    calibration_num_samples: Optional[int],
) -> Dict[str, Any]:
    if config.coordination_rounds != 1:
        raise NotImplementedError("DarkForest currently implements exactly --coordination_rounds 1")
    if config.coordinator_model not in clients:
        raise ValueError(f"Unknown coordinator_model: {config.coordinator_model}")

    existing_records = _prepare_jsonl(output_file, resume, overwrite)
    done = completed_indices(output_file) if resume else set()
    records = list(existing_records)
    start_wall = time.perf_counter()
    fixed_agents = list(config.fixed_agents)

    for sample in samples:
        if sample.idx in done:
            continue
        agent_outputs, initial_prompt, agent_prompts, initial_phase_latency = query_initial_agents(
            sample,
            clients,
            config,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            parallel_agents=parallel_agents,
            fixed_agents=fixed_agents,
        )
        agent_dict = _as_dict_outputs(agent_outputs, fixed_agents)
        belief = compute_darkforest_belief(agent_outputs, config)
        raw_coordinator_prompt, exposed_content, exposure_metrics = build_static_coordinator_prompt_with_exposure(
            sample.question,
            agent_outputs,
            config,
            belief,
        )
        coordinator_prompt = prompt_for_agent(raw_coordinator_prompt, config.coordinator_model, config)
        coordinator_response = _call_client(
            clients[config.coordinator_model],
            coordinator_prompt,
            temperature,
            max_tokens,
            seed,
        )
        final_parse = extract_final_answer(coordinator_response.text)
        guarded_final_parse, guardrail_report = apply_belief_guardrail(
            final_parse,
            belief,
            config,
            agent_outputs,
        )
        record = _sample_common_record(sample, initial_prompt)
        scored = bool(record["normalized_gold_answer"])
        correct = (
            math_answers_match(
                guarded_final_parse.get("parsed_answer") or guarded_final_parse.get("normalized_answer"),
                record.get("gold_answer") or record["normalized_gold_answer"],
                config.answer_match_backend,
            )
            if scored
            else None
        )
        record.update(
            {
                "final_response": coordinator_response.text,
                "parsed_answer": guarded_final_parse.get("parsed_answer"),
                "normalized_parsed_answer": guarded_final_parse.get("normalized_answer"),
                "correct": correct,
                "answer_match_backend": config.answer_match_backend,
                "scored": scored,
                "invalid_parse": bool(guarded_final_parse.get("invalid_parse")),
                "agents": agent_dict,
                "coordination": {
                    "method": config.coordination_method,
                    "design_name": config.design_name,
                    "coordinator_model": config.coordinator_model,
                    "coordination_rounds": config.coordination_rounds,
                    "fixed_agents": list(config.fixed_agents),
                    "exposure_policy": config.exposure_policy.to_dict(),
                    "darkforest_belief": belief,
                    "coordinator_prompt": coordinator_prompt,
                    "exposed_content": exposed_content,
                    "coordinator_response": coordinator_response.text,
                    "coordinator_error": coordinator_response.error,
                    "coordinator_usage": coordinator_response.usage,
                    "coordinator_latency_sec": coordinator_response.latency_sec,
                    "coordinator_final_parse": final_parse,
                    "belief_guardrail": guardrail_report,
                    "guarded_final_parse": guarded_final_parse,
                    "final_parse_method": guarded_final_parse.get("parse_method"),
                },
            }
        )
        initial_usages = [agent_outputs[agent].usage for agent in fixed_agents]
        individual_latencies = {agent: agent_outputs[agent].latency_sec for agent in fixed_agents}
        record["metrics"] = build_sample_metrics(
            correct=bool(record["correct"]),
            invalid_parse=record["invalid_parse"],
            initial_usages=initial_usages,
            coordination_usage=coordinator_response.usage,
            initial_latency_sec=initial_phase_latency,
            coordination_latency_sec=coordinator_response.latency_sec,
            individual_initial_latencies=individual_latencies,
            exposure_metrics=exposure_metrics,
            coordination_rounds=config.coordination_rounds,
            scored=scored,
        )
        if save_prompts:
            record["prompts"] = {
                "initial_agents": agent_prompts,
                "coordinator": coordinator_prompt,
            }
        append_jsonl(output_file, record)
        records.append(record)

    wall = time.perf_counter() - start_wall
    summary = aggregate_evaluation_summary(
        records,
        split=split,
        mode=mode,
        config=config,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        calibration_file=calibration_file,
        calibration_num_samples=calibration_num_samples,
        total_wall_time_sec=wall,
    )
    write_json(summary_file, summary)
    return summary


def run_dry_run_samples(
    samples: Iterable[MathSample],
    config: DarkForestConfig,
    *,
    output_file: Path,
    overwrite: bool,
    save_prompts: bool = True,
) -> List[Dict[str, Any]]:
    if output_file.exists() and overwrite:
        output_file.write_text("", encoding="utf-8")
    elif output_file.exists():
        raise FileExistsError(f"Dry-run output file exists; pass --overwrite true: {output_file}")
    records = []
    fixed_agents = list(config.fixed_agents)
    for sample in samples:
        initial_prompt = build_initial_agent_prompt(sample.question, config.initial_prompt_style)
        empty_outputs = {
            agent: parse_agent_response(agent, "", usage={}, error="dry_run_no_call")
            for agent in fixed_agents
        }
        belief = compute_darkforest_belief(empty_outputs, config)
        raw_coordinator_prompt, exposed_content, exposure_metrics = build_static_coordinator_prompt_with_exposure(
            sample.question,
            empty_outputs,
            config,
            belief,
        )
        coordinator_prompt = prompt_for_agent(raw_coordinator_prompt, config.coordinator_model, config)
        record = _sample_common_record(sample, initial_prompt)
        record.update(
            {
                "agents": _as_dict_outputs(empty_outputs, fixed_agents),
                "coordination": {
                    "method": config.coordination_method,
                    "design_name": config.design_name,
                    "coordinator_model": config.coordinator_model,
                    "coordination_rounds": config.coordination_rounds,
                    "fixed_agents": list(config.fixed_agents),
                    "exposure_policy": config.exposure_policy.to_dict(),
                    "darkforest_belief": belief,
                    "coordinator_prompt": coordinator_prompt,
                    "exposed_content": exposed_content,
                },
                "metrics": {
                    "exposure_metrics": exposure_metrics,
                    "llm_calls_total": 0,
                },
            }
        )
        if save_prompts:
            record["prompts"] = {
                "initial_agents": {
                    agent: prompt_for_agent(initial_prompt, agent, config) for agent in fixed_agents
                },
                "coordinator": coordinator_prompt,
            }
        append_jsonl(output_file, record)
        records.append(record)
    return records
