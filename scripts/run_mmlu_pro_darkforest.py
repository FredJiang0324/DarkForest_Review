#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from darkforest.dataset_mmlu_pro import (  # noqa: E402
    MMLUProSample,
    build_mmlu_pro_cot_prompt,
    default_mmlu_pro_paths,
    group_validation_by_category,
    load_goa_mmlu_pro_sampled_test,
    load_mmlu_pro_jsonl,
    verify_goa_sampled_test_subset,
)
from darkforest.eval_mmlu_pro import (  # noqa: E402
    MMLU_PRO_AGENTS,
    MMLUProDarkForestConfig,
    aggregate_mmlu_pro_summary,
    apply_mmlu_pro_calibration,
    apply_mmlu_pro_belief_guardrail,
    compute_mmlu_pro_belief,
    estimate_mmlu_pro_calibration,
    make_mmlu_pro_metrics,
    query_mmlu_pro_verifier,
    query_mmlu_pro_coordinator,
    query_mmlu_pro_initial_agents,
    score_mmlu_pro_prediction,
    should_trigger_mmlu_pro_verifier,
    write_records_jsonl,
)
from darkforest.llm_client import VLLMClient  # noqa: E402
from darkforest.utils import command_line, parse_bool, read_json, write_json  # noqa: E402


def _path_under_root(root_dir: Path, value: Optional[str], default_relative: str) -> Path:
    path = Path(value) if value else root_dir / default_relative
    if not path.is_absolute():
        path = root_dir / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside root_dir: {resolved}") from exc
    return resolved


def _read_path(root_dir: Path, value: Optional[str], default_path: Path) -> Path:
    path = Path(value) if value else default_path
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


def _limit(samples: Sequence[MMLUProSample], limit: Optional[int]) -> List[MMLUProSample]:
    if limit is None:
        return list(samples)
    return list(samples)[: max(0, int(limit))]


def _read_records_by_idx(path: Path) -> Tuple[List[Dict[str, Any]], set[int]]:
    if not path.exists():
        return [], set()
    records = []
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            completed.add(int(record["idx"]))
    return records, completed


def _append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n")
        handle.flush()


def _make_clients(args: argparse.Namespace) -> Dict[str, VLLMClient]:
    return {
        "qwen": VLLMClient(
            args.qwen_endpoint,
            args.qwen_model_name,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            api_style=args.api_style,
        ),
        "qwen_coder": VLLMClient(
            args.coder_endpoint,
            args.coder_model_name,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            api_style=args.api_style,
        ),
        "mathstral": VLLMClient(
            args.mathstral_endpoint,
            args.mathstral_model_name,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            api_style=args.api_style,
        ),
    }


def _make_config(args: argparse.Namespace) -> MMLUProDarkForestConfig:
    priors = {agent: 1.0 for agent in MMLU_PRO_AGENTS}
    if args.agent_priors:
        for item in args.agent_priors.split(","):
            key, raw = item.split("=", 1)
            key = key.strip()
            if key not in priors:
                raise ValueError(f"Unknown MMLU-Pro agent in --agent_priors: {key}")
            priors[key] = float(raw)
    return MMLUProDarkForestConfig(
        coordinator_model=args.coordinator_model,
        agent_priors=priors,
        accept_threshold=args.accept_threshold,
        uncertainty_threshold=args.uncertainty_threshold,
        min_support_pattern_count=args.min_support_pattern_count,
        expose_belief_summary=args.expose_belief_summary,
        expose_full_responses=args.expose_full_responses,
        expose_reasoning=args.expose_reasoning,
        max_peer_response_chars=args.max_peer_response_chars,
        max_reasoning_chars=args.max_reasoning_chars,
        decision_policy=args.mmlu_decision_policy,
        verifier_model=args.verifier_model,
        verifier_trigger=args.verifier_trigger,
        anchor_agent=args.anchor_agent,
        belief_guardrail=args.belief_guardrail,
        belief_guardrail_min_posterior=args.belief_guardrail_min_posterior,
        belief_guardrail_min_margin=args.belief_guardrail_min_margin,
        freeze_calibration=args.freeze_calibration,
    )


def _load_calibration_if_needed(
    args: argparse.Namespace,
    config: MMLUProDarkForestConfig,
    root_dir: Path,
) -> Tuple[MMLUProDarkForestConfig, Optional[Dict[str, Any]], Optional[str]]:
    if not args.calibration_file:
        if args.require_calibration:
            raise FileNotFoundError("--require_calibration true but --calibration_file was not provided")
        return config, None, None
    path = _read_path(root_dir, args.calibration_file, Path(args.calibration_file))
    if not path.exists():
        if args.require_calibration:
            raise FileNotFoundError(f"Calibration file not found: {path}")
        return config, None, str(path)
    calibration = read_json(path)
    return apply_mmlu_pro_calibration(config, calibration, str(path)), calibration, str(path)


def run_dry_run(
    validation_samples: Sequence[MMLUProSample],
    eval_samples: Sequence[MMLUProSample],
    output_dir: Path,
    ntrain: int,
) -> None:
    validation_by_category = group_validation_by_category(validation_samples)
    previews = []
    for sample in eval_samples[:3]:
        previews.append(
            {
                "idx": sample.idx,
                "category": sample.category,
                "answer": sample.answer,
                "prompt": build_mmlu_pro_cot_prompt(sample, validation_by_category, ntrain=ntrain),
            }
        )
    write_json(output_dir / "dry_run_prompts.json", {"dataset": "MMLU-Pro", "prompts": previews})
    print(f"Wrote dry-run prompt preview: {output_dir / 'dry_run_prompts.json'}")


def run_calibration(
    validation_samples: Sequence[MMLUProSample],
    clients: Dict[str, VLLMClient],
    config: MMLUProDarkForestConfig,
    args: argparse.Namespace,
    output_dir: Path,
    calibration_output: Path,
) -> Dict[str, Any]:
    validation_by_category = group_validation_by_category(validation_samples)
    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(validation_samples):
        print(f"[calibrate] {idx + 1}/{len(validation_samples)} {sample.category}#{sample.question_id}", flush=True)
        agent_outputs, initial_prompt, initial_latency = query_mmlu_pro_initial_agents(
            sample,
            validation_by_category,
            clients,
            config,
            ntrain=args.ntrain,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            parallel_agents=args.parallel_agents,
            exclude_validation_self=True,
        )
        belief = compute_mmlu_pro_belief(agent_outputs, config)
        agent_correct = {
            agent: (agent_outputs[agent].parsed_answer == sample.answer) for agent in config.fixed_agents
        }
        record = {
            "idx": idx,
            "question_id": sample.question_id,
            "category": sample.category,
            "question": sample.question,
            "options": sample.options,
            "gold_answer": sample.answer,
            "agents": {agent: agent_outputs[agent].to_dict() for agent in config.fixed_agents},
            "agent_correct": agent_correct,
            "darkforest_belief": belief,
            "metrics": make_mmlu_pro_metrics(
                correct=False,
                invalid_parse=False,
                agent_outputs=agent_outputs,
                coordinator_output=None,
                initial_latency_sec=initial_latency,
                exposure_metrics={
                    "num_agents": len(config.fixed_agents),
                    "num_agent_outputs_exposed_to_coordinator": 0,
                    "cross_agent_input_chars": 0,
                    "cross_agent_input_tokens": 0,
                    "raw_full_response_exposed": False,
                    "reasoning_exposed": False,
                    "confidence_exposed": False,
                    "belief_summary_exposed": False,
                },
                config=config,
            ),
        }
        if args.save_prompts:
            record["prompts"] = {"initial_agent_prompt": initial_prompt}
        records.append(record)

    calibration = estimate_mmlu_pro_calibration(
        records,
        config,
        seed=args.seed,
        command=command_line(),
        root_dir=str(args.root_dir),
    )
    write_records_jsonl(records, output_dir / "calibration_records.jsonl")
    write_json(calibration_output, calibration)
    print(f"Wrote MMLU-Pro calibration: {calibration_output}")
    return calibration


def run_evaluation(
    eval_samples: Sequence[MMLUProSample],
    validation_samples: Sequence[MMLUProSample],
    clients: Dict[str, VLLMClient],
    config: MMLUProDarkForestConfig,
    args: argparse.Namespace,
    output_dir: Path,
    summary_file: Path,
    calibration_file: Optional[str],
    calibration_num_samples: Optional[int],
) -> Dict[str, Any]:
    validation_by_category = group_validation_by_category(validation_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "evaluation_records.jsonl"
    if args.overwrite and records_path.exists():
        records_path.unlink()
    if records_path.exists() and not args.resume:
        raise FileExistsError(
            f"{records_path} already exists. Use --resume true to continue or --overwrite true to restart."
        )
    records, completed_indices = _read_records_by_idx(records_path) if args.resume else ([], set())
    start_wall = time.perf_counter()
    for idx, sample in enumerate(eval_samples):
        if idx in completed_indices:
            continue
        print(f"[evaluate] {idx + 1}/{len(eval_samples)} {sample.category} idx={sample.idx}", flush=True)
        agent_outputs, initial_prompt, initial_latency = query_mmlu_pro_initial_agents(
            sample,
            validation_by_category,
            clients,
            config,
            ntrain=args.ntrain,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            parallel_agents=args.parallel_agents,
            exclude_validation_self=False,
        )
        belief = compute_mmlu_pro_belief(agent_outputs, config)
        coordinator_output = None
        coordinator_prompt = None
        coordinator_exposed_content = ""
        verifier_output = None
        verifier_prompt = None
        verifier_exposed_content = ""
        verifier_triggered = False
        verifier_trigger_reason = "not_applicable"
        exposure_metrics = {
            "num_agents": len(config.fixed_agents),
            "num_agent_outputs_exposed_to_coordinator": 0,
            "cross_agent_input_chars": 0,
            "cross_agent_input_tokens": 0,
            "raw_full_response_exposed": False,
            "reasoning_exposed": False,
            "confidence_exposed": False,
            "belief_summary_exposed": False,
        }
        guardrail_info = {
            "policy": config.belief_guardrail,
            "applied": False,
            "original_answer": None,
            "final_answer": None,
            "reason": "not_applicable",
        }

        if config.decision_policy in {"coordinator", "coordinator_verifier"}:
            coordinator_output, coordinator_prompt, coordinator_exposed_content, exposure_metrics = query_mmlu_pro_coordinator(
                sample,
                clients,
                config,
                agent_outputs,
                belief,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed + 100000 + idx,
            )
            final_answer, guardrail_info = apply_mmlu_pro_belief_guardrail(
                coordinator_output.parsed_answer,
                belief,
                config,
            )
            if config.decision_policy == "coordinator_verifier":
                verifier_triggered, verifier_trigger_reason = should_trigger_mmlu_pro_verifier(
                    agent_outputs,
                    belief,
                    config,
                )
                if verifier_triggered:
                    verifier_output, verifier_prompt, verifier_exposed_content, verifier_exposure = query_mmlu_pro_verifier(
                        sample,
                        clients,
                        config,
                        agent_outputs,
                        belief,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        seed=args.seed + 200000 + idx,
                        coordinator_answer=final_answer,
                        mode="targeted_verifier",
                    )
                    final_answer = verifier_output.parsed_answer
                    exposure_metrics = verifier_exposure
        elif config.decision_policy == "qwen_anchor_verifier":
            anchor_output = agent_outputs.get(config.anchor_agent)
            final_answer = anchor_output.parsed_answer if anchor_output else None
            guardrail_info = {
                "policy": "qwen_anchor",
                "applied": False,
                "original_answer": final_answer,
                "final_answer": final_answer,
                "reason": "anchor_default",
            }
            verifier_triggered, verifier_trigger_reason = should_trigger_mmlu_pro_verifier(
                agent_outputs,
                belief,
                config,
            )
            if verifier_triggered:
                verifier_output, verifier_prompt, verifier_exposed_content, exposure_metrics = query_mmlu_pro_verifier(
                    sample,
                    clients,
                    config,
                    agent_outputs,
                    belief,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=args.seed + 200000 + idx,
                    coordinator_answer=None,
                    mode="qwen_anchor_verifier",
                )
                final_answer = verifier_output.parsed_answer
        else:
            raise ValueError(f"Unknown --mmlu_decision_policy: {config.decision_policy}")

        correct, scored_answer, invalid_parse = score_mmlu_pro_prediction(
            final_answer,
            sample.answer,
            len(sample.options),
            invalid_answer_policy=args.invalid_answer_policy,
            seed=args.seed + idx,
        )
        record = {
            "idx": idx,
            "category": sample.category,
            "question_id": sample.question_id,
            "question": sample.question,
            "options": sample.options,
            "gold_answer": sample.answer,
            "parsed_answer": final_answer,
            "coordinator_raw_parsed_answer": (
                coordinator_output.parsed_answer if coordinator_output is not None else None
            ),
            "scored_answer": scored_answer,
            "correct": correct,
            "invalid_parse": invalid_parse,
            "agents": {agent: agent_outputs[agent].to_dict() for agent in config.fixed_agents},
            "coordination": {
                "method": config.coordination_method,
                "design_name": config.design_name,
                "coordinator_model": config.coordinator_model,
                "fixed_agents": list(config.fixed_agents),
                "decision_policy": config.decision_policy,
                "darkforest_belief": belief,
                "coordinator_prompt": coordinator_prompt,
                "exposed_content": coordinator_exposed_content or verifier_exposed_content,
                "coordinator_response": coordinator_output.raw_response if coordinator_output is not None else None,
                "coordinator_parsed": coordinator_output.to_dict() if coordinator_output is not None else None,
                "verifier_model": config.verifier_model,
                "verifier_trigger": config.verifier_trigger,
                "verifier_triggered": verifier_triggered,
                "verifier_trigger_reason": verifier_trigger_reason,
                "verifier_prompt": verifier_prompt,
                "verifier_exposed_content": verifier_exposed_content,
                "verifier_response": verifier_output.raw_response if verifier_output is not None else None,
                "verifier_parsed": verifier_output.to_dict() if verifier_output is not None else None,
                "belief_guardrail": guardrail_info,
            },
            "metrics": make_mmlu_pro_metrics(
                correct=correct,
                invalid_parse=invalid_parse,
                agent_outputs=agent_outputs,
                coordinator_output=coordinator_output,
                initial_latency_sec=initial_latency,
                exposure_metrics=exposure_metrics,
                config=config,
                verification_output=verifier_output,
            ),
        }
        if args.save_prompts:
            record["prompts"] = {
                "initial_agent_prompt": initial_prompt,
                "coordinator_prompt": coordinator_prompt,
                "verifier_prompt": verifier_prompt,
            }
        records.append(record)
        _append_record(records_path, record)

    total_wall = time.perf_counter() - start_wall
    records = sorted(records, key=lambda row: int(row["idx"]))
    summary = aggregate_mmlu_pro_summary(
        records,
        mode=args.mode,
        config=config,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        calibration_file=calibration_file,
        calibration_num_samples=calibration_num_samples,
        total_wall_time_sec=total_wall,
        invalid_answer_policy=args.invalid_answer_policy,
    )
    write_json(summary_file, summary)
    print(f"Wrote MMLU-Pro records: {output_dir / 'evaluation_records.jsonl'}")
    print(f"Wrote MMLU-Pro summary: {summary_file}")
    print(f"MMLU-Pro accuracy: {summary['accuracy_percent']:.2f}% ({summary['num_correct']}/{summary['num_samples']})")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DarkForest on the sampled MMLU-Pro test set.")
    parser.add_argument("--root_dir", default=str(ROOT))
    parser.add_argument("--mode", choices=["calibrate", "evaluate", "calibrate_and_evaluate", "dry_run"], required=True)
    parser.add_argument("--validation_path", default=None)
    parser.add_argument("--full_test_path", default=None)
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--output_dir", default="outputs/MMLU-Pro/darkforest_sampled_qwen_coder_mathstral")
    parser.add_argument("--summary_file", default=None)
    parser.add_argument("--calibration_output", default="outputs/MMLU-Pro/darkforest_calibration/calibration.json")
    parser.add_argument("--calibration_file", default=None)
    parser.add_argument("--require_calibration", type=parse_bool, default=False)
    parser.add_argument("--freeze_calibration", type=parse_bool, default=True)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--limit_calibration_samples", type=int, default=None)
    parser.add_argument("--limit_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel_agents", type=parse_bool, default=True)
    parser.add_argument("--save_prompts", type=parse_bool, default=False)
    parser.add_argument("--resume", type=parse_bool, default=False)
    parser.add_argument("--overwrite", type=parse_bool, default=False)

    parser.add_argument("--qwen_endpoint", default="http://localhost:8000/v1/completions")
    parser.add_argument("--coder_endpoint", default="http://localhost:8001/v1/completions")
    parser.add_argument("--mathstral_endpoint", default="http://localhost:8002/v1/completions")
    parser.add_argument("--qwen_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--coder_model_name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--mathstral_model_name", default="mistralai/Mathstral-7B-v0.1")
    parser.add_argument("--api_style", choices=["chat", "completions"], default="completions")

    parser.add_argument("--coordinator_model", choices=MMLU_PRO_AGENTS, default="qwen_coder")
    parser.add_argument(
        "--mmlu_decision_policy",
        choices=["coordinator", "coordinator_verifier", "qwen_anchor_verifier"],
        default="coordinator",
    )
    parser.add_argument("--verifier_model", choices=MMLU_PRO_AGENTS, default="qwen")
    parser.add_argument(
        "--verifier_trigger",
        choices=["high_uncertainty_or_qwen_coder_disagree", "disagreement", "always"],
        default="high_uncertainty_or_qwen_coder_disagree",
    )
    parser.add_argument("--anchor_agent", choices=MMLU_PRO_AGENTS, default="qwen")
    parser.add_argument("--coordination_method", default="darkforest")
    parser.add_argument("--coordination_rounds", type=int, default=1)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--invalid_answer_policy", choices=["incorrect", "random"], default="random")
    parser.add_argument("--expose_belief_summary", type=parse_bool, default=True)
    parser.add_argument("--expose_full_responses", type=parse_bool, default=False)
    parser.add_argument("--expose_reasoning", type=parse_bool, default=False)
    parser.add_argument("--max_peer_response_chars", type=int, default=3000)
    parser.add_argument("--max_reasoning_chars", type=int, default=900)
    parser.add_argument("--belief_guardrail", choices=["none", "trust_supported_cluster"], default="none")
    parser.add_argument("--belief_guardrail_min_posterior", type=float, default=0.66)
    parser.add_argument("--belief_guardrail_min_margin", type=float, default=0.25)
    parser.add_argument("--agent_priors", default=None)
    parser.add_argument("--accept_threshold", type=float, default=0.75)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.60)
    parser.add_argument("--min_support_pattern_count", type=int, default=5)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--timeout_sec", type=float, default=300.0)
    parser.add_argument("--max_retries", type=int, default=3)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.root_dir = Path(args.root_dir).resolve()
    if args.coordination_rounds != 1:
        raise NotImplementedError("MMLU-Pro runner currently supports --coordination_rounds 1 only.")
    if args.coordination_method != "darkforest":
        raise ValueError("MMLU-Pro runner currently implements only --coordination_method darkforest.")

    validation_default, full_test_default, eval_default = default_mmlu_pro_paths(args.root_dir)
    validation_path = _read_path(args.root_dir, args.validation_path, validation_default)
    full_test_path = _read_path(args.root_dir, args.full_test_path, full_test_default)
    eval_path = _read_path(args.root_dir, args.eval_data_path, eval_default)

    validation_samples = load_mmlu_pro_jsonl(validation_path)
    eval_samples = load_goa_mmlu_pro_sampled_test(eval_path, full_test_path=full_test_path)
    subset_report = verify_goa_sampled_test_subset(full_test_path, eval_path)
    if args.limit_samples is not None:
        validation_samples = _limit(validation_samples, args.limit_samples)
        eval_samples = _limit(eval_samples, args.limit_samples)
    validation_samples = _limit(validation_samples, args.limit_calibration_samples)
    eval_samples = _limit(eval_samples, args.limit_eval_samples)

    output_dir = _path_under_root(args.root_dir, args.output_dir, "outputs/MMLU-Pro/darkforest_sampled_qwen_coder_mathstral")
    summary_file = _path_under_root(args.root_dir, args.summary_file, str(output_dir.relative_to(args.root_dir) / "summary.json"))
    calibration_output = _path_under_root(
        args.root_dir,
        args.calibration_output,
        "outputs/MMLU-Pro/darkforest_calibration/calibration.json",
    )
    config = _make_config(args)

    print(f"Root: {args.root_dir}")
    print(f"MMLU-Pro validation: {validation_path} ({len(validation_samples)} samples)")
    print(f"MMLU-Pro sampled test: {eval_path} ({len(eval_samples)} samples)")
    print(f"Sampled subset check: {subset_report}")

    if args.mode == "dry_run":
        run_dry_run(validation_samples, eval_samples, output_dir, ntrain=args.ntrain)
        return 0

    clients = _make_clients(args)
    calibration: Optional[Dict[str, Any]] = None
    calibration_file: Optional[str] = args.calibration_file
    if args.mode == "evaluate":
        config, calibration, calibration_file = _load_calibration_if_needed(args, config, args.root_dir)

    if args.mode in {"calibrate", "calibrate_and_evaluate"}:
        calibration = run_calibration(
            validation_samples,
            clients,
            config,
            args,
            output_dir if args.mode == "calibrate" else calibration_output.parent,
            calibration_output,
        )
        calibration_file = str(calibration_output)
        if args.mode == "calibrate":
            return 0
        config = apply_mmlu_pro_calibration(config, calibration, str(calibration_output))

    calibration_num_samples = int(calibration.get("num_calibration_samples", 0)) if calibration else None
    run_evaluation(
        eval_samples,
        validation_samples,
        clients,
        config,
        args,
        output_dir,
        summary_file,
        calibration_file=calibration_file,
        calibration_num_samples=calibration_num_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
