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

from darkforest.dataset_humaneval import (  # noqa: E402
    HumanEvalSample,
    default_humaneval_paths,
    split_calibration_and_eval,
)
from darkforest.eval_humaneval import (  # noqa: E402
    HUMANEVAL_AGENTS,
    HumanEvalDarkForestConfig,
    aggregate_humaneval_summary,
    apply_humaneval_calibration,
    apply_humaneval_guardrail,
    build_humaneval_initial_prompt,
    compute_humaneval_belief,
    estimate_humaneval_calibration,
    make_sample_metrics,
    query_humaneval_coordinator,
    query_humaneval_initial_agents,
    run_humaneval_evaluator,
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


def _limit(samples: Sequence[HumanEvalSample], limit: Optional[int]) -> List[HumanEvalSample]:
    if limit is None:
        return list(samples)
    return list(samples)[: max(0, int(limit))]


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


def _make_config(args: argparse.Namespace) -> HumanEvalDarkForestConfig:
    priors = {agent: 1.0 for agent in HUMANEVAL_AGENTS}
    if args.agent_priors:
        for item in args.agent_priors.split(","):
            key, raw = item.split("=", 1)
            key = key.strip()
            if key not in priors:
                raise ValueError(f"Unknown HumanEval agent in --agent_priors: {key}")
            priors[key] = float(raw)
    return HumanEvalDarkForestConfig(
        coordinator_model=args.coordinator_model,
        agent_priors=priors,
        accept_threshold=args.accept_threshold,
        uncertainty_threshold=args.uncertainty_threshold,
        min_support_pattern_count=args.min_support_pattern_count,
        expose_belief_summary=args.expose_belief_summary,
        expose_full_responses=args.expose_full_responses,
        max_peer_response_chars=args.max_peer_response_chars,
        belief_guardrail=args.belief_guardrail,
        anchor_agent=args.anchor_agent,
        belief_guardrail_min_posterior=args.belief_guardrail_min_posterior,
        belief_guardrail_min_margin=args.belief_guardrail_min_margin,
        anchor_fallback_min_posterior=args.anchor_fallback_min_posterior,
        freeze_calibration=args.freeze_calibration,
    )


def _load_calibration_if_needed(
    args: argparse.Namespace,
    config: HumanEvalDarkForestConfig,
    root_dir: Path,
) -> Tuple[HumanEvalDarkForestConfig, Optional[Dict[str, Any]], Optional[str]]:
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
    return apply_humaneval_calibration(config, calibration, str(path)), calibration, str(path)


def run_dry_run(
    samples: Sequence[HumanEvalSample],
    config: HumanEvalDarkForestConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    previews = []
    for idx, sample in enumerate(samples[:3]):
        previews.append(
            {
                "idx": idx,
                "task_id": sample.task_id,
                "entry_point": sample.entry_point,
                "initial_prompt": build_humaneval_initial_prompt(sample),
            }
        )
    write_json(output_dir / "dry_run_prompts.json", {"dataset": "HumanEval", "config": config.to_dict(), "prompts": previews})
    print(f"Wrote dry-run prompt preview: {output_dir / 'dry_run_prompts.json'}")


def run_calibration(
    samples: Sequence[HumanEvalSample],
    clients: Dict[str, VLLMClient],
    config: HumanEvalDarkForestConfig,
    args: argparse.Namespace,
    output_dir: Path,
    calibration_output: Path,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    completion_rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        print(f"[calibrate] {idx + 1}/{len(samples)} {sample.task_id}", flush=True)
        agent_outputs, initial_prompt, initial_latency = query_humaneval_initial_agents(
            sample,
            clients,
            config,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            parallel_agents=args.parallel_agents,
        )
        belief = compute_humaneval_belief(agent_outputs, config)
        record = {
            "idx": idx,
            "task_id": sample.task_id,
            "entry_point": sample.entry_point,
            "prompt": sample.prompt,
            "agents": {agent: agent_outputs[agent].to_dict() for agent in config.fixed_agents},
            "darkforest_belief": belief,
            "metrics": make_sample_metrics(
                passed=False,
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
            ),
        }
        if args.save_prompts:
            record["prompts"] = {"initial_agent_prompt": initial_prompt}
        records.append(record)
        for agent in config.fixed_agents:
            completion_rows.append(
                {
                    "task_id": sample.task_id,
                    "completion": agent_outputs[agent].parsed_completion or "",
                    "agent_key": agent,
                    "idx": idx,
                }
            )

    evaluator_result, result_rows, sample_file, problem_file = run_humaneval_evaluator(
        samples,
        completion_rows,
        output_dir,
        "calibration_agent_outputs",
        n_workers=args.eval_workers,
        timeout=args.eval_timeout,
    )
    passed_by_key = {
        (row.get("task_id"), row.get("agent_key")): bool(row.get("passed")) for row in result_rows
    }
    for record in records:
        agent_passed = {
            agent: passed_by_key.get((record["task_id"], agent), False) for agent in config.fixed_agents
        }
        record["agent_passed"] = agent_passed
        record["evaluator"] = {
            "sample_file": str(sample_file),
            "problem_file": str(problem_file),
        }

    calibration = estimate_humaneval_calibration(
        records,
        config,
        seed=args.seed,
        command=command_line(),
        root_dir=str(args.root_dir),
    )
    calibration["evaluator_result_raw"] = evaluator_result
    write_records_jsonl(records, output_dir / "calibration_records.jsonl")
    write_json(calibration_output, calibration)
    print(f"Wrote HumanEval calibration: {calibration_output}")
    return calibration


def run_evaluation(
    samples: Sequence[HumanEvalSample],
    clients: Dict[str, VLLMClient],
    config: HumanEvalDarkForestConfig,
    args: argparse.Namespace,
    output_dir: Path,
    summary_file: Path,
    calibration_file: Optional[str],
    calibration_num_samples: Optional[int],
) -> Dict[str, Any]:
    start_wall = time.perf_counter()
    records: List[Dict[str, Any]] = []
    completion_rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        print(f"[evaluate] {idx + 1}/{len(samples)} {sample.task_id}", flush=True)
        agent_outputs, initial_prompt, initial_latency = query_humaneval_initial_agents(
            sample,
            clients,
            config,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            parallel_agents=args.parallel_agents,
        )
        belief = compute_humaneval_belief(agent_outputs, config)
        coordinator_output, coordinator_prompt, exposed_content, exposure_metrics = query_humaneval_coordinator(
            sample,
            clients,
            config,
            agent_outputs,
            belief,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + 100000 + idx,
        )
        final_completion, guardrail_report = apply_humaneval_guardrail(
            coordinator_output, belief, agent_outputs, config
        )
        invalid_parse = not bool(final_completion.strip())
        record = {
            "idx": idx,
            "task_id": sample.task_id,
            "entry_point": sample.entry_point,
            "prompt": sample.prompt,
            "final_completion": final_completion,
            "passed": None,
            "invalid_parse": invalid_parse,
            "agents": {agent: agent_outputs[agent].to_dict() for agent in config.fixed_agents},
            "coordination": {
                "method": config.coordination_method,
                "design_name": config.design_name,
                "coordinator_model": config.coordinator_model,
                "fixed_agents": list(config.fixed_agents),
                "darkforest_belief": belief,
                "coordinator_prompt": coordinator_prompt,
                "exposed_content": exposed_content,
                "coordinator_response": coordinator_output.raw_response,
                "coordinator_parsed": coordinator_output.to_dict(),
                "belief_guardrail_report": guardrail_report,
            },
            "metrics": make_sample_metrics(
                passed=False,
                invalid_parse=invalid_parse,
                agent_outputs=agent_outputs,
                coordinator_output=coordinator_output,
                initial_latency_sec=initial_latency,
                exposure_metrics=exposure_metrics,
            ),
        }
        if args.save_prompts:
            record["prompts"] = {
                "initial_agent_prompt": initial_prompt,
                "coordinator_prompt": coordinator_prompt,
            }
        records.append(record)
        completion_rows.append({"task_id": sample.task_id, "completion": final_completion, "idx": idx})

    evaluator_result, result_rows, sample_file, problem_file = run_humaneval_evaluator(
        samples,
        completion_rows,
        output_dir,
        "evaluation_final_completions",
        n_workers=args.eval_workers,
        timeout=args.eval_timeout,
    )
    passed_by_task = {row.get("task_id"): bool(row.get("passed")) for row in result_rows}
    for record in records:
        passed = passed_by_task.get(record["task_id"], False)
        record["passed"] = passed
        record["metrics"]["correct"] = passed
        record["metrics"]["scored"] = True
        record["evaluator"] = {
            "sample_file": str(sample_file),
            "problem_file": str(problem_file),
            "result": next((row.get("result") for row in result_rows if row.get("task_id") == record["task_id"]), None),
        }

    total_wall = time.perf_counter() - start_wall
    write_records_jsonl(records, output_dir / "evaluation_records.jsonl")
    summary = aggregate_humaneval_summary(
        records,
        mode=args.mode,
        config=config,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        calibration_file=calibration_file,
        calibration_num_samples=calibration_num_samples,
        total_wall_time_sec=total_wall,
        evaluator_result=evaluator_result,
    )
    write_json(summary_file, summary)
    print(f"Wrote HumanEval evaluation records: {output_dir / 'evaluation_records.jsonl'}")
    print(f"Wrote HumanEval summary: {summary_file}")
    print(f"HumanEval pass@1: {summary['pass@1_percent']:.2f}% ({summary['num_passed']}/{summary['num_samples']})")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DarkForest on the sampled HumanEval subset.")
    parser.add_argument("--root_dir", default=str(ROOT))
    parser.add_argument("--mode", choices=["calibrate", "evaluate", "calibrate_and_evaluate", "dry_run"], required=True)
    parser.add_argument("--full_data_path", default=None)
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--output_dir", default="outputs/HumanEval/darkforest_sampled_qwen_coder_anchor")
    parser.add_argument("--summary_file", default=None)
    parser.add_argument("--calibration_output", default="outputs/HumanEval/darkforest_calibration/calibration.json")
    parser.add_argument("--calibration_file", default=None)
    parser.add_argument("--require_calibration", type=parse_bool, default=False)
    parser.add_argument("--freeze_calibration", type=parse_bool, default=True)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--limit_calibration_samples", type=int, default=None)
    parser.add_argument("--limit_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel_agents", type=parse_bool, default=True)
    parser.add_argument("--save_prompts", type=parse_bool, default=False)

    parser.add_argument("--qwen_endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--coder_endpoint", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--mathstral_endpoint", default="http://localhost:8002/v1/chat/completions")
    parser.add_argument("--qwen_model_name", default="qwen")
    parser.add_argument("--coder_model_name", default="qwen_coder")
    parser.add_argument("--mathstral_model_name", default="mathstral")
    parser.add_argument("--api_style", choices=["chat", "completions"], default="chat")

    parser.add_argument("--coordinator_model", choices=HUMANEVAL_AGENTS, default="qwen_coder")
    parser.add_argument("--coordination_method", default="darkforest")
    parser.add_argument("--coordination_rounds", type=int, default=1)
    parser.add_argument("--anchor_agent", choices=HUMANEVAL_AGENTS, default="qwen_coder")
    parser.add_argument("--belief_guardrail", default="coder_anchor_supported_cluster")
    parser.add_argument("--belief_guardrail_min_posterior", type=float, default=0.66)
    parser.add_argument("--belief_guardrail_min_margin", type=float, default=0.20)
    parser.add_argument("--anchor_fallback_min_posterior", type=float, default=0.80)
    parser.add_argument("--expose_belief_summary", type=parse_bool, default=True)
    parser.add_argument("--expose_full_responses", type=parse_bool, default=False)
    parser.add_argument("--max_peer_response_chars", type=int, default=4000)
    parser.add_argument("--agent_priors", default=None)
    parser.add_argument("--accept_threshold", type=float, default=0.75)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.60)
    parser.add_argument("--min_support_pattern_count", type=int, default=5)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--timeout_sec", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--eval_workers", type=int, default=4)
    parser.add_argument("--eval_timeout", type=float, default=3.0)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.root_dir = Path(args.root_dir).resolve()
    if args.coordination_rounds != 1:
        raise NotImplementedError("HumanEval runner currently supports --coordination_rounds 1 only.")
    if args.coordination_method != "darkforest":
        raise ValueError("HumanEval runner currently implements only --coordination_method darkforest.")

    full_default, eval_default = default_humaneval_paths(args.root_dir)
    full_path = _read_path(args.root_dir, args.full_data_path, full_default)
    eval_path = _read_path(args.root_dir, args.eval_data_path, eval_default)
    calibration_samples, eval_samples = split_calibration_and_eval(full_path, eval_path)
    if args.limit_samples is not None:
        calibration_samples = _limit(calibration_samples, args.limit_samples)
        eval_samples = _limit(eval_samples, args.limit_samples)
    calibration_samples = _limit(calibration_samples, args.limit_calibration_samples)
    eval_samples = _limit(eval_samples, args.limit_eval_samples)

    output_dir = _path_under_root(args.root_dir, args.output_dir, "outputs/HumanEval/darkforest_sampled_qwen_coder_anchor")
    summary_file = _path_under_root(args.root_dir, args.summary_file, str(output_dir.relative_to(args.root_dir) / "summary.json"))
    calibration_output = _path_under_root(
        args.root_dir,
        args.calibration_output,
        "outputs/HumanEval/darkforest_calibration/calibration.json",
    )
    config = _make_config(args)

    print(f"Root: {args.root_dir}")
    print(f"HumanEval full: {full_path}")
    print(f"HumanEval eval subset: {eval_path}")
    print(f"Calibration samples: {len(calibration_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")

    if args.mode == "dry_run":
        run_dry_run(eval_samples, config, output_dir)
        return 0

    clients = _make_clients(args)
    calibration: Optional[Dict[str, Any]] = None
    calibration_file: Optional[str] = args.calibration_file
    if args.mode == "evaluate":
        config, calibration, calibration_file = _load_calibration_if_needed(args, config, args.root_dir)

    if args.mode in {"calibrate", "calibrate_and_evaluate"}:
        calibration = run_calibration(
            calibration_samples,
            clients,
            config,
            args,
            output_dir if args.mode == "calibrate" else calibration_output.parent,
            calibration_output,
        )
        calibration_file = str(calibration_output)
        if args.mode == "calibrate":
            return 0
        config = apply_humaneval_calibration(config, calibration, str(calibration_output))

    calibration_num_samples = None
    if calibration:
        calibration_num_samples = int(calibration.get("num_calibration_samples", 0) or 0)
    run_evaluation(
        eval_samples,
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
