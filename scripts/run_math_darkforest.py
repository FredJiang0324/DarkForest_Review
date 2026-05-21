#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_GUESS = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_GUESS / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from darkforest.calibration import apply_calibration_to_config, load_calibration  # noqa: E402
from darkforest.dataset_math import load_math_samples, warn_missing_gold  # noqa: E402
from darkforest.eval_math import (  # noqa: E402
    run_calibration_samples,
    run_dry_run_samples,
    run_evaluation_samples,
)
from darkforest.llm_client import VLLMClient  # noqa: E402
from darkforest.schemas import AGENT_PROFILES, FIXED_AGENTS, DarkForestConfig, ExposurePolicy, agents_for_profile  # noqa: E402
from darkforest.utils import (  # noqa: E402
    command_line,
    deterministic_limit,
    parse_agent_priors,
    parse_bool,
    resolve_write_path,
    utc_now_iso,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DarkForest fixed 3-agent MATH benchmark runner")
    parser.add_argument("--root_dir", default=str(ROOT_GUESS))
    parser.add_argument(
        "--mode",
        choices=["calibrate", "evaluate", "calibrate_and_evaluate", "dry_run"],
        required=True,
    )
    parser.add_argument("--data", default="MATH")
    parser.add_argument("--eval", default="test", choices=["train", "test", "dev"])
    parser.add_argument("--data_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--output_file")
    parser.add_argument("--summary_file")
    parser.add_argument("--calibration_output")
    parser.add_argument("--calibration_file")
    parser.add_argument("--require_calibration", type=parse_bool, default=False)
    parser.add_argument("--freeze_calibration", type=parse_bool, default=True)
    parser.add_argument("--calibration_valid_fraction", type=float, default=0.2)
    parser.add_argument("--calibration_query_coordinator", type=parse_bool, default=False)
    parser.add_argument("--limit_samples", type=int)
    parser.add_argument("--shuffle_samples", type=parse_bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--parallel_agents", type=parse_bool, default=True)
    parser.add_argument("--resume", type=parse_bool, default=False)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--save_prompts", type=parse_bool, default=False)
    parser.add_argument("--cache_agent_outputs", type=parse_bool, default=False)
    parser.add_argument(
        "--agent_profile",
        choices=sorted(AGENT_PROFILES),
        default="fixed_mathstral_pair",
        help=(
            "Fixed 3-agent profile. fixed_mathstral_pair preserves the original "
            "qwen/mathstral_1/mathstral_2 setup; qwen_coder_mathstral uses "
            "qwen/qwen_coder/mathstral."
        ),
    )

    parser.add_argument("--qwen_model_path")
    parser.add_argument("--mathstral1_model_path")
    parser.add_argument("--mathstral2_model_path")
    parser.add_argument("--coder_model_path")
    parser.add_argument("--mathstral_model_path")

    parser.add_argument("--qwen_model_name", default="qwen")
    parser.add_argument("--mathstral1_model_name", default="mathstral_1")
    parser.add_argument("--mathstral2_model_name", default="mathstral_2")
    parser.add_argument("--coder_model_name", default="qwen_coder")
    parser.add_argument("--mathstral_model_name", default="mathstral")

    parser.add_argument("--qwen_endpoint", default="http://localhost:8000/v1/completions")
    parser.add_argument("--mathstral1_endpoint", default="http://localhost:8001/v1/completions")
    parser.add_argument("--mathstral2_endpoint", default="http://localhost:8002/v1/completions")
    parser.add_argument("--coder_endpoint", default="http://localhost:8001/v1/completions")
    parser.add_argument("--mathstral_endpoint", default="http://localhost:8002/v1/completions")

    parser.add_argument("--coordinator_model", default="qwen")
    parser.add_argument("--verifier_model", default="qwen")
    parser.add_argument("--coordination_method", default="darkforest")
    parser.add_argument("--coordination_rounds", type=int, default=1)

    parser.add_argument("--expose_reasoning", type=parse_bool, default=False)
    parser.add_argument("--expose_confidence", type=parse_bool, default=True)
    parser.add_argument("--expose_full_responses", type=parse_bool, default=False)
    parser.add_argument("--expose_belief_summary", type=parse_bool, default=True)
    parser.add_argument("--max_peer_response_chars", type=int, default=3000)
    parser.add_argument("--max_peer_response_tokens", type=int)

    parser.add_argument("--same_model_correlation_discount", type=float, default=0.5)
    parser.add_argument("--missing_confidence_default", type=float, default=0.5)
    parser.add_argument("--malformed_output_penalty", type=float, default=0.5)
    parser.add_argument("--accept_threshold", type=float, default=0.75)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.60)
    parser.add_argument("--min_support_pattern_count", type=int, default=10)
    parser.add_argument(
        "--answer_match_backend",
        choices=["exact", "math_verify"],
        default="exact",
        help="exact uses DarkForest's conservative normalization; math_verify matches GoA's math_verify-based scoring when installed.",
    )
    parser.add_argument(
        "--belief_guardrail",
        choices=["none", "trust_supported_cluster", "qwen_anchor_supported_cluster"],
        default="none",
        help=(
            "Optional deterministic post-coordinator guardrail. trust_supported_cluster replaces "
            "the coordinator answer with a supported belief cluster when the cluster has at least "
            "two agents and passes posterior/margin thresholds. qwen_anchor_supported_cluster "
            "uses an anchor agent answer by default and only lets a trusted multi-agent belief "
            "cluster override it."
        ),
    )
    parser.add_argument("--belief_guardrail_anchor_agent", default="qwen")
    parser.add_argument("--belief_guardrail_min_posterior", type=float, default=0.66)
    parser.add_argument("--belief_guardrail_min_margin", type=float, default=0.25)
    parser.add_argument("--agent_priors")
    parser.add_argument(
        "--api_style",
        choices=["completions", "chat"],
        default="completions",
        help="Use OpenAI-compatible completions or chat/completions. Chat lets vLLM apply each model's chat template.",
    )
    parser.add_argument(
        "--initial_prompt_style",
        choices=["freeform", "json"],
        default="freeform",
        help=(
            "freeform asks each initial agent for zero-shot CoT and a final boxed answer; "
            "json preserves the older strict JSON suffix for ablations."
        ),
    )
    parser.add_argument(
        "--coordinator_prompt_style",
        choices=["darkforest", "goa_pooling", "darkforest_audit", "darkforest_belief_audit"],
        default="darkforest",
        help=(
            "darkforest uses the original coordinator prompt; goa_pooling uses GoA-style synthesis wording; "
            "darkforest_audit asks the coordinator to independently solve and audit candidates; "
            "darkforest_belief_audit adds belief-conditioned selection and correction directives."
        ),
    )
    parser.add_argument(
        "--prompt_template_mode",
        choices=["raw", "model_native"],
        default="raw",
        help="raw preserves the original prompt; model_native wraps the same zero-shot task in each model's chat template.",
    )
    parser.add_argument("--qwen_prompt_template", choices=["auto", "raw", "qwen_chatml", "mistral_inst"], default="auto")
    parser.add_argument("--coder_prompt_template", choices=["auto", "raw", "qwen_chatml", "mistral_inst"], default="auto")
    parser.add_argument("--mathstral_prompt_template", choices=["auto", "raw", "qwen_chatml", "mistral_inst"], default="auto")
    parser.add_argument("--mathstral1_prompt_template", choices=["auto", "raw", "qwen_chatml", "mistral_inst"], default="auto")
    parser.add_argument("--mathstral2_prompt_template", choices=["auto", "raw", "qwen_chatml", "mistral_inst"], default="auto")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--timeout_sec", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    return parser


def _default_prompt_template(agent_key: str, mode: str) -> str:
    if mode == "raw":
        return "raw"
    if agent_key in {"qwen", "qwen_coder"}:
        return "qwen_chatml"
    if agent_key in {"mathstral", "mathstral_1", "mathstral_2"}:
        return "mistral_inst"
    return "raw"


def _resolve_prompt_templates(args: argparse.Namespace, fixed_agents: list[str]) -> Dict[str, str]:
    overrides = {
        "qwen": args.qwen_prompt_template,
        "qwen_coder": args.coder_prompt_template,
        "mathstral": args.mathstral_prompt_template,
        "mathstral_1": args.mathstral1_prompt_template,
        "mathstral_2": args.mathstral2_prompt_template,
    }
    templates: Dict[str, str] = {}
    for agent in fixed_agents:
        override = overrides.get(agent, "auto")
        templates[agent] = (
            _default_prompt_template(agent, args.prompt_template_mode)
            if override == "auto"
            else override
        )
    return templates


def build_config(args: argparse.Namespace) -> DarkForestConfig:
    if args.coordination_method != "darkforest":
        raise NotImplementedError("Only --coordination_method darkforest is implemented")
    fixed_agents = agents_for_profile(args.agent_profile)
    if args.coordinator_model not in fixed_agents:
        raise ValueError(
            f"--coordinator_model {args.coordinator_model!r} is not in profile "
            f"{args.agent_profile}: {fixed_agents}"
        )
    if args.verifier_model not in fixed_agents:
        raise ValueError(
            f"--verifier_model {args.verifier_model!r} is not in profile "
            f"{args.agent_profile}: {fixed_agents}"
        )
    if (
        args.belief_guardrail == "qwen_anchor_supported_cluster"
        and args.belief_guardrail_anchor_agent not in fixed_agents
    ):
        raise ValueError(
            f"--belief_guardrail_anchor_agent {args.belief_guardrail_anchor_agent!r} "
            f"is not in profile {args.agent_profile}: {fixed_agents}"
        )
    exposure = ExposurePolicy(
        expose_reasoning=args.expose_reasoning,
        expose_confidence=args.expose_confidence,
        expose_full_responses=args.expose_full_responses,
        expose_belief_summary=args.expose_belief_summary,
        max_peer_response_chars=args.max_peer_response_chars,
        max_peer_response_tokens=args.max_peer_response_tokens,
    )
    config = DarkForestConfig(
        coordination_method=args.coordination_method,
        fixed_agents=fixed_agents,
        coordinator_model=args.coordinator_model,
        coordination_rounds=args.coordination_rounds,
        agent_priors=parse_agent_priors(args.agent_priors, fixed_agents),
        same_model_correlation_discount=args.same_model_correlation_discount,
        missing_confidence_default=args.missing_confidence_default,
        malformed_output_penalty=args.malformed_output_penalty,
        accept_threshold=args.accept_threshold,
        uncertainty_threshold=args.uncertainty_threshold,
        min_support_pattern_count=args.min_support_pattern_count,
        answer_match_backend=args.answer_match_backend,
        belief_guardrail=args.belief_guardrail,
        belief_guardrail_anchor_agent=args.belief_guardrail_anchor_agent,
        belief_guardrail_min_posterior=args.belief_guardrail_min_posterior,
        belief_guardrail_min_margin=args.belief_guardrail_min_margin,
        exposure_policy=exposure,
        api_style=args.api_style,
        initial_prompt_style=args.initial_prompt_style,
        coordinator_prompt_style=args.coordinator_prompt_style,
        prompt_template_mode=args.prompt_template_mode,
        prompt_templates=_resolve_prompt_templates(args, fixed_agents),
        freeze_calibration=args.freeze_calibration,
        params_source="default",
    )
    config.parameter_sources = {
        "agent_priors": "default",
        "support_pattern_reliability": "default",
        "same_model_correlation_discount": "default",
        "missing_confidence_default": "default",
        "malformed_output_penalty": "default",
        "accept_threshold": "default",
        "uncertainty_threshold": "default",
    }
    return config


def build_clients(args: argparse.Namespace) -> Dict[str, VLLMClient]:
    if args.agent_profile == "qwen_coder_mathstral":
        specs = {
            "qwen": (args.qwen_endpoint, args.qwen_model_name),
            "qwen_coder": (args.coder_endpoint, args.coder_model_name),
            "mathstral": (args.mathstral_endpoint, args.mathstral_model_name),
        }
    else:
        specs = {
            "qwen": (args.qwen_endpoint, args.qwen_model_name),
            "mathstral_1": (args.mathstral1_endpoint, args.mathstral1_model_name),
            "mathstral_2": (args.mathstral2_endpoint, args.mathstral2_model_name),
        }
    return {
        key: VLLMClient(
            endpoint,
            model_name,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            api_style=args.api_style,
        )
        for key, (endpoint, model_name) in specs.items()
    }


def output_paths(args: argparse.Namespace, root_dir: Path, mode: str) -> Dict[str, Path]:
    timestamp = utc_now_iso().replace(":", "").replace("+", "_")
    default_dir = f"outputs/MATH/darkforest_{mode}_{timestamp}"
    output_dir = resolve_write_path(root_dir, args.output_dir, default_dir)
    if mode == "calibrate":
        default_records = output_dir.relative_to(root_dir) / "calibration_records.jsonl"
        default_summary = output_dir.relative_to(root_dir) / "calibration_summary.json"
    elif mode == "dry_run":
        default_records = output_dir.relative_to(root_dir) / "dry_run_records.jsonl"
        default_summary = output_dir.relative_to(root_dir) / "dry_run_summary.json"
    else:
        default_records = output_dir.relative_to(root_dir) / "evaluation_records.jsonl"
        default_summary = output_dir.relative_to(root_dir) / "summary.json"
    output_file = resolve_write_path(root_dir, args.output_file, str(default_records))
    summary_file = resolve_write_path(root_dir, args.summary_file, str(default_summary))
    calibration_output = resolve_write_path(
        root_dir,
        args.calibration_output,
        str(output_dir.relative_to(root_dir) / "calibration.json"),
    )
    return {
        "output_dir": output_dir,
        "output_file": output_file,
        "summary_file": summary_file,
        "calibration_output": calibration_output,
    }


def load_and_limit_samples(args: argparse.Namespace, root_dir: Path, split: str):
    if args.data != "MATH":
        raise NotImplementedError("Only --data MATH is implemented")
    samples = load_math_samples(
        args.data_path,
        split,
        root_dir,
        calibration_valid_fraction=args.calibration_valid_fraction,
        seed=args.seed,
    )
    if args.shuffle_samples:
        samples = list(samples)
        random.Random(args.seed).shuffle(samples)
    samples = deterministic_limit(samples, args.limit_samples)
    missing_gold = warn_missing_gold(samples)
    if missing_gold:
        print(
            f"Warning: {len(missing_gold)} samples have missing gold answers and will be excluded from scoring.",
            file=sys.stderr,
        )
    return samples


def load_calibration_if_needed(
    args: argparse.Namespace,
    config: DarkForestConfig,
) -> tuple[DarkForestConfig, Optional[Dict[str, Any]], Optional[str]]:
    if args.calibration_file:
        calibration_path = Path(args.calibration_file)
        calibration = load_calibration(calibration_path)
        return apply_calibration_to_config(config, calibration), calibration, str(calibration_path)
    if args.require_calibration:
        raise FileNotFoundError("--require_calibration true but --calibration_file was not provided")
    return config, None, None


def run_calibrate(args: argparse.Namespace, root_dir: Path, clients: Dict[str, VLLMClient]) -> Dict[str, Any]:
    if args.eval != "train":
        raise ValueError("--mode calibrate requires --eval train; MATH/test must not be used for calibration")
    config = build_config(args)
    paths = output_paths(args, root_dir, "calibrate")
    samples = load_and_limit_samples(args, root_dir, args.eval)
    calibration = run_calibration_samples(
        samples,
        clients,
        config,
        output_file=paths["output_file"],
        calibration_output=paths["calibration_output"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        parallel_agents=args.parallel_agents,
        resume=args.resume,
        overwrite=args.overwrite,
        save_prompts=args.save_prompts,
        calibration_query_coordinator=args.calibration_query_coordinator,
        root_dir=str(root_dir),
        command=command_line(),
    )
    print(f"Wrote calibration records: {paths['output_file']}")
    print(f"Wrote calibration file: {paths['calibration_output']}")
    return calibration


def run_evaluate(args: argparse.Namespace, root_dir: Path, clients: Dict[str, VLLMClient]) -> Dict[str, Any]:
    config = build_config(args)
    config, calibration, calibration_path = load_calibration_if_needed(args, config)
    paths = output_paths(args, root_dir, "evaluate")
    samples = load_and_limit_samples(args, root_dir, args.eval)
    summary = run_evaluation_samples(
        samples,
        clients,
        config,
        output_file=paths["output_file"],
        summary_file=paths["summary_file"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        parallel_agents=args.parallel_agents,
        resume=args.resume,
        overwrite=args.overwrite,
        save_prompts=args.save_prompts,
        split=args.eval,
        mode="evaluate",
        calibration_file=calibration_path,
        calibration_num_samples=calibration.get("num_calibration_samples") if calibration else None,
    )
    print(f"Wrote evaluation records: {paths['output_file']}")
    print(f"Wrote summary file: {paths['summary_file']}")
    return summary


def run_calibrate_and_evaluate(
    args: argparse.Namespace,
    root_dir: Path,
    clients: Dict[str, VLLMClient],
) -> Dict[str, Any]:
    config = build_config(args)
    paths = output_paths(args, root_dir, "calibrate_and_evaluate")
    train_samples = load_and_limit_samples(argparse.Namespace(**{**vars(args), "eval": "train"}), root_dir, "train")
    calibration_records = paths["output_dir"] / "calibration_records.jsonl"
    evaluation_records = paths["output_dir"] / "evaluation_records.jsonl"
    calibration = run_calibration_samples(
        train_samples,
        clients,
        config,
        output_file=calibration_records,
        calibration_output=paths["calibration_output"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        parallel_agents=args.parallel_agents,
        resume=args.resume,
        overwrite=args.overwrite,
        save_prompts=args.save_prompts,
        calibration_query_coordinator=args.calibration_query_coordinator,
        root_dir=str(root_dir),
        command=command_line(),
    )
    config = apply_calibration_to_config(config, calibration)
    config.freeze_calibration = True
    test_samples = load_and_limit_samples(argparse.Namespace(**{**vars(args), "eval": "test"}), root_dir, "test")
    summary = run_evaluation_samples(
        test_samples,
        clients,
        config,
        output_file=evaluation_records,
        summary_file=paths["summary_file"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        parallel_agents=args.parallel_agents,
        resume=args.resume,
        overwrite=args.overwrite,
        save_prompts=args.save_prompts,
        split="test",
        mode="calibrate_and_evaluate",
        calibration_file=str(paths["calibration_output"]),
        calibration_num_samples=calibration.get("num_calibration_samples"),
    )
    print(f"Wrote calibration records: {calibration_records}")
    print(f"Wrote calibration file: {paths['calibration_output']}")
    print(f"Wrote evaluation records: {evaluation_records}")
    print(f"Wrote summary file: {paths['summary_file']}")
    return summary


def run_dry(args: argparse.Namespace, root_dir: Path) -> None:
    config = build_config(args)
    paths = output_paths(args, root_dir, "dry_run")
    samples = load_and_limit_samples(args, root_dir, args.eval)
    run_dry_run_samples(
        samples,
        config,
        output_file=paths["output_file"],
        overwrite=args.overwrite,
        save_prompts=args.save_prompts,
    )
    print(f"Wrote dry-run prompt records: {paths['output_file']}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
    if root_dir != ROOT_GUESS.resolve():
        print(
            f"Warning: script root is {ROOT_GUESS.resolve()}, but --root_dir is {root_dir}. "
            "Writes are still constrained to --root_dir.",
            file=sys.stderr,
        )

    if args.mode == "dry_run":
        run_dry(args, root_dir)
        return 0

    clients = build_clients(args)
    if args.mode == "calibrate":
        run_calibrate(args, root_dir, clients)
    elif args.mode == "evaluate":
        run_evaluate(args, root_dir, clients)
    elif args.mode == "calibrate_and_evaluate":
        run_calibrate_and_evaluate(args, root_dir, clients)
    else:
        raise ValueError(f"Unhandled mode: {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
