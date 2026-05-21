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

from darkforest.dataset_finqa_legalbench import (  # noqa: E402
    DomainQASample,
    build_domain_initial_prompt,
    default_domain_paths,
    load_domain_qa_samples,
)
from darkforest.eval_finqa_legalbench import (  # noqa: E402
    DOMAIN_AGENTS,
    DomainDarkForestConfig,
    aggregate_domain_summary,
    apply_domain_belief_guardrail,
    apply_domain_calibration,
    build_prediction_record,
    choose_domain_majority_vote,
    compute_domain_belief,
    estimate_domain_calibration,
    load_finqa_metric_module,
    make_domain_metrics,
    query_domain_coordinator,
    query_domain_initial_agents,
    read_calibration,
    run_external_evaluator,
    score_and_enrich_output,
    write_records_jsonl,
)
from darkforest.llm_client import VLLMClient  # noqa: E402
from darkforest.llm_client import extract_completion_text  # noqa: E402
from darkforest.schemas import LLMResponse  # noqa: E402
from darkforest.utils import command_line, normalize_usage, parse_bool, read_json, write_json  # noqa: E402

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None


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


def _limit(samples: Sequence[DomainQASample], limit: Optional[int]) -> List[DomainQASample]:
    if limit is None:
        return list(samples)
    return list(samples)[: max(0, int(limit))]


def _read_records_by_idx(path: Path) -> Tuple[List[Dict[str, Any]], set[int]]:
    if not path.exists():
        return [], set()
    records: List[Dict[str, Any]] = []
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


class TemplateCompletionClient:
    _tokenizers: Dict[str, Any] = {}

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        tokenizer_path: Optional[str],
        timeout_sec: float,
        max_retries: int,
        backend: str,
    ) -> None:
        self.endpoint = VLLMClient._resolve_endpoint(endpoint, "completions")
        self.model_name = model_name
        self.tokenizer_path = tokenizer_path or model_name
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backend = backend
        self.chat_client = VLLMClient(
            endpoint,
            model_name,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            api_style="chat",
        )

    def complete(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        if self.backend == "chat_api":
            return self.chat_client.complete(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed)
        rendered_prompt = self._render_prompt(prompt)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": rendered_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        last_error: Optional[str] = None
        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                if requests is None:
                    raise RuntimeError("requests is required for TemplateCompletionClient")
                response = requests.post(self.endpoint, json=payload, timeout=self.timeout_sec)
                response.raise_for_status()
                response_json = response.json()
                text = extract_completion_text(response_json)
                return LLMResponse(
                    text=text,
                    latency_sec=time.perf_counter() - start,
                    usage=normalize_usage(response_json.get("usage"), rendered_prompt, text),
                    error=None,
                    raw_json=response_json,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        return LLMResponse(
            text="",
            latency_sec=time.perf_counter() - start,
            usage=normalize_usage(None, rendered_prompt, ""),
            error=last_error,
        )

    def _render_prompt(self, prompt: str) -> str:
        if self.backend == "raw":
            return prompt
        if AutoTokenizer is None:
            return self._chatml_fallback(prompt)
        tokenizer = self._get_tokenizer(self.tokenizer_path)
        messages = [{"role": "user", "content": prompt}]
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._chatml_fallback(prompt)

    @classmethod
    def _get_tokenizer(cls, path: str) -> Any:
        if path not in cls._tokenizers:
            cls._tokenizers[path] = AutoTokenizer.from_pretrained(path)
        return cls._tokenizers[path]

    @staticmethod
    def _chatml_fallback(prompt: str) -> str:
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


def _make_clients(args: argparse.Namespace) -> Dict[str, Any]:
    if args.prompt_template_backend == "raw_legacy":
        return {
            "qwen": VLLMClient(
                args.qwen_endpoint,
                args.qwen_model_name,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                api_style=args.api_style,
            ),
            "finance_llama": VLLMClient(
                args.finance_endpoint,
                args.finance_model_name,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                api_style=args.api_style,
            ),
            "saul": VLLMClient(
                args.saul_endpoint,
                args.saul_model_name,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                api_style=args.api_style,
            ),
        }
    return {
        "qwen": TemplateCompletionClient(
            args.qwen_endpoint,
            args.qwen_model_name,
            args.qwen_tokenizer_path,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            backend=args.prompt_template_backend,
        ),
        "finance_llama": TemplateCompletionClient(
            args.finance_endpoint,
            args.finance_model_name,
            args.finance_tokenizer_path,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            backend=args.prompt_template_backend,
        ),
        "saul": TemplateCompletionClient(
            args.saul_endpoint,
            args.saul_model_name,
            args.saul_tokenizer_path,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            backend=args.prompt_template_backend,
        ),
    }


def _make_config(args: argparse.Namespace) -> DomainDarkForestConfig:
    priors = {agent: 1.0 for agent in DOMAIN_AGENTS}
    if args.agent_priors:
        for item in args.agent_priors.split(","):
            key, raw = item.split("=", 1)
            key = key.strip()
            if key not in priors:
                raise ValueError(f"Unknown agent in --agent_priors: {key}")
            priors[key] = float(raw)
    decision_priority = [item.strip() for item in args.decision_priority.split(",") if item.strip()]
    for agent in decision_priority:
        if agent not in DOMAIN_AGENTS:
            raise ValueError(f"Unknown agent in --decision_priority: {agent}")
    return DomainDarkForestConfig(
        benchmark=args.benchmark,
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
        decision_policy=args.decision_policy,
        decision_priority=decision_priority,
        belief_guardrail=args.belief_guardrail,
        belief_guardrail_min_posterior=args.belief_guardrail_min_posterior,
        belief_guardrail_min_margin=args.belief_guardrail_min_margin,
        freeze_calibration=args.freeze_calibration,
    )


def _load_calibration_if_needed(
    args: argparse.Namespace,
    config: DomainDarkForestConfig,
    root_dir: Path,
) -> Tuple[DomainDarkForestConfig, Optional[Dict[str, Any]], Optional[str]]:
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
    return apply_domain_calibration(config, calibration, str(path)), calibration, str(path)


def _finqa_resources(benchmark: str) -> Tuple[Optional[Any], Optional[Any]]:
    if benchmark != "finqa":
        return None, None
    module = load_finqa_metric_module()
    evaluator = module.load_official_evaluator(ROOT / "data/FinQA/code/evaluate/evaluate.py")
    return module, evaluator


def _correct_cluster_keys(
    agent_outputs: Dict[str, Any],
    agent_correct: Dict[str, bool],
    config: DomainDarkForestConfig,
) -> List[str]:
    keys = []
    for agent in config.fixed_agents:
        output = agent_outputs.get(agent)
        if output and agent_correct.get(agent) and output.cluster_key:
            keys.append(str(output.cluster_key))
    return sorted(set(keys))


def run_dry_run(samples: Sequence[DomainQASample], output_dir: Path) -> None:
    previews = []
    for sample in samples[:5]:
        previews.append(
            {
                "idx": sample.idx,
                "id": sample.sample_id,
                "benchmark": sample.benchmark,
                "category": sample.category,
                "prompt": build_domain_initial_prompt(sample),
            }
        )
    write_json(output_dir / "dry_run_prompts.json", {"prompts": previews})
    print(f"Wrote dry-run prompt preview: {output_dir / 'dry_run_prompts.json'}")


def run_calibration(
    calibration_samples: Sequence[DomainQASample],
    clients: Dict[str, VLLMClient],
    config: DomainDarkForestConfig,
    args: argparse.Namespace,
    output_dir: Path,
    calibration_output: Path,
) -> Dict[str, Any]:
    finqa_module, finqa_evaluator = _finqa_resources(args.benchmark)
    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(calibration_samples):
        print(f"[calibrate] {idx + 1}/{len(calibration_samples)} {sample.category} id={sample.sample_id}", flush=True)
        agent_outputs, initial_prompt, initial_latency, agent_correct, score_details = query_domain_initial_agents(
            sample,
            clients,
            config,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            parallel_agents=args.parallel_agents,
            finqa_module=finqa_module,
            finqa_evaluator=finqa_evaluator,
        )
        belief = compute_domain_belief(agent_outputs, config, sample)
        correct_keys = _correct_cluster_keys(agent_outputs, agent_correct, config)
        record = {
            "idx": idx,
            "id": sample.sample_id,
            "benchmark": sample.benchmark,
            "category": sample.category,
            "question": sample.question,
            "prompt": sample.prompt,
            "gold_answer": sample.gold_answer,
            "normalized_gold_answer": sample.normalized_gold_answer,
            "gold_execution_answer": sample.gold_execution_answer,
            "gold_program": sample.gold_program,
            "agents": {agent: agent_outputs[agent].to_dict() for agent in config.fixed_agents},
            "agent_correct": agent_correct,
            "agent_score_details": score_details,
            "correct_cluster_keys": correct_keys,
            "darkforest_belief": belief,
            "metrics": make_domain_metrics(
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

    calibration = estimate_domain_calibration(
        records,
        config,
        seed=args.seed,
        command=command_line(),
        root_dir=str(args.root_dir),
        calibration_split="calibration",
    )
    write_records_jsonl(records, output_dir / "calibration_records.jsonl")
    write_json(calibration_output, calibration)
    print(f"Wrote calibration: {calibration_output}")
    return calibration


def run_evaluation(
    eval_samples: Sequence[DomainQASample],
    clients: Dict[str, VLLMClient],
    config: DomainDarkForestConfig,
    args: argparse.Namespace,
    output_dir: Path,
    summary_file: Path,
    calibration_file: Optional[str],
    calibration_num_samples: Optional[int],
    eval_data_path: Path,
) -> Dict[str, Any]:
    finqa_module, finqa_evaluator = _finqa_resources(args.benchmark)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "evaluation_records.jsonl"
    pred_file = output_dir / "predictions.jsonl"
    details_file = output_dir / "eval_details.jsonl"
    scoring_data_file = output_dir / "scoring_data.jsonl"
    if args.overwrite:
        for path in (records_path, pred_file, details_file, scoring_data_file, summary_file):
            if path.exists():
                path.unlink()
    if records_path.exists() and not args.resume:
        raise FileExistsError(
            f"{records_path} already exists. Use --resume true to continue or --overwrite true to restart."
        )
    records, completed_indices = _read_records_by_idx(records_path) if args.resume else ([], set())
    start_wall = time.perf_counter()
    for idx, sample in enumerate(eval_samples):
        if idx in completed_indices:
            continue
        print(f"[evaluate] {idx + 1}/{len(eval_samples)} {sample.category} id={sample.sample_id}", flush=True)
        agent_outputs, initial_prompt, initial_latency, agent_correct, score_details = query_domain_initial_agents(
            sample,
            clients,
            config,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            parallel_agents=args.parallel_agents,
            finqa_module=finqa_module,
            finqa_evaluator=finqa_evaluator,
        )
        belief = compute_domain_belief(agent_outputs, config, sample)
        coordinator_output = None
        coordinator_prompt = None
        exposed_content = ""
        guardrail_info = {
            "policy": config.belief_guardrail,
            "applied": False,
            "reason": "not_applicable",
        }
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

        if config.decision_policy == "majority_vote":
            final_output = choose_domain_majority_vote(agent_outputs, config)
            if final_output is None:
                final_output = next(iter(agent_outputs.values()))
            correct, final_score_detail = score_and_enrich_output(
                sample,
                final_output,
                sample.benchmark,
                finqa_module=finqa_module,
                finqa_evaluator=finqa_evaluator,
            )
        elif config.decision_policy == "coordinator":
            coordinator_output, coordinator_prompt, exposed_content, exposure_metrics, _, coordinator_score = (
                query_domain_coordinator(
                    sample,
                    clients,
                    config,
                    agent_outputs,
                    belief,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=args.seed + 100000 + idx,
                    finqa_module=finqa_module,
                    finqa_evaluator=finqa_evaluator,
                )
            )
            final_output, guardrail_info, correct, final_score_detail = apply_domain_belief_guardrail(
                coordinator_output,
                belief,
                config,
                sample,
                finqa_module=finqa_module,
                finqa_evaluator=finqa_evaluator,
            )
            final_score_detail.setdefault("coordinator_score_detail", coordinator_score)
        else:
            raise ValueError(f"Unknown decision_policy: {config.decision_policy}")

        invalid_parse = bool(final_output.invalid_parse)
        prediction_record = build_prediction_record(sample, final_output)
        record = {
            "idx": idx,
            "id": sample.sample_id,
            "benchmark": sample.benchmark,
            "category": sample.category,
            "question": sample.question,
            "prompt": sample.prompt,
            "gold_answer": sample.gold_answer,
            "normalized_gold_answer": sample.normalized_gold_answer,
            "gold_execution_answer": sample.gold_execution_answer,
            "gold_program": sample.gold_program,
            "final_response": final_output.raw_response,
            "parsed_answer": final_output.parsed_answer,
            "normalized_parsed_answer": final_output.normalized_answer,
            "parsed_program": final_output.parsed_program,
            "cluster_key": final_output.cluster_key,
            "correct": bool(correct),
            "invalid_parse": invalid_parse,
            "prediction_record": prediction_record,
            "agents": {agent: agent_outputs[agent].to_dict() for agent in config.fixed_agents},
            "agent_correct": agent_correct,
            "agent_score_details": score_details,
            "final_score_detail": final_score_detail,
            "coordination": {
                "method": config.coordination_method,
                "design_name": config.design_name,
                "coordinator_model": config.coordinator_model,
                "fixed_agents": list(config.fixed_agents),
                "decision_policy": config.decision_policy,
                "darkforest_belief": belief,
                "coordinator_prompt": coordinator_prompt,
                "exposed_content": exposed_content,
                "coordinator_response": coordinator_output.raw_response if coordinator_output is not None else None,
                "coordinator_parsed": coordinator_output.to_dict() if coordinator_output is not None else None,
                "belief_guardrail": guardrail_info,
            },
            "metrics": make_domain_metrics(
                correct=bool(correct),
                invalid_parse=invalid_parse,
                agent_outputs=agent_outputs,
                coordinator_output=coordinator_output,
                initial_latency_sec=initial_latency,
                exposure_metrics=exposure_metrics,
                config=config,
            ),
        }
        if args.save_prompts:
            record["prompts"] = {
                "initial_agent_prompt": initial_prompt,
                "coordinator_prompt": coordinator_prompt,
            }
        records.append(record)
        _append_record(records_path, record)

    total_wall = time.perf_counter() - start_wall
    records = sorted(_read_records_by_idx(records_path)[0], key=lambda row: int(row["idx"]))
    predictions = [record.get("prediction_record", {"id": record.get("id"), "prediction": ""}) for record in records]
    write_records_jsonl(predictions, pred_file)
    write_records_jsonl([sample.raw_sample for sample in eval_samples], scoring_data_file)
    evaluator_metrics = run_external_evaluator(args.benchmark, scoring_data_file, pred_file, details_file)
    summary = aggregate_domain_summary(
        records,
        benchmark=args.benchmark,
        mode=args.mode,
        config=config,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        calibration_file=calibration_file,
        calibration_num_samples=calibration_num_samples,
        total_wall_time_sec=total_wall,
        evaluator_metrics=evaluator_metrics,
    )
    summary["prediction_file"] = str(pred_file)
    summary["details_file"] = str(details_file)
    summary["scoring_data_file"] = str(scoring_data_file)
    summary["source_eval_data_path"] = str(eval_data_path)
    summary["records_file"] = str(records_path)
    write_json(summary_file, summary)
    print(f"Wrote records: {records_path}")
    print(f"Wrote predictions: {pred_file}")
    print(f"Wrote summary: {summary_file}")
    if args.benchmark == "finqa":
        print(
            f"FinQA execution accuracy: {summary['execution_accuracy_percent']:.2f}% "
            f"({summary['execution_correct']}/{summary['num_samples']})"
        )
    else:
        print(
            f"LegalBench exact match: {summary['exact_match_accuracy_percent']:.2f}% "
            f"({summary['num_correct']}/{summary['num_samples']})"
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DarkForest on FinQA text-only and LegalBench samples.")
    parser.add_argument("--root_dir", default=str(ROOT))
    parser.add_argument("--benchmark", choices=["finqa", "legalbench"], required=True)
    parser.add_argument("--mode", choices=["calibrate", "evaluate", "calibrate_and_evaluate", "dry_run"], required=True)
    parser.add_argument("--calibration_data_path", default=None)
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--summary_file", default=None)
    parser.add_argument("--calibration_output", default=None)
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
    parser.add_argument("--finance_endpoint", default="http://localhost:8004/v1/completions")
    parser.add_argument("--saul_endpoint", default="http://localhost:8005/v1/completions")
    parser.add_argument("--qwen_model_name", default="qwen")
    parser.add_argument("--finance_model_name", default="finance_llama")
    parser.add_argument("--saul_model_name", default="saul")
    parser.add_argument("--qwen_tokenizer_path", default=None)
    parser.add_argument("--finance_tokenizer_path", default=None)
    parser.add_argument("--saul_tokenizer_path", default=None)
    parser.add_argument("--api_style", choices=["chat", "completions"], default="completions")
    parser.add_argument(
        "--prompt_template_backend",
        choices=["goa", "raw", "chat_api", "raw_legacy"],
        default="goa",
        help=(
            "goa renders prompts with each model tokenizer chat_template and the GoA ChatML fallback "
            "before calling /v1/completions. raw sends the prompt directly. chat_api uses /v1/chat/completions. "
            "raw_legacy uses the shared legacy VLLMClient path."
        ),
    )

    parser.add_argument("--coordinator_model", choices=DOMAIN_AGENTS, default="qwen")
    parser.add_argument("--coordination_method", default="darkforest")
    parser.add_argument("--coordination_rounds", type=int, default=1)
    parser.add_argument("--decision_policy", choices=["coordinator", "majority_vote"], default="coordinator")
    parser.add_argument("--decision_priority", default="qwen,finance_llama,saul")
    parser.add_argument("--expose_belief_summary", type=parse_bool, default=True)
    parser.add_argument("--expose_full_responses", type=parse_bool, default=False)
    parser.add_argument("--expose_reasoning", type=parse_bool, default=False)
    parser.add_argument("--max_peer_response_chars", type=int, default=3000)
    parser.add_argument("--max_reasoning_chars", type=int, default=900)
    parser.add_argument("--belief_guardrail", choices=["none", "trust_supported_cluster"], default="trust_supported_cluster")
    parser.add_argument("--belief_guardrail_min_posterior", type=float, default=0.66)
    parser.add_argument("--belief_guardrail_min_margin", type=float, default=0.20)
    parser.add_argument("--agent_priors", default=None)
    parser.add_argument("--accept_threshold", type=float, default=0.75)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.60)
    parser.add_argument("--min_support_pattern_count", type=int, default=3)

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
        raise NotImplementedError("FinQA/LegalBench runner supports --coordination_rounds 1 only.")
    if args.coordination_method != "darkforest":
        raise ValueError("FinQA/LegalBench runner implements only --coordination_method darkforest.")

    calibration_default, eval_default = default_domain_paths(args.root_dir, args.benchmark)
    calibration_path = _read_path(args.root_dir, args.calibration_data_path, calibration_default)
    eval_path = _read_path(args.root_dir, args.eval_data_path, eval_default)
    calibration_samples = load_domain_qa_samples(calibration_path, args.benchmark, split="calibration")
    eval_samples = load_domain_qa_samples(eval_path, args.benchmark, split="test")

    if args.limit_samples is not None:
        calibration_samples = _limit(calibration_samples, args.limit_samples)
        eval_samples = _limit(eval_samples, args.limit_samples)
    calibration_samples = _limit(calibration_samples, args.limit_calibration_samples)
    eval_samples = _limit(eval_samples, args.limit_eval_samples)

    dataset_name = "FinQA" if args.benchmark == "finqa" else "LegalBench"
    output_default = f"outputs/{dataset_name}/darkforest_{args.benchmark}_qwen_finance_saul"
    calibration_default_out = f"outputs/{dataset_name}/darkforest_calibration_qwen_finance_saul/calibration.json"
    output_dir = _path_under_root(args.root_dir, args.output_dir, output_default)
    summary_file = _path_under_root(args.root_dir, args.summary_file, str(output_dir.relative_to(args.root_dir) / "summary.json"))
    calibration_output = _path_under_root(args.root_dir, args.calibration_output, calibration_default_out)

    config = _make_config(args)
    print(f"Root: {args.root_dir}")
    print(f"{dataset_name} calibration: {calibration_path} ({len(calibration_samples)} samples)")
    print(f"{dataset_name} eval: {eval_path} ({len(eval_samples)} samples)")
    print(f"Fixed agents: {config.fixed_agents}; coordinator={config.coordinator_model}; policy={config.decision_policy}")

    if args.mode == "dry_run":
        run_dry_run(eval_samples, output_dir)
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
        config = apply_domain_calibration(config, calibration, str(calibration_output))

    calibration_num_samples = int(calibration.get("num_calibration_samples", 0)) if calibration else None
    run_evaluation(
        eval_samples,
        clients,
        config,
        args,
        output_dir,
        summary_file,
        calibration_file=calibration_file,
        calibration_num_samples=calibration_num_samples,
        eval_data_path=eval_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
