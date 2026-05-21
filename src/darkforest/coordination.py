from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .schemas import DarkForestConfig, ExposurePolicy, ParsedAgentOutput
from .utils import estimate_tokens


BASE_TASK_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    'Provide your step-by-step reasoning first, and then print "The answer is \\\\boxed{X}", '
    "where X is the final answer, at the end of your response."
)

INITIAL_AGENT_JSON_SUFFIX = (
    "Provide brief reasoning (2-3 key sentences), then output your final answer in JSON format:\n"
    '{"reasoning": "<brief reasoning>", "answer": "<a mathematical expression or number (e.g., \\"42\\" or \\"3/4\\")>", "confidence_level": "<a float between 0.0 and 1.0>"}\n'
    "Please strictly output in JSON format."
)


def apply_prompt_template(
    prompt: str,
    template: str = "raw",
) -> str:
    if template in {"raw", "none", ""}:
        return prompt
    if template == "qwen_chatml":
        return (
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if template == "mistral_inst":
        return f"<s>[INST] {prompt} [/INST]"
    raise ValueError(f"Unknown prompt template: {template}")


def prompt_for_agent(prompt: str, agent_key: str, config: DarkForestConfig) -> str:
    if config.api_style == "chat":
        return prompt
    template = config.prompt_templates.get(agent_key, "raw")
    return apply_prompt_template(prompt, template)


def build_initial_agent_prompt(question: str, initial_prompt_style: str = "freeform") -> str:
    base_prompt = BASE_TASK_PROMPT_TEMPLATE.replace("{question}", question)
    if initial_prompt_style == "freeform":
        return base_prompt
    if initial_prompt_style == "json":
        return f"{base_prompt}\n\n{INITIAL_AGENT_JSON_SUFFIX}"
    raise ValueError(f"Unknown initial prompt style: {initial_prompt_style}")


def _as_dict_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, ParsedAgentOutput):
        return output.to_dict()
    return dict(output)


def _truncate_exposed_text(text: str, policy: ExposurePolicy) -> str:
    char_limit = policy.max_peer_response_chars
    token_limit = policy.max_peer_response_tokens
    effective_limit = char_limit if char_limit is not None and char_limit >= 0 else None
    if token_limit is not None:
        token_char_limit = max(0, int(token_limit)) * 4
        effective_limit = token_char_limit if effective_limit is None else min(effective_limit, token_char_limit)
    if effective_limit is None or len(text) <= effective_limit:
        return text
    return text[:effective_limit] + "\n[truncated]"


def _belief_summary_text(belief_state: Dict[str, Any], expose_confidence: bool) -> str:
    clusters = belief_state.get("answer_clusters", [])
    top_posterior = float(belief_state.get("top_posterior") or 0.0)
    posterior_margin = float(belief_state.get("posterior_margin") or 0.0)
    num_distinct = int(belief_state.get("num_distinct_answers") or 0)
    high_uncertainty = bool(belief_state.get("high_uncertainty"))
    if num_distinct == 0:
        directive = "no valid candidate answers; solve from scratch."
    elif num_distinct >= 3 and top_posterior <= 0.45:
        directive = (
            "all candidate answers are in strong conflict; do not vote. "
            "solve independently, then use candidate reasoning only as checkable hints."
        )
    elif high_uncertainty or posterior_margin < 0.15:
        directive = (
            "belief is uncertain; audit every derivation and allow a singleton answer "
            "if its reasoning is the only valid one."
        )
    elif top_posterior >= 0.65:
        directive = (
            "one cluster has support; verify that cluster first, but reject it if the "
            "derivation has a concrete mathematical error."
        )
    else:
        directive = "use belief as a weak prior and choose only after independent verification."
    lines = [
        "DarkForest belief summary:",
        f"top_answer: {belief_state.get('top_answer')}",
        f"top_posterior: {belief_state.get('top_posterior')}",
        f"posterior_margin: {belief_state.get('posterior_margin')}",
        f"num_distinct_answers: {belief_state.get('num_distinct_answers')}",
        f"num_invalid_agent_parses: {belief_state.get('num_invalid_agent_parses')}",
        f"disagreement: {belief_state.get('disagreement')}",
        f"high_uncertainty: {belief_state.get('high_uncertainty')}",
        f"coordinator_directive: {directive}",
        "clusters:",
    ]
    for cluster in clusters:
        line = (
            f"- answer={cluster.get('normalized_answer')} "
            f"posterior={cluster.get('posterior')} "
            f"support={cluster.get('support_pattern')} "
            f"score={cluster.get('score')}"
        )
        if expose_confidence:
            line += f" mean_confidence={cluster.get('mean_confidence')}"
        lines.append(line)
    return "\n".join(lines)


def build_exposed_agent_content(
    agent_outputs: Mapping[str, Any],
    exposure_policy: ExposurePolicy,
    belief_state: Dict[str, Any] | None = None,
    agent_order: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    sections = []
    exposed_count = 0
    ordered_agents = list(agent_order or agent_outputs.keys())
    for agent_key in ordered_agents:
        if agent_key not in agent_outputs:
            continue
        exposed_count += 1
        output = _as_dict_output(agent_outputs[agent_key])
        lines = [
            f"Agent: {agent_key}",
            f"parsed_answer: {output.get('parsed_answer')}",
            f"normalized_answer: {output.get('normalized_answer')}",
            (
                "parse_status: "
                f"parse_method={output.get('parse_method')}; "
                f"invalid_parse={output.get('invalid_parse')}; "
                f"malformed_json={output.get('malformed_json')}"
            ),
        ]
        if exposure_policy.expose_confidence:
            lines.append(f"confidence: {output.get('confidence')}")
        if exposure_policy.expose_reasoning:
            reasoning = _truncate_exposed_text(str(output.get("parsed_reasoning") or ""), exposure_policy)
            lines.append(f"parsed_reasoning: {reasoning}")
        if exposure_policy.expose_full_responses:
            raw = _truncate_exposed_text(str(output.get("raw_response") or ""), exposure_policy)
            lines.append(f"raw_response: {raw}")
        sections.append("\n".join(lines))

    if exposure_policy.expose_belief_summary and belief_state is not None:
        sections.append(_belief_summary_text(belief_state, exposure_policy.expose_confidence))

    exposed_content = "\n\n".join(sections)
    metrics = {
        "num_agents": len(ordered_agents),
        "num_agent_outputs_exposed_to_coordinator": exposed_count,
        "cross_agent_input_chars": len(exposed_content),
        "cross_agent_input_tokens": estimate_tokens(exposed_content),
        "raw_full_response_exposed": bool(exposure_policy.expose_full_responses),
        "reasoning_exposed": bool(exposure_policy.expose_reasoning),
        "confidence_exposed": bool(exposure_policy.expose_confidence),
        "belief_summary_exposed": bool(exposure_policy.expose_belief_summary and belief_state is not None),
    }
    return exposed_content, metrics


def build_static_coordinator_prompt_with_exposure(
    question: str,
    agent_outputs: Mapping[str, Any],
    config: DarkForestConfig,
    belief_state: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    exposed_content, exposure_metrics = build_exposed_agent_content(
        agent_outputs,
        config.exposure_policy,
        belief_state,
        config.fixed_agents,
    )
    if config.coordinator_prompt_style == "darkforest_belief_audit":
        prompt = (
            "You are the DarkForest final coordinator for a MATH problem.\n"
            "You have three fixed agents' exposed answers, their exposed reasoning, and a compact "
            "Bayesian-style DarkForest belief summary.\n\n"
            "Core rule: do not vote. Use belief to decide how hard to audit, not to skip auditing.\n\n"
            "Belief-conditioned procedure:\n"
            "1. First solve the problem independently from the question until you have your own candidate answer.\n"
            "2. Read the DarkForest belief summary. If it says the candidates are in strong conflict or high uncertainty, "
            "treat every answer as suspect and rely on your independent derivation.\n"
            "3. Audit each exposed candidate derivation against the problem statement. Check algebra, definitions, edge cases, "
            "units, signs, branch choices, and exact forms.\n"
            "4. If your independent answer matches a candidate cluster, use the verified candidate answer.\n"
            "5. If a singleton candidate has the only valid derivation, choose it even if a majority disagrees.\n"
            "6. If a majority cluster has a concrete mathematical error, reject it.\n"
            "7. If all candidate answers are wrong or incomplete, synthesize any useful intermediate steps from the exposed reasoning "
            "and produce your independently corrected answer.\n"
            "8. Prefer exact mathematical forms over rounded decimals unless the problem asks for a decimal.\n\n"
            "Conclude exactly with:\n"
            "The answer is \\boxed{X}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Exposed candidate information and DarkForest belief:\n"
            f"{exposed_content}\n\n"
            "Final belief-audited solution:"
        )
    elif config.coordinator_prompt_style == "darkforest_audit":
        prompt = (
            "You are the DarkForest final coordinator for a MATH problem.\n"
            "Your job is not to vote. Your job is to solve, audit, and then choose or correct the final answer.\n\n"
            "Procedure:\n"
            "1. Solve the problem independently from the question.\n"
            "2. Audit each candidate answer and its exposed reasoning. A majority can be wrong; a singleton can be right.\n"
            "3. Treat the DarkForest belief summary as a prior, not proof.\n"
            "4. If a candidate derivation is valid, use its answer even if other agents disagree.\n"
            "5. If all candidates appear wrong or incomplete, ignore them and solve from scratch.\n"
            "6. Prefer exact mathematical forms over rounded decimals unless the problem asks for a decimal.\n\n"
            "Conclude exactly with:\n"
            "The answer is \\boxed{X}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Exposed candidate information:\n"
            f"{exposed_content}\n\n"
            "Final audited solution:"
        )
    elif config.coordinator_prompt_style == "goa_pooling":
        prompt = (
            "Synthesize these model responses into one final answer.\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Model responses:\n"
            f"{exposed_content}\n\n"
            "Produce an accurate, coherent answer integrating the best insights. "
            "Be critical - some information may be incorrect.\n"
            'Please conclude by printing "The answer is \\boxed{X}", '
            "where X is the final answer, at the end of your response."
        )
    else:
        prompt = (
            "You are given a MATH problem and several candidate solutions from fixed agents.\n"
            "Your task is to produce the final answer.\n"
            "Be critical: some candidate solutions may be wrong.\n"
            "Use only the useful parts of the candidate solutions.\n"
            "Conclude exactly with:\n"
            "The answer is \\boxed{X}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Candidate solutions:\n"
            f"{exposed_content}\n\n"
            "Final solution:"
        )
    return prompt, exposed_content, exposure_metrics


def build_static_coordinator_prompt(
    question: str,
    agent_outputs: Mapping[str, Any],
    config: DarkForestConfig,
    belief_state: Dict[str, Any],
) -> str:
    prompt, _, _ = build_static_coordinator_prompt_with_exposure(
        question,
        agent_outputs,
        config,
        belief_state,
    )
    return prompt
