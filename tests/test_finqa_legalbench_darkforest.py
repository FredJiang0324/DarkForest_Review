import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.dataset_finqa_legalbench import build_domain_initial_prompt, load_domain_qa_samples  # noqa: E402
from darkforest.eval_finqa_legalbench import (  # noqa: E402
    DomainDarkForestConfig,
    apply_domain_calibration,
    build_prediction_record,
    compute_domain_belief,
)
from darkforest.parsing_finqa_legalbench import (  # noqa: E402
    ParsedDomainOutput,
    normalize_label,
    parse_domain_agent_output,
    repair_finqa_program,
)


def _out(agent, key, answer=None, program=None, invalid=False):
    return ParsedDomainOutput(
        agent_key=agent,
        raw_response="",
        parsed_answer=answer or key,
        normalized_answer=normalize_label(answer or key),
        parsed_program=program,
        cluster_key=key,
        invalid_parse=invalid,
        parse_method="test",
        error=None,
        latency_sec=0.0,
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


def test_parse_finqa_program_and_answer():
    raw = "Reasoning...\nProgram: subtract(153.7, 139.9), divide(#0, 139.9)\nFinal Answer: 9.9%"
    parsed = parse_domain_agent_output("finance_llama", raw, "finqa")
    assert parsed.parsed_program == "subtract(153.7, 139.9), divide(#0, 139.9)"
    assert parsed.parsed_answer == "9.9%"
    assert parsed.invalid_parse is False


def test_repair_finqa_infix_program_ratio():
    program = repair_finqa_program("subtract(153.7, 139.9) / 139.9 * 100")
    assert program == "subtract(153.7, 139.9), divide(#0, 139.9)"


def test_repair_finqa_assignment_program():
    program = repair_finqa_program("#0 = subtract(153.7, 139.9)\n#1 = divide(#0, 139.9)")
    assert program == "subtract(153.7, 139.9), divide(#0, 139.9)"


def test_parse_legalbench_allowed_choice():
    parsed = parse_domain_agent_output("saul", "Final Answer: No", "legalbench", answer_choices=["yes", "no"])
    assert parsed.normalized_answer == "no"
    assert parsed.cluster_key == "no"
    assert parsed.invalid_parse is False


def test_parse_legalbench_option_prefix():
    parsed = parse_domain_agent_output("saul", "Final Answer: Option C: Reasonable best efforts", "legalbench", answer_choices=["a", "b", "c"])
    assert parsed.normalized_answer == "c"
    assert parsed.cluster_key == "c"
    assert parsed.invalid_parse is False


def test_legalbench_prompt_strips_dangling_final_answer_marker():
    path_prompt = (
        "You are answering a legal benchmark question.\n\n"
        "Answer choices:\n- yes\n- no\n\n"
        "Question:\nDoes the clause apply?\n\n"
        "Final Answer:"
    )
    # Construct directly to keep the test independent of JSONL IO.
    from darkforest.dataset_finqa_legalbench import DomainQASample

    built = build_domain_initial_prompt(
        DomainQASample(
            idx=0,
            benchmark="legalbench",
            sample_id=0,
            prompt=path_prompt,
            question="Does the clause apply?",
            category="toy",
            gold_answer="yes",
            normalized_gold_answer="yes",
            answer_choices=["yes", "no"],
        )
    )
    assert built.count("Final Answer:") == 1
    assert "Final Answer:\n\nAllowed final answers" not in built


def test_legalbench_prediction_record_scores_guarded_answer():
    output = _out("coordinator", "yes", answer="yes")
    output.raw_response = "Final Answer: no"
    from darkforest.dataset_finqa_legalbench import DomainQASample

    record = build_prediction_record(
        DomainQASample(
            idx=0,
            benchmark="legalbench",
            sample_id=7,
            prompt="Question\nFinal Answer:",
            question="Question",
            category="toy",
            gold_answer="yes",
            normalized_gold_answer="yes",
            answer_choices=["yes", "no"],
        ),
        output,
    )
    assert record["prediction"] == "Final Answer: yes"


def test_domain_belief_support_pattern_calibration():
    config = DomainDarkForestConfig(
        benchmark="legalbench",
        agent_priors={"qwen": 0.6, "finance_llama": 0.3, "saul": 0.6},
        support_pattern_reliability={
            "qwen+saul": {"num": 5, "smoothed_accuracy": 0.9},
        },
        min_support_pattern_count=3,
    )
    outputs = {
        "qwen": _out("qwen", "yes"),
        "finance_llama": _out("finance_llama", "no"),
        "saul": _out("saul", "yes"),
    }
    belief = compute_domain_belief(outputs, config)
    assert belief["top_cluster_key"] == "yes"
    assert belief["answer_clusters"][0]["support_pattern"] == "qwen+saul"
    assert belief["answer_clusters"][0]["support_prior_source"] == "calibrated"
    assert abs(sum(row["posterior"] for row in belief["answer_clusters"]) - 1.0) < 1e-9


def test_apply_domain_calibration_overrides_priors():
    config = DomainDarkForestConfig(benchmark="finqa")
    calibration = {
        "learned_darkforest_params": {
            "agent_priors": {"qwen": 0.2, "finance_llama": 0.8, "saul": 0.4},
            "support_pattern_reliability": {"finance_llama": {"num": 10, "smoothed_accuracy": 0.7}},
            "min_support_pattern_count": 3,
        }
    }
    apply_domain_calibration(config, calibration, "calibration.json")
    assert config.params_source == "calibrated"
    assert config.agent_priors["finance_llama"] == 0.8
    assert config.support_pattern_reliability["finance_llama"]["smoothed_accuracy"] == 0.7


def test_load_domain_samples_from_jsonl(tmp_path):
    path = tmp_path / "legal.jsonl"
    row = {
        "id": 1,
        "task": "toy",
        "prompt": "Question...\nFinal Answer:",
        "answer": "Yes",
        "normalized_answer": "yes",
        "answer_choices": ["yes", "no"],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    samples = load_domain_qa_samples(path, "legalbench", "test")
    assert len(samples) == 1
    assert samples[0].sample_id == 1
    assert samples[0].answer_choices == ["yes", "no"]
