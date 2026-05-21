import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkforest.metrics import (  # noqa: E402
    aggregate_evaluation_summary,
    build_sample_metrics,
    llm_call_count_for_method,
)
from darkforest.schemas import DarkForestConfig  # noqa: E402


def usage(i, o):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def test_default_darkforest_call_count_is_four():
    metrics = build_sample_metrics(
        correct=True,
        invalid_parse=False,
        initial_usages=[usage(1, 1), usage(1, 1), usage(1, 1)],
        coordination_usage=usage(2, 2),
        initial_latency_sec=3.0,
        coordination_latency_sec=1.0,
        individual_initial_latencies={},
        exposure_metrics={"cross_agent_input_tokens": 3, "cross_agent_input_chars": 12, "num_agent_outputs_exposed_to_coordinator": 3},
    )
    assert metrics["llm_calls_total"] == 4
    assert metrics["llm_calls_by_phase"]["initial_agents"] == 3
    assert metrics["llm_calls_by_phase"]["coordination"] == 1


def test_majority_vote_call_count_is_three():
    assert llm_call_count_for_method("majority_vote") == 3


def test_aggregate_summary_metrics_and_percentiles():
    config = DarkForestConfig()
    records = []
    for idx, latency in enumerate([1.0, 2.0, 3.0]):
        records.append(
            {
                "correct": idx < 2,
                "invalid_parse": idx == 2,
                "metrics": {
                    "llm_calls_total": 4,
                    "input_tokens_total": 10,
                    "output_tokens_total": 5,
                    "total_tokens": 15,
                    "latency_sec_total": latency,
                    "latency_by_phase_sec": {"initial_agents": latency / 2, "verification": 0.0, "coordination": latency / 2},
                    "exposure_metrics": {
                        "cross_agent_input_tokens": 3,
                        "cross_agent_input_chars": 12,
                        "num_agent_outputs_exposed_to_coordinator": 3,
                    },
                },
            }
        )
    summary = aggregate_evaluation_summary(records, "test", "evaluate", config, 0.7, 800, 0, None, None)
    assert summary["em_percent"] == 100.0 * 2 / 3
    assert summary["avg_total_tokens_per_sample"] == 15
    assert summary["p90_latency_sec_per_sample"] >= summary["median_latency_sec_per_sample"]
    assert summary["p95_latency_sec_per_sample"] >= summary["p90_latency_sec_per_sample"]


def test_aggregate_summary_skips_missing_gold_records_for_scoring():
    config = DarkForestConfig(
        support_pattern_reliability={"qwen": {"num": 20, "num_correct": 10, "accuracy": 0.5, "smoothed_accuracy": 0.5}},
    )
    config.params_source = "calibrated"
    config.parameter_sources = {"agent_priors": "calibrated"}
    records = [
        {
            "correct": True,
            "scored": True,
            "invalid_parse": False,
            "metrics": {
                "llm_calls_total": 4,
                "input_tokens_total": 10,
                "output_tokens_total": 5,
                "total_tokens": 15,
                "latency_sec_total": 1.0,
                "latency_by_phase_sec": {"initial_agents": 0.5, "verification": 0.0, "coordination": 0.5},
                "exposure_metrics": {
                    "cross_agent_input_tokens": 3,
                    "cross_agent_input_chars": 12,
                    "num_agent_outputs_exposed_to_coordinator": 3,
                },
            },
        },
        {
            "correct": None,
            "scored": False,
            "invalid_parse": True,
            "metrics": {
                "llm_calls_total": 4,
                "input_tokens_total": 100,
                "output_tokens_total": 50,
                "total_tokens": 150,
                "latency_sec_total": 10.0,
                "latency_by_phase_sec": {"initial_agents": 5.0, "verification": 0.0, "coordination": 5.0},
                "exposure_metrics": {
                    "cross_agent_input_tokens": 30,
                    "cross_agent_input_chars": 120,
                    "num_agent_outputs_exposed_to_coordinator": 3,
                },
            },
        },
    ]
    summary = aggregate_evaluation_summary(records, "test", "evaluate", config, 0.7, 800, 0, "cal.json", 10)
    assert summary["num_samples"] == 1
    assert summary["num_records_processed"] == 2
    assert summary["num_unscored"] == 1
    assert summary["em_percent"] == 100.0
    assert summary["learned_darkforest_params_snapshot"]["support_pattern_reliability"]["qwen"]["num"] == 20
    assert summary["darkforest_config"]["parameter_sources"]["agent_priors"] == "calibrated"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
