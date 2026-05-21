#!/usr/bin/env python3
"""Evaluate FinQA-style predictions on the text-only sample.

Main metrics:
- Execution Accuracy: execute the predicted FinQA program and compare to the
  gold FinQA execution answer.
- Program Accuracy: compare predicted and gold programs with the official FinQA
  symbolic equivalence rule when available.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


ALL_OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
]
OP_RE = re.compile(r"\b(" + "|".join(re.escape(op) for op in ALL_OPS) + r")\s*\(")
PROGRAM_LINE_RE = re.compile(
    r"(?im)^\s*(?:Predicted\s+Program|Program)\s*:\s*(.+?)\s*$"
)
ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:Final\s+Answer|Answer)\s*:\s*(.+?)\s*$"
)


@dataclass
class EvaluatorFns:
    eval_program: Callable[[List[str], Sequence[Sequence[str]]], Tuple[int, Any]]
    equal_program: Callable[[List[str], List[str]], bool]
    source: str
    path: Optional[str]
    import_error: Optional[str] = None


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data
    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        records.append(item)
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def default_finqa_root() -> Path:
    return Path(__file__).resolve().parents[1] / "FinQA"


def load_official_evaluator(evaluator_path: Path) -> EvaluatorFns:
    if evaluator_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("official_finqa_evaluate", evaluator_path)
            if spec is None or spec.loader is None:
                raise ImportError("could not create module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            missing = [
                name
                for name in ("eval_program", "equal_program")
                if not hasattr(module, name)
            ]
            if missing:
                raise ImportError(f"missing functions: {missing}")
            return EvaluatorFns(
                eval_program=module.eval_program,
                equal_program=module.equal_program,
                source="official_import",
                path=str(evaluator_path),
            )
        except Exception as exc:  # pragma: no cover - exercised only when deps are absent.
            return EvaluatorFns(
                eval_program=fallback_eval_program,
                equal_program=fallback_equal_program,
                source="embedded_fallback",
                path=str(evaluator_path),
                import_error=repr(exc),
            )
    return EvaluatorFns(
        eval_program=fallback_eval_program,
        equal_program=fallback_equal_program,
        source="embedded_fallback",
        path=str(evaluator_path),
        import_error="official evaluator path not found",
    )


# Minimal fallback logic copied/adapted from:
# data/FinQA/code/evaluate/evaluate.py
# It is used only if the official evaluator cannot be imported.
def fallback_str_to_num(text: Any) -> Any:
    text = str(text).replace(",", "")
    try:
        return float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                return float(text) / 100.0
            except ValueError:
                return "n/a"
        if "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            return float(text)
        return "n/a"


def fallback_process_row(row_in: Sequence[str]) -> Any:
    row_out = []
    for num in row_in:
        num = str(num).replace("$", "").strip()
        num = num.split("(")[0].strip()
        num = fallback_str_to_num(num)
        if num == "n/a":
            return "n/a"
        row_out.append(num)
    return row_out


def fallback_eval_program(program: List[str], table: Sequence[Sequence[str]]) -> Tuple[int, Any]:
    invalid_flag = 0
    this_res: Any = "n/a"
    try:
        program = program[:-1]
        for ind, token in enumerate(program):
            if ind % 4 == 0 and token.strip("(") not in ALL_OPS:
                return 1, "n/a"
            if (ind + 1) % 4 == 0 and token != ")":
                return 1, "n/a"

        program_joined = "|".join(program)
        steps = program_joined.split(")")[:-1]
        res_dict: Dict[int, Any] = {}
        for ind, step in enumerate(steps):
            step = step.strip()
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if op in {"add", "subtract", "multiply", "divide", "exp", "greater"}:
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    arg1 = fallback_str_to_num(arg1)
                    if arg1 == "n/a":
                        invalid_flag = 1
                        break
                if "#" in arg2:
                    arg2 = res_dict[int(arg2.replace("#", ""))]
                else:
                    arg2 = fallback_str_to_num(arg2)
                    if arg2 == "n/a":
                        invalid_flag = 1
                        break
                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1 ** arg2
                elif op == "greater":
                    this_res = "yes" if arg1 > arg2 else "no"
                res_dict[ind] = this_res
            elif "table" in op:
                table_dict = {row[0]: row[1:] for row in table if row}
                if "#" in arg1:
                    invalid_flag = 1
                    break
                if arg1 not in table_dict:
                    invalid_flag = 1
                    break
                num_row = fallback_process_row(table_dict[arg1])
                if num_row == "n/a":
                    invalid_flag = 1
                    break
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)
                res_dict[ind] = this_res

        if this_res not in {"yes", "no", "n/a"}:
            this_res = round(this_res, 5)
    except Exception:
        invalid_flag = 1
    return invalid_flag, this_res


def fallback_equal_program(program1: List[str], program2: List[str]) -> bool:
    # This mirrors the official symbolic comparison in evaluate.py and requires sympy.
    try:
        from sympy import simplify
    except Exception:
        return "".join(program1) == "".join(program2)

    try:
        sym_map: Dict[str, str] = {}
        program1_body = "|".join(program1[:-1])
        steps1 = program1_body.split(")")[:-1]
        sym_ind = 0
        step_dict_1: Dict[int, str] = {}
        for ind, step in enumerate(steps1):
            step = step.strip()
            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            step_dict_1[ind] = step
            if "table" in op:
                if step not in sym_map:
                    sym_map[step] = f"a{sym_ind}"
                    sym_ind += 1
            else:
                for arg in (arg1, arg2):
                    if "#" not in arg and arg not in sym_map:
                        sym_map[arg] = f"a{sym_ind}"
                        sym_ind += 1

        for ind, token in enumerate(program2[:-1]):
            if ind % 4 == 0 and token.strip("(") not in ALL_OPS:
                return False
            if (ind + 1) % 4 == 0 and token != ")":
                return False

        program2_body = "|".join(program2[:-1])
        steps2 = program2_body.split(")")[:-1]
        step_dict_2: Dict[int, str] = {}
        for ind, step in enumerate(steps2):
            step = step.strip()
            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            step_dict_2[ind] = step
            if "table" in op:
                if step not in sym_map:
                    return False
            else:
                for arg in (arg1, arg2):
                    if "#" in arg:
                        if int(arg.strip("#")) >= ind:
                            return False
                    elif arg not in sym_map:
                        return False

        def symbol_recur(step: str, step_dict: Dict[int, str]) -> str:
            step = step.strip()
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            if "table" in op:
                return sym_map[step]
            arg1_part = symbol_recur(step_dict[int(arg1.replace("#", ""))], step_dict) if "#" in arg1 else sym_map[arg1]
            arg2_part = symbol_recur(step_dict[int(arg2.replace("#", ""))], step_dict) if "#" in arg2 else sym_map[arg2]
            if op == "add":
                return f"( {arg1_part} + {arg2_part} )"
            if op == "subtract":
                return f"( {arg1_part} - {arg2_part} )"
            if op == "multiply":
                return f"( {arg1_part} * {arg2_part} )"
            if op == "divide":
                return f"( {arg1_part} / {arg2_part} )"
            if op == "exp":
                return f"( {arg1_part} ** {arg2_part} )"
            if op == "greater":
                return f"( {arg1_part} > {arg2_part} )"
            raise ValueError(f"unsupported op: {op}")

        sym_prog1 = simplify(symbol_recur(steps1[-1], step_dict_1), evaluate=False)
        sym_prog2 = simplify(symbol_recur(steps2[-1], step_dict_2), evaluate=False)
        return sym_prog1 == sym_prog2
    except Exception:
        return False


def strip_program_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = text.strip("`").strip()
    text = re.sub(r"\bEOF\b\s*$", "", text, flags=re.IGNORECASE).strip()
    return text


def extract_labeled_program(text: str) -> Optional[str]:
    match = PROGRAM_LINE_RE.search(text)
    if not match:
        return None
    program = match.group(1).strip()
    program = re.split(r"(?i)\bFinal\s+Answer\s*:", program)[0].strip()
    return strip_program_text(program)


def find_matching_paren(text: str, open_index: int) -> Optional[int]:
    depth = 0
    for idx in range(open_index, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def split_top_level_args(text: str) -> List[str]:
    args = []
    depth = 0
    start = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:idx].strip())
            start = idx + 1
    args.append(text[start:].strip())
    return args


def validate_program_tokens(tokens: List[str]) -> bool:
    if len(tokens) < 5 or tokens[-1] != "EOF" or (len(tokens) - 1) % 4 != 0:
        return False
    body = tokens[:-1]
    for ind, token in enumerate(body):
        if ind % 4 == 0:
            if token.strip("(") not in ALL_OPS or not token.endswith("("):
                return False
        elif (ind + 1) % 4 == 0:
            if token != ")":
                return False
        elif token == "":
            return False
    return True


def tokenize_program_string(program_text: str) -> Optional[List[str]]:
    text = strip_program_text(program_text)
    labeled = extract_labeled_program(text)
    if labeled:
        text = labeled
    tokens: List[str] = []
    pos = 0
    while True:
        match = OP_RE.search(text, pos)
        if not match:
            break
        op = match.group(1)
        open_index = text.find("(", match.start())
        close_index = find_matching_paren(text, open_index)
        if close_index is None:
            return None
        args = split_top_level_args(text[open_index + 1 : close_index])
        if len(args) != 2 or not args[0] or not args[1]:
            return None
        tokens.extend([f"{op}(", args[0].strip(), args[1].strip(), ")"])
        pos = close_index + 1
    if not tokens:
        return None
    tokens.append("EOF")
    return tokens if validate_program_tokens(tokens) else None


def coerce_program_tokens(value: Any) -> Optional[List[str]]:
    if isinstance(value, list):
        tokens = [str(item).strip() for item in value if str(item).strip()]
        if tokens and tokens[-1] != "EOF":
            tokens.append("EOF")
        return tokens if validate_program_tokens(tokens) else None
    if isinstance(value, str):
        return tokenize_program_string(value)
    return None


def extract_program(prediction: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[str], str]:
    for key in ("program", "predicted"):
        if key in prediction and prediction[key] not in (None, ""):
            tokens = coerce_program_tokens(prediction[key])
            return tokens, tokens_to_program(tokens) if tokens else None, key
    for key in ("prediction", "response", "output", "text"):
        value = prediction.get(key)
        if isinstance(value, str) and value.strip():
            labeled = extract_labeled_program(value)
            tokens = tokenize_program_string(labeled or value)
            return tokens, tokens_to_program(tokens) if tokens else None, key
    return None, None, "missing"


def tokens_to_program(tokens: Optional[List[str]]) -> Optional[str]:
    if not tokens:
        return None
    steps = []
    for i in range(0, len(tokens) - 1, 4):
        steps.append(f"{tokens[i]}{tokens[i + 1]}, {tokens[i + 2]})")
    return ", ".join(steps)


def extract_final_answer(prediction: Dict[str, Any]) -> Tuple[Optional[str], str]:
    for key in ("final_answer", "answer"):
        if key in prediction and prediction[key] not in (None, ""):
            return str(prediction[key]).strip(), key
    for key in ("prediction", "response", "output", "text"):
        value = prediction.get(key)
        if isinstance(value, str):
            match = ANSWER_LINE_RE.search(value)
            if match:
                answer = match.group(1).strip().strip("`")
                return answer, key
    return None, "missing"


def parse_number_variants(value: Any, allow_percent_dual: bool) -> List[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        number = float(value)
        return [number] if math.isfinite(number) else []
    text = str(value).strip()
    if not text or text.lower() in {"yes", "no", "n/a", "nan", "none"}:
        return []
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    text = text.replace("$", "").replace(",", "").replace(" ", "")
    has_percent = text.endswith("%")
    if has_percent:
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return []
    if negative:
        number = -number
    if not math.isfinite(number):
        return []
    variants = [number]
    if has_percent or allow_percent_dual:
        variants.append(number / 100.0)
    return variants


def normalized_answer_string(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).strip().lower().replace("$", "").replace(",", ""))


def numbers_close(left: float, right: float, tol: float = 1e-4) -> bool:
    return abs(left - right) <= max(tol, tol * max(abs(left), abs(right), 1.0))


def answer_diagnostic_equal(pred_answer: Optional[str], gold_answer: Any, gold_exec: Any) -> bool:
    if pred_answer is None:
        return False
    if normalized_answer_string(pred_answer) == normalized_answer_string(gold_answer):
        return True
    gold_has_percent = "%" in str(gold_answer)
    pred_variants = parse_number_variants(pred_answer, allow_percent_dual=gold_has_percent)
    gold_variants = parse_number_variants(gold_answer, allow_percent_dual=gold_has_percent)
    gold_variants.extend(parse_number_variants(gold_exec, allow_percent_dual=False))
    return any(numbers_close(p, g) for p in pred_variants for g in gold_variants)


def official_equal_program(evaluator: EvaluatorFns, gold: List[str], pred: List[str]) -> bool:
    try:
        # The official function prints "structure error" for invalid predictions.
        with contextlib.redirect_stdout(io.StringIO()):
            return bool(evaluator.equal_program(gold, pred))
    except Exception:
        return False


def official_eval_program(
    evaluator: EvaluatorFns,
    pred: List[str],
    table: Sequence[Sequence[str]],
) -> Tuple[int, Any]:
    try:
        return evaluator.eval_program(pred, table)
    except Exception:
        return 1, "n/a"


def index_predictions(predictions: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for item in predictions:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            idx = int(item["id"])
        except (TypeError, ValueError):
            continue
        indexed[idx] = item
    return indexed


def evaluate(data: List[Dict[str, Any]], predictions: List[Dict[str, Any]], evaluator: EvaluatorFns) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pred_by_id = index_predictions(predictions)
    total = len(data)
    execution_correct = 0
    program_correct = 0
    invalid_program_parse = 0
    invalid_execution = 0
    invalid_answer_parse = 0
    answer_correct_diagnostic = 0
    details: List[Dict[str, Any]] = []

    for example in data:
        ex_id = int(example["id"])
        pred_record = pred_by_id.get(ex_id, {})
        gold_program = example.get("gold_program")
        gold_tokens = coerce_program_tokens(gold_program)
        if gold_tokens is None:
            raise ValueError(f"Gold program for id={ex_id} is unparsable: {gold_program}")

        pred_tokens, pred_program_text, program_source = extract_program(pred_record)
        final_answer, answer_source = extract_final_answer(pred_record)
        final_answer_parseable = bool(parse_number_variants(final_answer, allow_percent_dual=False))
        if not final_answer_parseable:
            invalid_answer_parse += 1

        pred_execution = "n/a"
        execution_ok = False
        program_ok = False
        execution_invalid = False
        parse_invalid = pred_tokens is None

        if pred_tokens is None:
            invalid_program_parse += 1
        else:
            invalid_flag, pred_execution = official_eval_program(evaluator, pred_tokens, [])
            if invalid_flag != 0:
                invalid_execution += 1
                execution_invalid = True
            else:
                gold_exec = example.get("gold_execution_answer", example.get("gold_answer"))
                execution_ok = pred_execution == gold_exec
                if execution_ok:
                    execution_correct += 1
            program_ok = official_equal_program(evaluator, gold_tokens, pred_tokens)
            if program_ok:
                program_correct += 1

        answer_diag_ok = answer_diagnostic_equal(
            final_answer,
            example.get("gold_answer"),
            example.get("gold_execution_answer"),
        )
        if answer_diag_ok:
            answer_correct_diagnostic += 1

        details.append(
            {
                "id": ex_id,
                "raw_id": example.get("raw_id"),
                "source_split": example.get("source_split"),
                "source_index": example.get("source_index"),
                "program_source": program_source,
                "answer_source": answer_source,
                "predicted_program": pred_program_text,
                "predicted_program_tokens": pred_tokens,
                "predicted_execution": pred_execution,
                "predicted_final_answer": final_answer,
                "gold_program": gold_program,
                "gold_execution_answer": example.get("gold_execution_answer"),
                "gold_answer": example.get("gold_answer"),
                "execution_correct": execution_ok,
                "program_correct": program_ok,
                "invalid_program_parse": parse_invalid,
                "invalid_execution": execution_invalid,
                "invalid_answer_parse": not final_answer_parseable,
                "answer_correct_diagnostic": answer_diag_ok,
            }
        )

    metrics = {
        "total": total,
        "execution_correct": execution_correct,
        "execution_accuracy": execution_correct / total if total else 0.0,
        "program_correct": program_correct,
        "program_accuracy": program_correct / total if total else 0.0,
        "invalid_program_parse": invalid_program_parse,
        "invalid_execution": invalid_execution,
        "invalid_answer_parse": invalid_answer_parse,
        "answer_correct_diagnostic": answer_correct_diagnostic,
        "answer_accuracy_diagnostic": answer_correct_diagnostic / total if total else 0.0,
        "official_evaluator_source": evaluator.source,
        "official_evaluator_path": evaluator.path,
        "official_evaluator_import_error": evaluator.import_error,
    }
    return metrics, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finqa_root", type=Path, default=default_finqa_root())
    parser.add_argument("--official_evaluator", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_records(args.data)
    predictions = load_records(args.pred)
    evaluator_path = args.official_evaluator or (
        args.finqa_root / "code" / "evaluate" / "evaluate.py"
    )
    evaluator = load_official_evaluator(evaluator_path)
    metrics, details = evaluate(data, predictions, evaluator)
    if args.output:
        write_jsonl(args.output, details)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
