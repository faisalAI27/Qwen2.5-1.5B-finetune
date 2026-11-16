"""Explicit final-answer extraction and pinned local mathematical equivalence."""

import importlib.metadata
import re

from s1small.extraction import extract_answer


GRADER_SETTINGS = {"version": "local_math_verify_v1", "parsing_timeout": 5, "verification_timeout": 5,
                   "float_rounding": 6, "numeric_precision": 15, "strict": True,
                   "allow_set_relation_comp": False, "fallback_mode": "no_fallback",
                   "extraction_mode": "first_match", "prediction_source": "last_balanced_box_or_explicit_answer_line",
                   "bare_trailing_number_guess": False,
                   "structured_readout_context": "A frozen shared injected answer label may precede the generated answer body for extraction; origins and raw generated text are retained separately."}


def extract_final_answer(text):
    boxes = list(re.finditer(r"\\boxed\s*\{", text))
    lines = list(re.finditer(r"(?im)^[ \t]*(?:final[ \t]+)?answer[ \t]*:[ \t]*([^\n]*)", text))
    if lines and (not boxes or lines[-1].start() > boxes[-1].start()):
        value = lines[-1].group(1).strip()
        if not value:
            following = [line.strip() for line in text[lines[-1].end():].splitlines() if line.strip()]
            value = following[0] if following else ""
        return {"extracted_answer": value or None,
                "extraction_status": "explicit_answer_line" if value else "incomplete_final_answer"}
    result = extract_answer(text)
    return {"extracted_answer": result["extracted_answer"], "extraction_status": result["extraction_status"]}


def strip_math_environment(text):
    value = text.strip()
    for left, right in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
            return value[len(left):-len(right)].strip()
    return value


def equivalent(gold, prediction):
    """Use gold first, with explicit parser/verification limits and no eval()."""
    if prediction is None:
        return {"is_correct": False, "grading_status": "no_extracted_answer"}
    expected, actual = strip_math_environment(gold), strip_math_environment(prediction)
    if expected and expected == actual:
        return {"is_correct": True, "grading_status": "exact_literal_match"}
    from math_verify import LatexExtractionConfig, parse, verify
    config = [LatexExtractionConfig(try_extract_without_anchor=True, boxed_match_priority=0)]
    try:
        gold_parsed = parse("$" + expected + "$", extraction_config=config, fallback_mode="no_fallback",
                            extraction_mode="first_match", parsing_timeout=5, raise_on_error=True)
    except Exception as error:
        return {"is_correct": None, "grading_status": "gold_parse_exception", "grader_error_type": type(error).__name__}
    if not gold_parsed:
        return {"is_correct": None, "grading_status": "gold_parse_failure"}
    try:
        prediction_parsed = parse("$" + actual + "$", extraction_config=config, fallback_mode="no_fallback",
                                  extraction_mode="first_match", parsing_timeout=5, raise_on_error=True)
    except Exception as error:
        return {"is_correct": False, "grading_status": "prediction_parse_exception", "grader_error_type": type(error).__name__}
    if not prediction_parsed:
        return {"is_correct": False, "grading_status": "prediction_parse_failure"}
    try:
        correct = verify(gold_parsed, prediction_parsed, float_rounding=6, numeric_precision=15,
                         strict=True, allow_set_relation_comp=False, timeout_seconds=5, raise_on_error=True)
        return {"is_correct": bool(correct), "grading_status": "symbolic_or_numeric_comparison"}
    except Exception as error:
        return {"is_correct": None, "grading_status": "grader_exception", "grader_error_type": type(error).__name__}


def grader_manifest():
    from dataclasses import asdict
    from math_verify import LatexExtractionConfig
    packages = {name: importlib.metadata.version(name) for name in
                ("math-verify", "latex2sympy2_extended", "antlr4-python3-runtime", "sympy")}
    if packages != {"math-verify": "0.9.0", "latex2sympy2_extended": "1.11.0",
                    "antlr4-python3-runtime": "4.13.2", "sympy": "1.14.0"}:
        raise ValueError("Grader packages differ from the validated pins")
    return {"settings": GRADER_SETTINGS, "packages": packages,
            "latex_normalization": asdict(LatexExtractionConfig().normalization_config)}
