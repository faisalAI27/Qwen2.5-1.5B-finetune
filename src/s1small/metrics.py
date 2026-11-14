"""Fixed-denominator summaries and paired question-level uncertainty."""

import math
import statistics


def mean_or_none(values):
    return statistics.mean(values) if values else None


def summarize_condition(records, planned):
    if len(records) != planned or len({r["question_id"] for r in records}) != planned:
        raise ValueError("Every planned ID must have exactly one record")
    outputs = [row for row in records if row["status"] == "ok"]
    resolved = [row for row in outputs if isinstance(row.get("is_correct"), bool)]
    correct = sum(row["is_correct"] for row in resolved)
    complete = len(resolved) == planned
    def values(field):
        return [row[field] for row in outputs if isinstance(row.get(field), (float, int)) and math.isfinite(row[field])]
    latency = values("latency_seconds")
    reason = values("reasoning_tokens")
    total = values("total_generated_tokens")
    return {"planned": planned, "model_outputs": len(outputs), "resolved_grades": len(resolved),
            "correct": correct, "complete": complete,
            "accuracy": correct / planned if complete else None,
            "resolved_accuracy": correct / len(resolved) if resolved else None,
            "confirmed_correct_fraction_of_planned": correct / planned,
            "infrastructure_failures": sum(row["status"] == "infrastructure_failure" for row in records),
            "not_attempted": sum(row["status"] == "not_attempted" for row in records),
            "unresolved_grader_outputs": len(outputs) - len(resolved),
            "extraction_failures": sum(row.get("extracted_answer") is None for row in outputs),
            "extraction_failure_rate": sum(row.get("extracted_answer") is None for row in outputs) / len(outputs) if outputs else None,
            "mean_reasoning_tokens": mean_or_none(reason), "median_reasoning_tokens": statistics.median(reason) if reason else None,
            "mean_total_generated_tokens": mean_or_none(total), "median_total_generated_tokens": statistics.median(total) if total else None,
            "mean_latency_seconds": mean_or_none(latency), "median_latency_seconds": statistics.median(latency) if latency else None,
            "timeouts": sum(bool(row.get("timed_out")) for row in outputs),
            "safety_cap_hits": sum(bool(row.get("safety_cap_hit")) for row in outputs),
            "reasoning_cap_hits": sum(bool(row.get("reasoning_cap_hit")) for row in outputs),
            "answer_cap_hits": sum(bool(row.get("answer_cap_hit")) for row in outputs),
            "early_reasoning_eos_outputs": sum(bool(row.get("early_reasoning_eos")) for row in outputs),
            "wait_triggered_rate": sum(row.get("forced_wait_count", 0) > 0 for row in outputs) / len(outputs) if outputs else None,
            "mean_actual_wait_count": mean_or_none([row.get("forced_wait_count", 0) for row in outputs]),
            "cost_denominator": len(outputs)}


def paired_comparison(reference, intervention, seed=42, iterations=10000, confidence=0.95):
    import numpy as np
    left, right = {r["question_id"]: r for r in reference}, {r["question_id"]: r for r in intervention}
    if len(left) != len(reference) or len(right) != len(intervention) or set(left) != set(right):
        raise ValueError("Paired analysis requires identical unique planned IDs")
    if iterations < 100 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    valid = [key for key in sorted(left) if all(row["status"] == "ok" and isinstance(row.get("is_correct"), bool)
                                               for row in (left[key], right[key]))]
    transitions = {name: 0 for name in ("wrong_to_right", "right_to_wrong", "wrong_to_wrong", "right_to_right")}
    for key in valid:
        a, b = left[key]["is_correct"], right[key]["is_correct"]
        transitions[("right" if a else "wrong") + "_to_" + ("right" if b else "wrong")] += 1
    report = {"paired_resolved_denominator": len(valid), "planned_denominator": len(left),
              "excluded_unresolved_ids": [key for key in sorted(left) if key not in set(valid)],
              "transitions": transitions, "bootstrap_seed": seed, "bootstrap_iterations": iterations,
              "confidence_level": confidence, "resampling_unit": "paired_question_ID_with_replacement",
              "reference_extraction_failures": sum(left[k].get("extracted_answer") is None for k in valid),
              "intervention_extraction_failures": sum(right[k].get("extracted_answer") is None for k in valid)}
    if not valid:
        return {**report, "accuracy_difference": None, "confidence_interval": None}
    difference = np.array([int(right[k]["is_correct"]) - int(left[k]["is_correct"]) for k in valid], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, iterations, 1000):
        indices = rng.integers(0, len(valid), size=(min(1000, iterations - start), len(valid)))
        samples.append(difference[indices].mean(axis=1))
    tail = (1 - confidence) / 2
    low, high = np.quantile(np.concatenate(samples), [tail, 1 - tail]).tolist()
    return {**report, "accuracy_difference": float(difference.mean()), "confidence_interval": [low, high]}
