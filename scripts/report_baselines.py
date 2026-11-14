"""Summarize development smoke outputs and check A/B identity and repeats."""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import read_jsonl


def inspect_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_jsonl(directory / "predictions.jsonl")
    summary = json.loads((directory / "summary.json").read_text())
    if manifest["scope"] != "development_smoke_only":
        raise ValueError("Expected development smoke outputs")
    if len(rows) != manifest["planned_records"]:
        raise ValueError("Saved predictions do not match the planned denominator")
    if len({(row["question_id"], row["condition"], row["repeat"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate prediction keys")
    if {row["weights_sha256"] for row in rows} != {manifest["weights_sha256"]}:
        raise ValueError("A/B checkpoint identities differ")
    expected_ids = manifest["question_ids"]
    for repeat in range(manifest["config"]["repeats"]):
        for condition in ("A", "B"):
            if [row["question_id"] for row in rows if row["repeat"] == repeat and row["condition"] == condition] != expected_ids:
                raise ValueError("A/B question IDs or order differ")
    report = {"run_id": manifest["run_id"], "status": summary["status"], "question_ids": expected_ids,
              "weights_sha256": manifest["weights_sha256"], "model": json.loads((directory / "model.json").read_text()),
              "prompts_sha256": manifest["prompts_sha256"],
              "tokenizer_revision": manifest["sources"]["tokenizer"]["revision"],
              "config": manifest["config"], "conditions": {}, "repeat_checks": summary["repeats"],
              "accuracy": None, "grading_status": "not_scored"}
    for condition in ("A", "B"):
        predictions = [row for row in rows if row["condition"] == condition]
        successful = [row for row in predictions if row["status"] == "ok"]
        count = len(successful)
        report["conditions"][condition] = {
            "planned_records": len(predictions), "successful_generations": count,
            "extracted_answers": sum(row.get("extracted_answer") is not None for row in successful),
            "extraction_failures": sum(row.get("extracted_answer") is None for row in successful),
            "token_limit_events": sum(row["termination_reason"] == "max_new_tokens" for row in successful),
            "timeouts": sum(row["termination_reason"] == "timeout" for row in successful),
            "mean_generated_tokens": statistics.mean(row["total_generated_tokens"] for row in successful) if count else None,
            "median_generation_seconds": statistics.median(row["latency_seconds"] for row in successful) if count else None,
            "maximum_sampled_mps_driver_bytes": max((row["memory_after_generation"].get("mps_driver_allocated_bytes", 0) for row in successful), default=None),
        }
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--compare-with", help="Another fresh model-load run with the same generation settings")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "results/dev_baselines"
    for value in [args.run_id, args.compare_with]:
        if value is not None and (Path(value).name != value or value in (".", "..")):
            parser.error("Run IDs must be directory names, not paths")
    report, rows = inspect_run(base / args.run_id)
    if args.compare_with:
        other, other_rows = inspect_run(base / args.compare_with)
        comparable_keys = ("device", "dtype", "attention_implementation", "seed", "max_new_tokens", "timeout_seconds", "deterministic_algorithms")
        if (any(report[key] != other[key] for key in ("weights_sha256", "prompts_sha256", "tokenizer_revision"))
                or any(report["config"][key] != other["config"][key] for key in comparable_keys)):
            raise ValueError("Fresh-load runs use different weights or generation settings")
        index = {(row["question_id"], row["condition"], row["repeat"]): row for row in other_rows}
        matches = []
        for row in rows:
            key = row["question_id"], row["condition"], row["repeat"]
            if key in index:
                other_row = index[key]
                if row.get("prompt_sha256") != other_row.get("prompt_sha256"):
                    raise ValueError("Fresh-load runs used different rendered prompts for the same question")
                matches.append({"question_id": key[0], "condition": key[1], "repeat": key[2],
                                "comparable": row["status"] == other_row["status"] == "ok",
                                "token_ids_identical": row["status"] == other_row["status"] == "ok" and row["generated_token_ids"] == other_row["generated_token_ids"]})
        report["fresh_load_comparison"] = {"other_run_id": args.compare_with, "overlapping_records": matches}
    output = base / args.run_id / "audit_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    (root / "docs/baseline_smoke_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
