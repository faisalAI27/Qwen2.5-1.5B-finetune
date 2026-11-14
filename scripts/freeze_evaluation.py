"""Freeze all test choices from selected-model development evidence; no gold IO."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.baselines import load_development
from s1small.budget_forcing import CONDITIONS
from s1small.data import digest, file_sha256, read_jsonl, save_json
from s1small.experiment import PredictionStore, validate_question_identity, verify_freeze
from s1small.generate import build_prompt
from s1small.mlx_decode import runtime_signature, s1_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run-id", default="protocol-dev-v1")
    parser.add_argument("--study-prefix", default="local-study-v1")
    args = parser.parse_args()
    if any(Path(value).name != value or value in (".", "..") for value in (args.development_run_id, args.study_prefix)):
        parser.error("Use simple run IDs")
    root = Path(__file__).resolve().parents[1]
    output = root / "manifests/evaluation_freeze.json"
    if output.exists():
        verify_freeze(root, json.loads(output.read_text()))
        print("Retaining verified existing evaluation freeze"); return
    if (root / "data/raw/math500/test_with_answers.jsonl").exists() or any((root / "results/test").glob("*/manifest.json")):
        raise ValueError("Test artifacts already exist without this freeze; cannot retroactively freeze choices")
    training = json.loads((root / "manifests/training_freeze.json").read_text())
    stage6 = json.loads((root / "results/study" / args.study_prefix / "stage6_summary.json").read_text())
    if stage6["status"] != "complete" or stage6["test_scored"]:
        raise ValueError("Verified final training must be available without prior test scoring")
    directory = root / "results/protocol_dev" / args.development_run_id
    manifest = json.loads((directory / "manifest.json").read_text())
    protocol = json.loads((root / "configs/budget_protocol_v1.json").read_text())
    expected_ids = [row["id"] for row in load_development(root, 5)]
    selected = json.loads((root / "results/training" / training["selected_run_id"] / "summary.json").read_text())
    if (manifest["scope"] != "development" or manifest["question_ids"] != expected_ids
            or manifest["protocol"] != protocol or manifest["adapter_directory"] != selected["adapter_directory"]
            or manifest["generation_runtime"] != runtime_signature()):
        raise ValueError("Protocol validation must use held-out questions and the selected 896-row adapter")
    store = PredictionStore(directory, [(key, condition) for key in expected_ids for condition in CONDITIONS], digest(manifest))
    records = store.records()
    if len(records) != 45 or any(row["status"] != "ok" or row.get("gold_answer") is not None for row in records):
        raise ValueError("All 45 development protocol generations must execute without benchmark gold")
    # Audit only already-sealed questions and prompt lengths. No answers/solutions.
    questions = read_jsonl(root / "data/raw/math500/questions.jsonl")
    validate_question_identity(questions, json.loads((root / "manifests/math500_questions.json").read_text()))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / "data/raw/tokenizer", local_files_only=True)
    templates = json.loads((root / "configs/prompts/baseline_v1.json").read_text())
    input_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for row in questions
                     for text in (build_prompt(row["question"], "A", templates, tokenizer),
                                  build_prompt(row["question"], "B", templates, tokenizer), s1_prompt(row["question"], tokenizer))]
    if max(input_lengths) + protocol["absolute_safety_ceiling"] > 32768:
        raise ValueError("A sealed question plus safety allowance exceeds model context; do not truncate it")
    grader = json.loads(subprocess.check_output([str(root / ".venv-eval/bin/python"), str(root / "scripts/grade_results.py"), "--print-manifest"],
                                               cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")}, text=True))
    final = stage6["final_training"]
    files = ["configs/sources.json", "configs/budget_protocol_v1.json", "configs/prompts/baseline_v1.json", "configs/formatting.json",
             "manifests/training_freeze.json", "manifests/s1k_eligibility.json", "manifests/math500_questions.json",
             "data/raw/math500/questions.jsonl", "docs/baseline_smoke_summary.json",
             "scripts/evaluate.py", "scripts/grade_results.py", "src/s1small/budget_forcing.py", "src/s1small/benchmark.py",
             "src/s1small/mlx_decode.py", "src/s1small/experiment.py", "src/s1small/grading.py", "src/s1small/mlx_backend.py",
             "src/s1small/mlx_runtime.py", "src/s1small/data.py", "src/s1small/formatting.py", "src/s1small/generate.py",
             "scripts/analyze_results.py", "scripts/report_project.py", "src/s1small/metrics.py",
             "requirements-eval-lock.txt", "requirements-mlx-lock.txt", "requirements-reporting-lock.txt"]
    for adapter in (selected["adapter_directory"], final["adapter_directory"]):
        files.extend(str(Path(adapter) / name) for name in ("adapters.safetensors", "adapter_config.json", "source_provenance.json"))
    revision = json.loads((root / "configs/sources.json").read_text())["model"]["revision"]
    conversion_directory = Path("artifacts/base-mlx4") / (revision + "-g64")
    conversion = json.loads((root / conversion_directory / "provenance.json").read_text())
    files.append(str(conversion_directory / "provenance.json"))
    files.extend(str(conversion_directory / row["name"]) for row in conversion["files"])
    record = {"scope": "frozen_test_evaluation", "test_used_for_selection": False,
              "generation_runtime": runtime_signature(),
              "protocol": {**protocol, "scope": "frozen_test_evaluation"}, "grader": grader,
              "final_adapter_directory": final["adapter_directory"], "selected_development_adapter_directory": selected["adapter_directory"],
              "development_manifest_sha256": file_sha256(directory / "manifest.json"),
              "development_outputs_sha256": digest(records), "development_question_ids": expected_ids,
              "development_run_id": args.development_run_id,
              "development_protocol_diagnostics": json.loads((directory / "summary.json").read_text()),
              "maximum_sealed_prompt_tokens": max(input_lengths), "planned_questions": 500, "planned_conditions": list(CONDITIONS),
              "analysis": {"bootstrap_seed": 42, "bootstrap_iterations": 10000, "confidence_level": 0.95,
                           "reference_condition": "C", "multiple_comparison_adjustment": "none; intervals are descriptive",
                           "partial_score_policy": "No full accuracy if planned outputs/grades are incomplete; report exact denominators and paired resolved IDs."},
              "files_sha256": {name: file_sha256(root / name) for name in files}}
    save_json(output, record)
    print(json.dumps({"freeze": str(output.relative_to(root)), "planned_predictions": 4500,
                      "maximum_prompt_tokens": max(input_lengths), "sha256": file_sha256(output)}), flush=True)


if __name__ == "__main__":
    main()
