"""Adopt the reviewed existing adapter for evaluation without more training."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256, save_json
from review_first_trial import compare_validation, validate_completed_trial


def prepare(root):
    root = Path(root)
    decision_path = root / "configs/single_run_decision.json"
    decision = json.loads(decision_path.read_text())
    if (decision["strategy"] != "evaluate_existing_adapter" or decision["additional_training_authorized"]
            or decision["test_used_for_decision"]):
        raise ValueError("An explicit single-existing-adapter decision before test is required")
    if (root / "data/raw/math500/test_with_answers.jsonl").exists() or any((root / "results/test").glob("*/manifest.json")):
        raise ValueError("Cannot change model selection after benchmark access")
    run_id, prefix = decision["selected_run_id"], decision["study_prefix"]
    if any(not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value)
           for value in (run_id, prefix)):
        raise ValueError("Use simple run IDs")
    source = root / "results/training" / run_id
    summary = json.loads((source / "summary.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    eligibility_path = root / "manifests/s1k_eligibility.json"
    eligibility = json.loads(eligibility_path.read_text())
    validate_completed_trial(summary, manifest, eligibility)
    review = json.loads((root / "docs/first_trial_review.json").read_text())
    baseline_path = root / "results/review" / run_id / "base_validation.json"
    baseline = json.loads(baseline_path.read_text())
    if (review["run_id"] != run_id or review["source_hashes"]["trial_summary"] != file_sha256(source / "summary.json")
            or review["source_hashes"]["base_validation"] != file_sha256(baseline_path)
            or decision["review_sha256"] != file_sha256(root / "docs/first_trial_review.json")
            or manifest["identity"]["eligibility_sha256"] != file_sha256(eligibility_path)):
        raise ValueError("The user decision does not match the completed reviewed run")
    comparison = compare_validation(summary, baseline)
    for name, expected in manifest["identity"]["code_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("Original training code identity changed")
    # Exported files must still equal the final durable checkpoint files.
    checkpoints = root / "artifacts/training" / run_id / "checkpoints"
    pointer = json.loads((checkpoints / "latest.json").read_text())
    checkpoint = checkpoints / pointer["directory"]
    if file_sha256(checkpoint / "checkpoint.json") != pointer["checkpoint_sha256"]:
        raise ValueError("Final checkpoint pointer changed")
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    if metadata["examples_committed"] != len(eligibility["train"]) or metadata["recipe_sha256"] != manifest["recipe_sha256"]:
        raise ValueError("Final checkpoint does not cover this complete training run")
    for row in metadata["files"]:
        if file_sha256(checkpoint / row["name"]) != row["sha256"]:
            raise ValueError("Checkpoint file changed")
    for name in ("adapters.safetensors", "adapter_config.json"):
        if file_sha256(root / summary["adapter_directory"] / name) != file_sha256(checkpoint / name):
            raise ValueError("Exported adapter differs from the verified final checkpoint")
    deviation = {"strategy": decision["strategy"], "completed_development_trials": 1,
                 "second_trial": "skipped_by_user_for_current_study", "fresh_all_data_retraining": "deferred_by_user",
                 "evaluation_training_examples": len(eligibility["train"]),
                 "held_out_development_examples": len(eligibility["development"]),
                 "original_planned_final_examples": len(eligibility["final_training"]),
                 "hyperparameter_search_performed": False, "additional_training_authorized": False}
    freeze = {"scope": "existing_reviewed_adapter_selection_before_test", "test_scored": False,
              "selection_rule": "user_accepts_first_reviewed_run_under_local_compute_constraint",
              "selection_metric": "held_out_teacher_target_nll_is_supporting_evidence_not_benchmark_accuracy",
              "selected_run_id": run_id, "selected_development_loss": summary["development_loss"],
              "trial_summaries": [summary], "training_recipe": manifest["recipe"],
              "initial_adapter_tensor_sha256": manifest["initial_adapter_tensor_sha256"],
              "eligibility_sha256": file_sha256(eligibility_path),
              "development_config_sha256": file_sha256(root / "configs/train_development.yaml"),
              "conversion_provenance_sha256": manifest["identity"]["conversion_provenance_sha256"],
              "decision_path": str(decision_path.relative_to(root)), "decision_sha256": file_sha256(decision_path),
              "training_deviation": deviation, "matched_base_comparison": comparison,
              "adapter_files_sha256": {name: file_sha256(root / summary["adapter_directory"] / name)
                  for name in ("adapters.safetensors", "adapter_config.json", "source_provenance.json")}}
    freeze_path = root / "manifests/training_freeze.json"
    save_json(freeze_path, freeze)
    result = {"status": "complete", "completion_scope": "existing_adapter_selected_for_evaluation",
              "selected_learning_rate": summary["learning_rate"], "trials": [summary],
              "evaluation_model": summary, "final_training": None, "training_deviation": deviation,
              "training_freeze_sha256": file_sha256(freeze_path), "test_scored": False,
              "hardware_report": "docs/hardware_pilot.md"}
    output = root / "results/study" / prefix
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "stage6_summary.json", result)
    save_json(root / "docs/development_summary.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(Path(__file__).resolve().parents[1]), indent=2))
