"""Execute the predeclared two-trial grid, freeze by held-out loss, retrain fresh."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256, save_json
from s1small.training_plan import select_trial


def execute(root, run_id, mode, learning_rate=None):
    output = root / "results/training" / run_id
    final = output / "summary.json"
    if final.exists():
        summary = json.loads(final.read_text())
        if summary["status"] != "complete" or not summary["fresh_process_reload_identical"]:
            raise ValueError("Existing unsuccessful result is retained for review")
        print(f"Retaining completed verified run {run_id}", flush=True)
        return summary
    trained = output / "training_summary.json"
    if not (trained.exists() and (output / "generation_before_reload.json").exists()):
        command = [sys.executable, str(root / "scripts/train_full.py"), "--mode", mode, "--run-id", run_id]
        if learning_rate is not None:
            command += ["--learning-rate", str(learning_rate)]
        if output.exists():
            command += ["--resume"]
        print(f"Starting or resuming {run_id}", flush=True)
        subprocess.run(command, cwd=root, check=True)
    subprocess.run([sys.executable, str(root / "scripts/train_full.py"), "--mode", "verify", "--run-id", run_id],
                   cwd=root, check=True)
    return json.loads(final.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="local-study-v1")
    args = parser.parse_args()
    if not args.prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.prefix):
        parser.error("Use a simple run prefix")
    import yaml
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/train_development.yaml"
    config = yaml.safe_load(config_path.read_text())
    eligibility = json.loads((root / "manifests/s1k_eligibility.json").read_text())
    dev_ids = [row["id"] for row in eligibility["development"]]
    trials = []
    initial = None
    for index, learning_rate in enumerate(config["learning_rates"]):
        run_id = f"{args.prefix}-trial-{index + 1}"
        row = execute(root, run_id, "development", learning_rate)
        if (row["scope"] != "development_training" or row["training_examples"] != len(eligibility["train"])
                or row["learning_rate"] != learning_rate):
            raise ValueError("Trial differs from the predeclared complete-data grid")
        manifest = json.loads((root / "results/training" / run_id / "manifest.json").read_text())
        if initial is not None and manifest["initial_adapter_tensor_sha256"] != initial:
            raise ValueError("Development trials did not begin from the same fresh adapter initialization")
        initial = manifest["initial_adapter_tensor_sha256"]
        trials.append(row)
    selected = select_trial(trials, dev_ids)
    selected_manifest = json.loads((root / "results/training" / selected["run_id"] / "manifest.json").read_text())
    freeze = {"scope": "training_recipe_freeze_before_final_retraining", "test_scored": False,
              "selection_rule": config["selection_rule"], "selection_metric": config["selection_metric"],
              "development_config_sha256": file_sha256(config_path),
              "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json"),
              "selected_run_id": selected["run_id"], "selected_development_loss": selected["development_loss"],
              "trial_summaries": trials, "training_recipe": selected_manifest["recipe"],
              "initial_adapter_tensor_sha256": initial,
              "conversion_provenance_sha256": selected_manifest["identity"]["conversion_provenance_sha256"]}
    save_json(root / "manifests/training_freeze.json", freeze)
    final = execute(root, args.prefix + "-final", "final")
    if final["training_examples"] != len(eligibility["final_training"]) or final["scope"] != "final_training":
        raise ValueError("Final training did not cover the exact eligible union")
    report = {"status": "complete", "selected_learning_rate": selected["learning_rate"], "trials": trials,
              "final_training": final, "training_freeze_sha256": file_sha256(root / "manifests/training_freeze.json"),
              "test_scored": False, "hardware_report": "docs/hardware_pilot.md"}
    output = root / "results/study" / args.prefix
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "stage6_summary.json", report)
    (root / "docs/development_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Stage 6 completed with a fresh final adapter; no MATH-500 evaluation has run", flush=True)


if __name__ == "__main__":
    main()
