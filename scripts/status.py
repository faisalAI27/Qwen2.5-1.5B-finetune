"""Read local project progress without loading a model or starting any job."""

import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess


def process_alive(pid, expected_script):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    except OSError:
        return None  # Process inspection is unavailable inside some sandboxes.
    return result.returncode == 0 and expected_script in result.stdout


def main():
    root = Path(__file__).resolve().parents[1]
    runs = []
    base = root / "results/training"
    if base.exists():
        for directory in sorted(base.iterdir()):
            if not directory.is_dir() or not (directory / "manifest.json").exists():
                continue
            manifest = json.loads((directory / "manifest.json").read_text())
            record = {"run_id": directory.name, "scope": manifest["scope"], "learning_rate": manifest["recipe"]["learning_rate"],
                      "planned_examples": len(manifest["ordered_partition_ids"]), "status": "running_or_interrupted"}
            summary = directory / "summary.json"
            trained = directory / "training_summary.json"
            if summary.exists():
                result = json.loads(summary.read_text())
                record.update(status=result["status"], development_loss=result["development_loss"],
                              fresh_process_reload_identical=result["fresh_process_reload_identical"])
            elif trained.exists():
                record["status"] = "trained_reload_pending"
            checkpoints = root / "artifacts/training" / directory.name / "checkpoints"
            if (checkpoints / "latest.json").exists():
                latest = json.loads((checkpoints / "latest.json").read_text())
                state = json.loads((checkpoints / latest["directory"] / "checkpoint.json").read_text())
                for key in ("completed_steps", "examples_committed", "input_tokens_committed", "training_seconds_committed"):
                    record[key] = state[key]
                record["checkpoint"] = latest["directory"]
            attempts = sorted((directory / "attempts").glob("attempt-*"))
            if attempts:
                failure = attempts[-1] / "failure.json"
                if failure.exists() and record["status"] != "complete":
                    record.update(status="failed_attempt_resume_available", error=json.loads(failure.read_text())["error"])
                progress = attempts[-1] / "microbatches.jsonl"
                if progress.exists():
                    # Bounded tail read; a record is much shorter than 8 KiB.
                    with progress.open("rb") as stream:
                        stream.seek(0, 2)
                        size = stream.tell()
                        stream.seek(max(0, size - 8192))
                        lines = stream.read().decode().splitlines()
                    rows = []
                    for line in lines:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            # The first bounded-tail line or a concurrent final
                            # write can be incomplete; retain complete records.
                            continue
                    if rows:
                        record["last_microbatch"] = rows[-1]
                        pending = rows[-1]["pending_step"]
                        record["pending_step"] = pending if pending > record.get("completed_steps", 0) else None
                        record["microbatches_logged_in_pending_step"] = (
                            len({row["id"] for row in rows if row["pending_step"] == pending})
                            if record["pending_step"] is not None else 0)
                        record["last_progress_at_utc"] = datetime.fromtimestamp(progress.stat().st_mtime, timezone.utc).isoformat()
            runs.append(record)
    pipeline_path = root / "results/runtime/project_pipeline.json"
    pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else None
    if pipeline:
        pipeline["controller_process_alive"] = process_alive(pipeline.get("controller_pid"), "scripts/complete_project.py")
        pipeline["original_training_parent_alive"] = process_alive(pipeline.get("stage6_pid"), "scripts/run_development.py")
    review_path = root / "results/runtime/first_trial_review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    if review:
        review["controller_process_alive"] = process_alive(review.get("controller_pid"), "scripts/review_first_trial.py")
    request_path = root / "results/runtime/first_trial_review_request.json"
    request = json.loads(request_path.read_text()) if request_path.exists() else None
    decision_path = root / "configs/single_run_decision.json"
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else None
    diagnostic_path = root / "results/runtime/readout_diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    if diagnostic:
        diagnostic["controller_process_alive"] = process_alive(diagnostic.get("controller_pid"), "scripts/diagnose_readout.py")
    print(json.dumps({"free_disk_gib": round(shutil.disk_usage(root).free / 1024**3, 2), "training_runs": runs,
                      "continuation_pipeline": pipeline,
                      "first_trial_review": review, "first_trial_review_request": request,
                      "single_run_decision": decision,
                      "readout_diagnostic": diagnostic,
                      "test_protocol_frozen": (root / "manifests/evaluation_freeze.json").exists(),
                      "note": "running_or_interrupted is not a claim that a process is alive; committed counts come from durable checkpoints. Pending microbatches are not yet durable optimizer updates."}, indent=2))


if __name__ == "__main__":
    main()
