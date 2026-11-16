"""Perform the requested one-off timed check and preserve safe continuation."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256
from s1small.experiment import atomic_json


def lock_held(path):
    path = Path(path)
    if not path.exists():
        return False
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
        return False


def next_action(status, controller_active, worker_active, paused, identities_match):
    if paused:
        return "honor_pause"
    if controller_active:
        return "workflow_already_continuing"
    if not identities_match:
        return "configuration_changed_no_restart"
    if worker_active:
        return "wait_for_existing_worker"
    if status in ("complete", "partial"):
        return "handoff_finished"
    if status == "failed":
        return "failure_requires_investigation"
    return "resume_same_evaluation_pipeline_once"


def collect(root):
    state_path = root / "results/runtime/project_pipeline.json"
    pipeline = json.loads(state_path.read_text()) if state_path.exists() else {}
    output = {"pipeline": pipeline, "generation_runs": {}}
    for name in ("results/protocol_dev/protocol-dev-v2", "results/test/math500-local-v1"):
        directory = root / name
        if not directory.exists():
            continue
        counts, extracted = {}, 0
        for path in (directory / "slots").glob("*.json"):
            try:
                row = json.loads(path.read_text())
            except json.JSONDecodeError:
                counts["being_written"] = counts.get("being_written", 0) + 1
                continue
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            extracted += bool(row.get("extracted_answer"))
        output["generation_runs"][name] = {"statuses": counts, "extractable_answers": extracted,
            "interpretation": "Extraction and execution counts are not correctness scores."}
    output["benchmark_frozen"] = (root / "manifests/evaluation_freeze.json").exists()
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "results/runtime"
    request = json.loads((runtime / "requested_20_minute_check.json").read_text())
    due = datetime.fromisoformat(request["check_due_time"]).timestamp()
    lock = (runtime / "timed_generation_check.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print("The requested timed checker is already active", flush=True)
        return
    path = runtime / "timed_generation_check.json"
    record = {"status": "scheduled", "checker_pid": os.getpid(), "request": request,
              "check_completed": False, "additional_training": False}
    atomic_json(path, record)
    try:
        while time.time() < due:
            time.sleep(min(30, max(0, due - time.time())))
        while True:
            snapshot = collect(root)
            stage6 = json.loads((root / "results/study/local-study-v1/stage6_summary.json").read_text())
            identities = (stage6.get("completion_scope") == "existing_adapter_selected_for_evaluation"
                          and stage6.get("final_training") is None
                          and all(file_sha256(root / name) == expected for name, expected in request["source_sha256"].items()))
            workers = [*root.glob("results/protocol_dev/*/.run.lock"), *root.glob("results/test/*/.run.lock")]
            action = next_action(snapshot["pipeline"].get("status"), lock_held(runtime / "project_pipeline.lock"),
                                 any(lock_held(p) for p in workers), (runtime / "PAUSE").exists(), identities)
            record.update(status="checked", check_completed=True,
                          checked_at_local=datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
                          checked_at_utc=datetime.now(timezone.utc).isoformat(), action=action, snapshot=snapshot)
            if action == "resume_same_evaluation_pipeline_once":
                command = ["/usr/bin/caffeinate", "-i", str(root / ".venv-mlx/bin/python"),
                           str(root / "scripts/complete_project.py"), "--protocol-run-id", "protocol-dev-v2",
                           "--protocol-config", "budget_protocol_v2.json"]
                with (runtime / "protocol_v2_pipeline.log").open("ab", buffering=0) as stream:
                    child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                record.update(resumed_controller_pid=child.pid, continuation_command=command)
            atomic_json(path, record)
            lines = ["# Requested timed generation check", "", f"Requested: {request['requested_check_time']}",
                     f"Due: {request['check_due_time']}", f"Checked: {record['checked_at_local']}", "",
                     f"Action: `{action}`. No additional training.", "",
                     f"Pipeline stage: `{snapshot['pipeline'].get('stage')}`; recorded status: `{snapshot['pipeline'].get('status')}`.", ""]
            for name, value in snapshot["generation_runs"].items():
                lines.extend([f"- `{name}`: {value['statuses']}; {value['extractable_answers']} extractable answers."])
            lines.extend(["", "Extraction counts are not accuracy. Validation/freeze/benchmark gates remain enforced.", ""])
            (root / "docs/timed_generation_check.md").write_text("\n".join(lines))
            print(json.dumps(record, indent=2), flush=True)
            if action != "wait_for_existing_worker":
                break
            time.sleep(30)
    except BaseException as error:
        record.update(status="failed", error=str(error), traceback=traceback.format_exc())
        atomic_json(path, record)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
