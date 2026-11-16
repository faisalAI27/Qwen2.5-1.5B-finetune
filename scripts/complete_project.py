"""Continue the local project through protocol validation, sealed test and report."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.experiment import atomic_json
from s1small.resume import prepare_training_resume


def training_failure(root, prefix):
    for suffix in ("trial-1", "trial-2", "final"):
        path = root / "results/training" / (prefix + "-" + suffix)
        attempts = sorted((path / "attempts").glob("attempt-*"))
        if attempts and (attempts[-1] / "failure.json").exists():
            summary = path / "summary.json"
            if not summary.exists() or json.loads(summary.read_text())["status"] != "complete":
                return str(attempts[-1] / "failure.json")
    return None


def live_training_workers(prefix):
    output = subprocess.run(["/bin/ps", "-axo", "pid=,args="], capture_output=True, text=True, check=True).stdout
    workers = []
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "scripts/train_full.py" in parts[1] and f"--run-id {prefix}-" in parts[1]:
            workers.append(int(parts[0]))
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-prefix", default="local-study-v1")
    parser.add_argument("--protocol-run-id", default="protocol-dev-v2")
    parser.add_argument("--test-run-id", default="math500-local-v1")
    parser.add_argument("--protocol-config", default="budget_protocol_v2.json")
    parser.add_argument("--wait-for-stage6", action="store_true")
    parser.add_argument("--stage6-pid", type=int)
    parser.add_argument("--resume-interrupted-stage6", action="store_true",
                        help="Permit one same-recipe restart after process interruption; numerical failures still halt")
    args = parser.parse_args()
    if Path(args.protocol_config).name != args.protocol_config or not args.protocol_config.endswith(".json"):
        parser.error("Use a protocol JSON filename in configs/")
    if args.resume_interrupted_stage6 and (not args.wait_for_stage6 or not args.stage6_pid):
        parser.error("Automatic interruption recovery requires --wait-for-stage6 and the existing --stage6-pid")
    for value in (args.study_prefix, args.protocol_run_id, args.test_run_id):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
            parser.error("Use simple run IDs")
    root = Path(__file__).resolve().parents[1]
    runtime = root / "results/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / "project_pipeline.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("The project continuation controller is already active")
    status_path = runtime / "project_pipeline.json"
    state = {"status": "running", "stage": "setup", "study_prefix": args.study_prefix,
             "protocol_run_id": args.protocol_run_id, "test_run_id": args.test_run_id,
             "protocol_config": args.protocol_config,
             "stage6_pid": args.stage6_pid, "controller_pid": __import__("os").getpid()}
    def update(stage, **values):
        state.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(status_path, state)
    def run(stage, interpreter, script, *arguments, allow_partial=False):
        update(stage)
        command = [str(root / interpreter), str(root / "scripts" / script), *arguments]
        print(json.dumps({"stage": stage, "command": command}), flush=True)
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode and not allow_partial:
            raise RuntimeError(f"{stage} exited {result.returncode}; previous evidence is retained")
        return result.returncode
    try:
        if (runtime / "PAUSE").exists():
            update("paused_by_local_marker", status="paused")
            return
        stage6 = root / "results/study" / args.study_prefix / "stage6_summary.json"
        if args.wait_for_stage6:
            update("waiting_for_existing_stage6")
            heartbeat = 0
            while not stage6.exists():
                if (runtime / "PAUSE").exists():
                    update("paused_by_local_marker", status="paused")
                    return
                failure = training_failure(root, args.study_prefix)
                if args.stage6_pid:
                    process = subprocess.run(["/bin/ps", "-p", str(args.stage6_pid), "-o", "command="], capture_output=True, text=True)
                    if process.returncode or f"run_development.py --prefix {args.study_prefix}" not in process.stdout:
                        if stage6.exists():
                            break
                        workers = live_training_workers(args.study_prefix)
                        if workers:
                            update("waiting_for_orphaned_training_worker", live_worker_pids=workers)
                            time.sleep(30)
                            continue
                        can_restart = args.resume_interrupted_stage6
                        if failure:
                            error = json.loads(Path(failure).read_text())
                            can_restart = can_restart and error["error_type"] in ("KeyboardInterrupt", "SystemExit")
                        if can_restart:
                            recovery = prepare_training_resume(root, args.study_prefix)
                            update("resuming_interrupted_stage6", recovered_checkpoint_events=recovery, automatic_training_restarts=1)
                            run("development_and_final_training", ".venv-mlx/bin/python", "run_development.py", "--prefix", args.study_prefix)
                            break
                        raise RuntimeError(f"Training controller ended before Stage 6 completion; checkpoints are retained. Failure evidence: {failure}")
                elif failure:
                    raise RuntimeError(f"Full training stopped; resume evidence: {failure}")
                if time.monotonic() >= heartbeat:
                    update("waiting_for_existing_stage6")
                    print("Waiting for verified Stage 6 output; no duplicate model is loaded", flush=True)
                    heartbeat = time.monotonic() + 600
                time.sleep(30)
        elif not stage6.exists():
            run("development_and_final_training", ".venv-mlx/bin/python", "run_development.py", "--prefix", args.study_prefix)
        completed = json.loads(stage6.read_text())
        if completed["status"] != "complete":
            raise ValueError("Stage 6 did not complete")
        run("held_out_protocol_validation", ".venv-mlx/bin/python", "evaluate.py", "--scope", "development", "--run-id", args.protocol_run_id,
            "--protocol-config", args.protocol_config)
        run("pre_test_freeze", ".venv-mlx/bin/python", "freeze_evaluation.py", "--development-run-id", args.protocol_run_id,
            "--study-prefix", args.study_prefix, "--protocol-config", args.protocol_config)
        # Resource failures remain explicit partial coverage; still produce the
        # honest analysis/report if the raw test plan exists.
        run("frozen_math500_generation", ".venv-mlx/bin/python", "evaluate.py", "--scope", "test", "--run-id", args.test_run_id,
            "--protocol-config", args.protocol_config, allow_partial=True)
        run("local_equivalence_grading", ".venv-eval/bin/python", "grade_results.py", "--run-id", args.test_run_id)
        run("paired_analysis_and_figures", ".venv-eval/bin/python", "analyze_results.py", "--run-id", args.test_run_id)
        run("paper_and_demo_handoff", ".venv-eval/bin/python", "report_project.py", "--run-id", args.test_run_id, "--study-prefix", args.study_prefix)
        report = json.loads((root / "docs/handoff.json").read_text())
        update("finished", status="complete" if report["status"] == "complete" else "partial", handoff=report)
        print(json.dumps(state, indent=2), flush=True)
    except BaseException as error:
        update(state["stage"], status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
