"""Recover only verified checkpoints after an interrupted project worker."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from s1small.data import file_sha256
from s1small.experiment import atomic_json


def prepare_training_resume(root, prefix):
    root = Path(root)
    events = []
    for suffix in ("trial-1", "trial-2", "final"):
        run_id = prefix + "-" + suffix
        manifest_path = root / "results/training" / run_id / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        artifact = root / "artifacts/training" / run_id
        checkpoints = artifact / "checkpoints"
        if not checkpoints.exists():
            continue
        archive = artifact / "interrupted_checkpoints"
        valid = []
        for candidate in sorted(checkpoints.glob("step-*")):
            if not candidate.is_dir():
                continue
            reason = None
            if candidate.name.endswith("-incomplete"):
                reason = "uncommitted_staging_directory"
            else:
                try:
                    metadata = json.loads((candidate / "checkpoint.json").read_text())
                    if metadata["recipe_sha256"] != manifest["recipe_sha256"]:
                        raise ValueError("Checkpoint belongs to a different recipe")
                    if any(file_sha256(candidate / row["name"]) != row["sha256"] for row in metadata["files"]):
                        reason = "checkpoint_file_checksum_failure"
                    else:
                        valid.append((metadata["completed_steps"], candidate))
                except (OSError, KeyError, json.JSONDecodeError) as error:
                    reason = "incomplete_checkpoint_metadata: " + type(error).__name__
            if reason:
                archive.mkdir(parents=True, exist_ok=True)
                retained = archive / (candidate.name + f"-{time.time_ns()}")
                candidate.rename(retained)
                events.append({"run_id": run_id, "action": "archived_without_deleting", "reason": reason,
                               "retained_path": str(retained.relative_to(root))})
        if valid:
            _, newest = max(valid, key=lambda item: item[0])
            pointer = {"directory": newest.name, "checkpoint_sha256": file_sha256(newest / "checkpoint.json")}
            latest = checkpoints / "latest.json"
            try:
                prior = json.loads(latest.read_text())
            except (OSError, json.JSONDecodeError):
                prior = None
            if prior != pointer:
                atomic_json(latest, pointer)
                events.append({"run_id": run_id, "action": "recovered_verified_checkpoint_pointer", "directory": newest.name})
    output = root / "results/runtime/resume_events"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"event-{time.time_ns()}.json", {"date_utc": datetime.now(timezone.utc).isoformat(), "events": events})
    return events
