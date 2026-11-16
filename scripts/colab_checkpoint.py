"""Save/restore only the isolated CUDA run's mutable state to its Drive folder."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256
from s1small.experiment import atomic_json


def state_path(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts and
            (name.startswith("results/") or name.startswith("docs/results") or name == "manifests/evaluation_freeze.json"))


def checkpoint(root, drive):
    root, drive = Path(root), Path(drive)
    drive.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    local = root / "artifacts" / ("cuda-state-" + stamp + ".zip")
    local.parent.mkdir(exist_ok=True)
    metadata = {"bundle_manifest_sha256": file_sha256(root / "colab_bundle_manifest.json"), "files_sha256": {}}
    with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for folder in (root / "results", root / "docs", root / "manifests"):
            for path in folder.rglob("*"):
                name = str(path.relative_to(root))
                if not path.is_file() or not state_path(name) or path.name.endswith((".lock", ".incomplete")):
                    continue
                # Read each atomic slot once so content and checksum agree even
                # while the inference child is completing other predictions.
                content = path.read_bytes()
                import hashlib
                metadata["files_sha256"][name] = hashlib.sha256(content).hexdigest()
                archive.writestr(name, content)
        archive.writestr("checkpoint_metadata.json", json.dumps(metadata))
    temporary = drive / (local.name + ".incomplete")
    shutil.copyfile(local, temporary)
    destination = drive / local.name
    temporary.replace(destination)
    atomic_json(drive / "LATEST.json", {"name": destination.name, "sha256": file_sha256(local)})
    local.unlink()
    # Only this helper's own completed snapshots are pruned, retaining two.
    for old in sorted(drive.glob("cuda-state-*.zip"))[:-2]:
        old.unlink()
    print(json.dumps({"saved_checkpoint": str(destination), "files": len(metadata["files_sha256"])}))


def restore(root, drive):
    root, drive = Path(root), Path(drive)
    latest = drive / "LATEST.json"
    if not latest.exists():
        print("No prior CUDA checkpoint; starting the separate CUDA run")
        return
    record = json.loads(latest.read_text())
    if Path(record["name"]).name != record["name"]:
        raise ValueError("Invalid checkpoint name")
    path = drive / record["name"]
    if file_sha256(path) != record["sha256"]:
        raise ValueError("Drive checkpoint checksum differs")
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("checkpoint_metadata.json"))
        if metadata["bundle_manifest_sha256"] != file_sha256(root / "colab_bundle_manifest.json"):
            raise ValueError("Checkpoint belongs to a different source/model bundle")
        import hashlib
        for name, expected in metadata["files_sha256"].items():
            if not state_path(name):
                raise ValueError("Checkpoint attempted to change immutable source/model files")
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Checkpoint member checksum differs")
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != content:
                if name.endswith("manifest.json") or name == "manifests/evaluation_freeze.json":
                    raise ValueError("Existing local run has a different immutable manifest")
                if "/attempts/" in name and name.endswith(".json"):
                    raise ValueError("Existing durable attempt differs; preserve both for review")
                # Keep newer local prediction slots; never replace a completed
                # model output with an older Drive snapshot or reroll it.
                if "/slots/" in name and name.endswith(".json"):
                    local, saved = json.loads(destination.read_text()), json.loads(content)
                    if local["manifest_id"] != saved["manifest_id"]:
                        raise ValueError("Local slot belongs to a different run")
                    if local["status"] == saved["status"] == "ok" and local != saved:
                        raise ValueError("Conflicting completed outputs must both be retained for review")
                    if local["status"] == "ok" or local["attempt"] >= saved["attempt"]:
                        continue
                else:
                    continue  # Preserve locally newer logs/derived summaries.
            destination.write_bytes(content)
    print(json.dumps({"restored_checkpoint": str(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("save", "restore"))
    parser.add_argument("--drive-folder", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (root / "COLAB_ISOLATED_WORKSPACE").exists():
        raise ValueError("Checkpoint only an extracted isolated Colab workspace")
    (checkpoint if args.mode == "save" else restore)(root, Path(args.drive_folder))
