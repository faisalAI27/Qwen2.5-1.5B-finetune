"""Package only inference inputs, preserving the active Mac run unchanged."""

import hashlib
import argparse
import json
from pathlib import Path
import sys
import subprocess
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import digest, file_sha256
from s1small.experiment import verify_freeze


def build(root, destination, include_code=False):
    root, destination = Path(root), Path(destination)
    original = root / "manifests/evaluation_freeze.json"
    parent = json.loads(original.read_text())
    verify_freeze(root, parent)
    selected = json.loads((root / "manifests/training_freeze.json").read_text())["selected_run_id"]
    paths = set(parent["files_sha256"])
    for folder, patterns in (("scripts", ("*.py",)), ("src", ("*.py",)), ("configs", ("*.json", "*.yaml")),
                             ("manifests", ("*.json",)), ("data/raw/tokenizer", ("*",))):
        for pattern in patterns:
            paths.update(str(p.relative_to(root)) for p in (root / folder).rglob(pattern) if p.is_file())
    paths.update(str(p.relative_to(root)) for p in root.glob("requirements*.txt"))
    paths.update(["README.md", "pyproject.toml", "docs/colab_inference.md", "docs/license_metadata.json",
                  "docs/hardware.md", "notebooks/colab_inference.ipynb", "manifests/data_provenance.json",
                  "data/raw/s1k/data/train-00000-of-00001.parquet",
                  f"results/training/{selected}/manifest.json", f"results/training/{selected}/summary.json"])
    paths.discard("manifests/evaluation_freeze.json")
    # Gold answers, live Mac predictions, optimizer states and environments are
    # intentionally absent. Only frozen model files are included from artifacts/.
    reference = []
    directory = root / "results/protocol_dev" / parent["development_run_id"]
    raw_records = {}
    for path in sorted((directory / "slots").glob("*.json")):
        row = json.loads(path.read_text())
        key = (row["question_id"], row["condition"])
        if key in raw_records or row["status"] != "ok":
            raise ValueError("Invalid or incomplete Mac development reference")
        raw_records[key] = row
        reference.append({"question_id": row["question_id"], "condition": row["condition"],
            "model_token_segments": [e["token_ids"] for e in row["segments"] if e["origin"] == "model"],
            "latency_seconds": row["latency_seconds"]})
    if len(reference) != 45:
        raise ValueError("The complete 45-output Mac development reference is required")
    ordered = [raw_records[(key, condition)] for key in parent["development_question_ids"] for condition in parent["planned_conditions"]]
    if digest(ordered) != parent["development_outputs_sha256"]:
        raise ValueError("Mac development outputs differ from the frozen reference")
    virtual = {"manifests/evaluation_freeze_macos.json": original.read_bytes(),
               "COLAB_ISOLATED_WORKSPACE": b"Separate CUDA replication. Never overlay the active Mac project.\n",
               "colab_macos_development_reference.json": (json.dumps(reference) + "\n").encode()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.incomplete")
    hashes, total = {}, 0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    source_commit = commit.stdout.strip() if commit.returncode == 0 else None
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(paths):
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"Missing or out-of-workspace bundle input: {name}")
            member_hash = hashlib.sha256()
            code = (name.startswith(("scripts/", "src/", "configs/", "notebooks/"))
                    or name.startswith("requirements") or name in ("README.md", "pyproject.toml"))
            if code and not include_code:
                hashes[name] = file_sha256(path)
                continue
            with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as output:
                while chunk := source.read(1024 * 1024):
                    member_hash.update(chunk); output.write(chunk); total += len(chunk)
            hashes[name] = member_hash.hexdigest()
            if name in parent["files_sha256"] and hashes[name] != parent["files_sha256"][name]:
                raise ValueError("Frozen file changed during transfer packaging")
        for name, content in virtual.items():
            archive.writestr(name, content)
            hashes[name] = hashlib.sha256(content).hexdigest()
            total += len(content)
        metadata = {"scope": "separate_cuda_replication_no_training", "source_commit": source_commit,
                    "bundle_mode": "complete" if include_code else "assets_with_github_code",
                    "parent_macos_freeze_sha256": file_sha256(original),
                    "uncompressed_payload_bytes": total, "files_sha256": hashes,
                    "cuda_tested": False, "mac_benchmark_predictions_included": False}
        archive.writestr("colab_bundle_manifest.json", json.dumps(metadata, indent=2) + "\n")
    temporary.replace(destination)
    archive_hash = file_sha256(destination)
    destination.with_suffix(".zip.sha256").write_text(archive_hash + "  " + destination.name + "\n")
    return {"archive": str(destination), "bytes": destination.stat().st_size, "sha256": archive_hash,
            "files": len(hashes), "cuda_tested": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-code", action="store_true", help="Make a complete bundle without requiring a GitHub clone")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    name = "nlp-colab-inference.zip" if args.include_code else "nlp-colab-assets.zip"
    print(json.dumps(build(root, root / "artifacts/colab" / name, args.include_code), indent=2))
