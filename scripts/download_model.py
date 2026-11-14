"""Download and checksum the pinned base checkpoint, without loading it."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Verify already-downloaded files without network")
    parser.add_argument("--transport", choices=["http", "xet", "ranges"], default="ranges", help="Bounded HTTP ranges can resume a slow serial transfer")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "configs/sources.json").read_text())
    source = sources["model"]
    destination = root / "artifacts/base" / source["revision"]
    manifest_path = destination / "provenance.json"
    if args.offline:
        provenance = json.loads(manifest_path.read_text())
        if provenance["source"] != source:
            raise ValueError("Checkpoint source differs from pinned configuration")
        for record in provenance["files"]:
            if file_sha256(destination / record["name"]) != record["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {record['name']}")
        print("Offline checkpoint verification passed")
        return
    url = f"https://huggingface.co/api/models/{source['id']}/tree/{source['revision']}?recursive=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        tree = {row["path"]: row for row in json.load(response) if row["type"] == "file"}
    requested = ["config.json", "generation_config.json", "model.safetensors"]
    total = sum(tree[name]["size"] for name in requested)
    remaining = sum(tree[name]["size"] for name in requested if not (destination / name).exists())
    if shutil.disk_usage(root).free < remaining + 2 * 1024**3:
        raise RuntimeError("Insufficient disk space for checkpoint plus 2 GiB working allowance")
    print(f"Pinned checkpoint files total {total:,} bytes; missing {remaining:,} bytes", flush=True)
    os.environ.setdefault("HF_XET_CACHE", str(root / "data/xet"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ["HF_HUB_DISABLE_XET"] = "0" if args.transport == "xet" else "1"
    from huggingface_hub import hf_hub_download
    for name in requested:
        print(f"Acquiring {name}", flush=True)
        if name == "model.safetensors" and args.transport == "ranges":
            from s1small.download import download_ranges
            lfs = tree[name].get("lfs", {})
            expected = lfs.get("oid") or lfs.get("sha256")
            if not expected:
                raise ValueError("Upstream checkpoint SHA-256 is required")
            download_ranges(f"https://huggingface.co/{source['id']}/resolve/{source['revision']}/{name}",
                            destination / name, tree[name]["size"], expected)
        else:
            hf_hub_download(repo_id=source["id"], filename=name, revision=source["revision"],
                            local_dir=destination, cache_dir=root / "data/hub", token=False)
    # Copy the exact tokenizer assets audited in Stage 2, not a new template.
    for path in sorted((root / "data/raw/tokenizer").iterdir()):
        if path.is_file():
            target = destination / path.name
            if target.exists() and file_sha256(target) != file_sha256(path):
                raise ValueError(f"Tokenizer differs from Stage 2: {path.name}")
            if not target.exists():
                shutil.copyfile(path, target)
    records = []
    for path in sorted(destination.iterdir()):
        if not path.is_file() or path.name == "provenance.json":
            continue
        observed = file_sha256(path)
        expected = tree.get(path.name, {}).get("lfs", {})
        expected = expected.get("oid") or expected.get("sha256")
        if path.name == "model.safetensors" and not expected:
            raise ValueError("Upstream weight SHA-256 is required for checkpoint verification")
        if expected and observed != expected:
            raise ValueError(f"Upstream LFS checksum mismatch: {path.name}")
        records.append({"name": path.name, "bytes": path.stat().st_size, "sha256": observed,
                        "upstream_lfs_sha256": expected})
    save_json(manifest_path, {"source": source, "tokenizer_source": sources["tokenizer"], "files": records})
    print(json.dumps({"checkpoint_directory": str(destination.relative_to(root)),
                      "files": len(records), "stored_bytes": sum(row["bytes"] for row in records)}, indent=2))


if __name__ == "__main__":
    main()
