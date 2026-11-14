"""Finish the bounded Stage 3 smoke sequence once the checkpoint is verified."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="stage3-local-v1")
    parser.add_argument("--wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    if not args.prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.prefix):
        parser.error("Use letters, digits, underscores and hyphens in --prefix")
    if args.wait_seconds < 0:
        parser.error("--wait-seconds must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "configs/sources.json").read_text())
    checkpoint = root / "artifacts/base" / sources["model"]["revision"]
    deadline = time.monotonic() + args.wait_seconds
    print("Waiting for verified checkpoint provenance; no model is loaded while waiting", flush=True)
    while not (checkpoint / "provenance.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Checkpoint has not completed within the workflow wait allowance")
        time.sleep(5)
    runs = [("short", 1, 64), ("main", 5, 512), ("reload", 1, 512)]
    for suffix, count, tokens in runs:
        run_id = args.prefix + "-" + suffix
        destination = root / "results/dev_baselines" / run_id
        if destination.exists():
            summary_path = destination / "summary.json"
            if not summary_path.exists() or json.loads(summary_path.read_text())["status"] != "complete":
                raise RuntimeError(f"Existing incomplete run needs review: {run_id}")
            print(f"Retaining existing completed run {run_id}", flush=True)
            continue
        print(f"Starting {run_id}: {count} development questions, {tokens} tokens, 2 repetitions", flush=True)
        subprocess.run([
            sys.executable, str(root / "scripts/run_baselines.py"), "--run-id", run_id,
            "--samples", str(count), "--repeats", "2", "--max-new-tokens", str(tokens),
        ], cwd=root, check=True)
    subprocess.run([
        sys.executable, str(root / "scripts/report_baselines.py"), args.prefix + "-main",
        "--compare-with", args.prefix + "-reload",
    ], cwd=root, check=True)
    print("Stage 3 smoke workflow finished; inspect the saved audit summary before marking acceptance", flush=True)


if __name__ == "__main__":
    main()
