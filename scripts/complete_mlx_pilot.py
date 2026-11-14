"""Run bounded local QLoRA gates sequentially, stopping at any measured failure."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="mlx-stage5-qlora-v1")
    args = parser.parse_args()
    if not args.prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.prefix):
        parser.error("Use a simple run prefix")
    root = Path(__file__).resolve().parents[1]
    base = root / "results/training_pilot"
    phases = [("short", "profile", []), ("2000", "profile", ["--profile-target-tokens", "2000"]),
              ("pilot20", "pilot", []), ("pilot20", "verify", []),
              ("6000", "profile", ["--profile-target-tokens", "6000"]),
              ("max-train", "profile", ["--profile-target-tokens", "10000"]),
              ("max-dev", "profile", ["--profile-target-tokens", "10000", "--profile-partition", "development"])]
    reports = {}
    for suffix, mode, extra in phases:
        run_id = args.prefix + "-" + suffix
        directory = base / run_id
        path = directory / ("training_summary.json" if mode == "pilot" else "summary.json")
        if path.exists():
            summary = json.loads(path.read_text())
            if summary["status"] not in ("complete", "trained_reload_pending"):
                raise RuntimeError(f"Existing unsuccessful result requires review: {run_id}")
            print(f"Retaining measured result {run_id} / {mode}", flush=True)
        else:
            if directory.exists() and mode != "verify":
                raise RuntimeError(f"Existing incomplete run retained for review: {run_id}")
            print(f"Starting {run_id} / {mode}", flush=True)
            subprocess.run([sys.executable, str(root / "scripts/train_mlx_pilot.py"), "--mode", mode,
                            "--run-id", run_id, "--base-format", "mlx4", "--float32-activations", *extra],
                           cwd=root, check=True)
            summary = json.loads(path.read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["config"]["base_format"] != "mlx4" or manifest["config"]["activation_dtype"] != "float32":
            raise ValueError("Different backend/precision under the workflow prefix")
        reports[suffix + "-" + mode] = summary
    subprocess.run([sys.executable, str(root / "scripts/report_training_pilot.py")], cwd=root, check=True)
    accepted = {"status": "complete", "scope": "stage5_technical_feasibility_only", "runs": reports,
                "note": "Longest development example was profiled with zero updates and discarded. Only eligible training rows received pilot optimizer updates. No benchmark scores."}
    destination = base / (args.prefix + "-acceptance.json")
    if destination.exists() and json.loads(destination.read_text()) != accepted:
        raise ValueError("Refusing to replace different acceptance evidence")
    destination.write_text(json.dumps(accepted, indent=2) + "\n")
    print("MLX Stage 5 workflow passed; inspect acceptance evidence before starting full training", flush=True)


if __name__ == "__main__":
    main()
