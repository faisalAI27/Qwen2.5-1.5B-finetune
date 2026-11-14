"""Export small measured profile/pilot summaries; never infer successful training."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256


def main():
    root = Path(__file__).resolve().parents[1]
    reports = []
    for directory in sorted((root / "results/training_pilot").iterdir()):
        if not directory.is_dir() or not (directory / "manifest.json").exists():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        report = {"run_id": directory.name, "mode": manifest["mode"],
                  "config": manifest["config"], "weights_sha256": manifest["weights_sha256"],
                  "manifest_sha256": file_sha256(directory / "manifest.json"),
                  "loss_implementation": manifest.get("loss_implementation", "checkpointed_masked_chunks" if manifest["config"].get("backend") == "mlx_lm" else "standard"),
                  "loss_chunk_tokens": manifest.get("loss_chunk_tokens"), "status": "running_or_interrupted"}
        for name in ("examples", "forward", "summary", "memory"):
            path = directory / (name + ".json")
            if path.exists():
                report[name] = json.loads(path.read_text())
        if "summary" in report:
            report["status"] = report["summary"]["status"]
        failure = directory / "failure.json"
        if failure.exists():
            data = json.loads(failure.read_text())
            report.update(status="failed", error_type=data["error_type"], error=data["error"],
                          optimizer_updates_completed=data["optimizer_updates_completed"], memory=data["memory"])
        interruption = directory / "interruption.json"
        if interruption.exists():
            report["interruption"] = json.loads(interruption.read_text())
            report["status"] = report["interruption"]["status"]
        reports.append(report)
    output = {"scope": "technical_feasibility_only_not_accuracy", "runs": reports,
              "note": "Raw logs under ignored results/training_pilot; running_or_interrupted is not completion."}
    (root / "docs/training_pilot_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps([{ "run_id": row["run_id"], "status": row["status"] } for row in reports], indent=2))


if __name__ == "__main__":
    main()
