"""Grade immutable generated test records with the frozen local grader only."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import digest, file_sha256, save_json
from s1small.experiment import PredictionStore, verify_freeze
from s1small.grading import equivalent, grader_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args()
    current_grader = grader_manifest()
    if args.print_manifest:
        print(json.dumps(current_grader, indent=2)); return
    if not args.run_id or Path(args.run_id).name != args.run_id:
        parser.error("Provide a simple test run ID")
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads((root / "manifests/evaluation_freeze.json").read_text())
    verify_freeze(root, freeze)
    if current_grader != freeze["grader"]:
        raise ValueError("Grader settings/runtime differ from the pre-test freeze")
    directory = root / "results/test" / args.run_id
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["scope"] != "test" or manifest["evaluation_freeze_sha256"] != file_sha256(root / "manifests/evaluation_freeze.json"):
        raise ValueError("Generation did not use this frozen test protocol")
    planned = [(identifier, condition) for identifier in manifest["question_ids"] for condition in freeze["protocol"]["conditions"]]
    store = PredictionStore(directory, planned, digest(manifest))
    grades = directory / "grades"
    grades.mkdir(exist_ok=True)
    rows = []
    for record in store.records():
        key = store.key(record["question_id"], record["condition"])
        path = grades / (key + ".json")
        raw_sha = digest(record)
        if path.exists():
            grade = json.loads(path.read_text())
            if grade["generation_sha256"] != raw_sha or grade["grader_manifest_sha256"] != digest(current_grader):
                raise ValueError("Grade belongs to different generation/runtime; preserve it for review")
        else:
            outcome = (equivalent(record["gold_answer"], record.get("extracted_answer")) if record["status"] == "ok"
                       else {"is_correct": None, "grading_status": record["status"]})
            grade = {**outcome, "generation_sha256": raw_sha, "grader_manifest_sha256": digest(current_grader)}
            save_json(path, grade)
        rows.append({**record, **grade})
    output = directory / "evaluated.jsonl"
    temporary = directory / "evaluated.jsonl.incomplete"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    temporary.replace(output)
    print(json.dumps({"records": len(rows), "resolved_grades": sum(isinstance(row["is_correct"], bool) for row in rows),
                      "evaluated_sha256": file_sha256(output)}), flush=True)


if __name__ == "__main__":
    main()
