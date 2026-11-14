"""Acquire only pinned Stage 2 inputs, then validate and freeze data manifests."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import (apply_eligibility, digest, file_sha256, find_overlaps, identify_rows,
                          make_splits, read_jsonl, save_json, save_jsonl, validate_s1k)


def download(url, destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def project_benchmark(url, destination, expected_count):
    if destination.exists():
        return
    rows = []
    # Upstream is JSONL, so transport includes all fields. Only these two fields
    # leave the transient parser. Gold answers/solutions are never saved/logged.
    with urllib.request.urlopen(url, timeout=60) as response:
        for index, line in enumerate(response):
            row = json.loads(line)
            question = row["problem"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Empty benchmark question at row {index}")
            rows.append({"id": str(row.get("unique_id", f"math500-{index:04d}")), "question": question})
    if len(rows) != expected_count or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Unexpected benchmark count or duplicate benchmark IDs")
    save_jsonl(destination, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Require existing files; make no network requests")
    args = parser.parse_args()
    import pyarrow.parquet as parquet
    import yaml
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "configs/sources.json").read_text())
    config = yaml.safe_load((root / "configs/dataset.yaml").read_text())
    raw = root / "data/raw"
    training = sources["training_dataset"]
    benchmark = sources["test_dataset"]
    tokenizer = sources["tokenizer"]
    train_path = raw / "s1k" / config["training_file"]
    benchmark_path = raw / "math500/questions.jsonl"
    files = [(f"https://huggingface.co/datasets/{training['id']}/resolve/{training['revision']}/{config['training_file']}", train_path)]
    files += [(f"https://huggingface.co/{tokenizer['id']}/resolve/{tokenizer['revision']}/{name}", raw / "tokenizer" / name)
              for name in config["tokenizer_files"]]
    for url, path in files:
        if args.offline and not path.exists():
            raise FileNotFoundError(path)
        if not args.offline:
            download(url, path)
    if args.offline and not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)
    if not args.offline:
        project_benchmark(f"https://huggingface.co/datasets/{benchmark['id']}/resolve/{benchmark['revision']}/{config['benchmark_file']}",
                          benchmark_path, config["expected_math500_rows"])

    rows = parquet.read_table(train_path).to_pylist()
    validation = validate_s1k(rows, config["expected_s1k_rows"])
    audit = root / "results/data_audit"
    save_json(audit / "schema.json", validation)
    if validation["problems"]:
        raise ValueError(f"Data validation failed; inspect {audit / 'schema.json'}")
    records = identify_rows(rows)
    splits = make_splits(records, training["revision"], config["seed"], config["development_size"])
    manifests = root / "manifests"
    save_json(manifests / "s1k_split.json", splits)
    questions = read_jsonl(benchmark_path)
    if (len(questions) != config["expected_math500_rows"]
            or len({row["id"] for row in questions}) != len(questions)
            or any(set(row) != {"id", "question"} or not isinstance(row["question"], str) or not row["question"].strip()
                   for row in questions)):
        raise ValueError("Benchmark projection failed validation")
    save_json(manifests / "math500_questions.json", {
        "dataset_id": benchmark["id"], "dataset_revision": benchmark["revision"],
        "split": "test", "projection": ["unique_id", "problem"],
        "rows": [{"id": row["id"], "question_sha256": digest(row["question"])} for row in questions],
    })
    overlaps = find_overlaps([{**record, "question": row["question"]} for record, row in zip(records, rows)], questions,
                             config["near_duplicate_threshold"], config["near_duplicate_minimum_tokens"])
    save_json(audit / "overlaps.json", overlaps)
    persisted = [path for _, path in files] + [benchmark_path]
    provenance = {
        "sources": {key: sources[key] for key in ("training_dataset", "test_dataset", "tokenizer")},
        "dataset_config_sha256": file_sha256(root / "configs/dataset.yaml"),
        "files": [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in persisted],
        "split_manifest_sha256": file_sha256(manifests / "s1k_split.json"),
        "benchmark_manifest_sha256": file_sha256(manifests / "math500_questions.json"),
        "benchmark_transfer": "Original JSONL streamed; only question and ID persisted; all other fields discarded without logging",
        "exclusions_applied": [],
    }
    save_json(manifests / "data_provenance.json", provenance)
    exclusions_path = root / "configs/data_exclusions.json"
    exclusions = json.loads(exclusions_path.read_text())["exclusions"]
    candidates = {(row["s1k_id"], row["benchmark_id"]) for row in overlaps["exact"] + overlaps["near_candidates"]}
    if any((row["s1k_id"], row["benchmark_id"]) not in candidates for row in exclusions):
        raise ValueError("Exclusions must refer to audited overlap candidates")
    eligible = apply_eligibility(splits, exclusions)
    eligible.update({"original_split_sha256": file_sha256(manifests / "s1k_split.json"),
                     "exclusion_config_sha256": file_sha256(exclusions_path)})
    save_json(manifests / "s1k_eligibility.json", eligible)
    print(json.dumps({"s1k_rows": len(rows), "train": len(splits["train"]), "development": len(splits["development"]),
                      "benchmark_questions": len(questions), "exact_overlap_pairs": len(overlaps["exact"]),
                      "near_overlap_candidates": len(overlaps["near_candidates"]),
                      "eligible_train": len(eligible["train"]), "eligible_development": len(eligible["development"]),
                      "eligible_final_training": len(eligible["final_training"]),
                      "persisted_input_bytes": sum(path.stat().st_size for path in persisted)}, indent=2))


if __name__ == "__main__":
    main()
