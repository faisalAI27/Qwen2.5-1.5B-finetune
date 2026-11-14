"""Run A/B on a small fixed development subset; never reads MATH-500."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.baselines import load_development, validate_config
from s1small.data import canonical_json, digest, file_sha256, save_json
from s1small.environment import collect_environment
from s1small.generate import generate_one, repeat_comparison
from s1small.model import load_base_model, verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="New output directory name (default: timestamp)")
    parser.add_argument("--samples", type=int, help="Use first N eligible development IDs (1-5)")
    parser.add_argument("--repeats", type=int, help="Repeat each condition (1-3)")
    parser.add_argument("--max-new-tokens", type=int, help="Explicit development smoke-test override")
    args = parser.parse_args()
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/baseline.yaml").read_text())
    for argument, key in [(args.samples, "sample_count"), (args.repeats, "repeats"), (args.max_new_tokens, "max_new_tokens")]:
        if argument is not None:
            config[key] = argument
    validate_config(config)
    sources = json.loads((root / "configs/sources.json").read_text())
    templates = json.loads((root / config["prompts_file"]).read_text())
    selected = load_development(root, config["sample_count"])
    directory = root / "artifacts/base" / sources["model"]["revision"]
    checkpoint = verify_checkpoint(directory, sources)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("stage3-%Y%m%dT%H%M%S%fZ")
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
        parser.error("--run-id may contain only letters, digits, underscores and hyphens")
    output = root / "results/dev_baselines" / run_id
    output.mkdir(parents=True, exist_ok=False)
    config_id = digest(config)
    code_paths = [*sorted((root / "src/s1small").glob("*.py")), Path(__file__).resolve()]
    manifest = {
        "schema_version": 1, "run_id": run_id, "scope": "development_smoke_only",
        "config": config, "config_sha256": config_id,
        "source_config_sha256": file_sha256(root / "configs/baseline.yaml"),
        "sources": sources, "prompts": templates, "prompts_sha256": digest(templates),
        "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json"),
        "checkpoint_provenance_sha256": file_sha256(directory / "provenance.json"),
        "weights_sha256": next(row["sha256"] for row in checkpoint["files"] if row["name"] == "model.safetensors"),
        "question_ids": [row["id"] for row in selected],
        "planned_records": len(selected) * 2 * config["repeats"],
        "warmup_performed": False,
        "latency_note": "First generation may include backend compilation; no cross-condition timing claim is made from this smoke test",
        "environment": collect_environment(root),
        "code_sha256": {str(path.relative_to(root)): file_sha256(path) for path in code_paths},
    }
    save_json(output / "manifest.json", manifest)
    try:
        model, tokenizer, model_info = load_base_model(directory, config)
    except Exception as error:
        save_json(output / "load_failure.json", {"status": "failed", "error_type": type(error).__name__,
                                                  "error": str(error), "traceback": traceback.format_exc()})
        raise
    save_json(output / "model.json", model_info)
    print(json.dumps({"run_id": run_id, "model": model_info, "planned_records": manifest["planned_records"]}), flush=True)
    records = []
    halted = None
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for repeat in range(config["repeats"]):
            for question in selected:
                for condition in ("A", "B"):
                    record = {
                        "schema_version": 1, "scope": "development_smoke_only", "run_id": run_id,
                        "question_id": question["id"], "source_row": question["source_row"], "question": question["question"],
                        "dataset_id": sources["training_dataset"]["id"], "dataset_revision": sources["training_dataset"]["revision"],
                        "model_id": sources["model"]["id"], "model_revision": sources["model"]["revision"],
                        "tokenizer_revision": sources["tokenizer"]["revision"], "adapter_id": None,
                        "weights_sha256": manifest["weights_sha256"], "condition": condition,
                        "repeat": repeat, "run_config_id": config_id,
                        "decoder": {key: config[key] for key in ("do_sample", "num_beams", "max_new_tokens", "seed", "timeout_seconds")},
                        "is_correct": None, "gold_answer": None, "grading_status": "not_scored",
                    }
                    if halted is not None:
                        record.update(status="not_attempted", error=halted, raw_output=None, generated_token_ids=None,
                                      extracted_answer=None, extraction_status="unavailable", total_generated_tokens=None,
                                      reasoning_tokens=None, latency_seconds=None, termination_reason="previous_infrastructure_failure")
                    else:
                        try:
                            record.update(generate_one(model, tokenizer, question["question"], condition, templates, config))
                        except Exception as error:
                            halted = f"{type(error).__name__}: {error}"
                            record.update(status="error", error=halted, raw_output=None, generated_token_ids=None,
                                          extracted_answer=None, extraction_status="unavailable", total_generated_tokens=None,
                                          reasoning_tokens=None, latency_seconds=None, termination_reason="infrastructure_failure",
                                          traceback=traceback.format_exc())
                    stream.write(canonical_json(record) + "\n")
                    stream.flush()
                    records.append(record)
                    print(json.dumps({"question_id": question["id"], "condition": condition, "repeat": repeat,
                                      "status": record["status"], "tokens": record["total_generated_tokens"],
                                      "seconds": record["latency_seconds"], "termination": record["termination_reason"],
                                      "extraction": record["extraction_status"]}), flush=True)
    repeat_check = repeat_comparison(records)
    summary = {"run_id": run_id, "planned_records": manifest["planned_records"], "saved_records": len(records),
               "successful_generations": sum(row["status"] == "ok" for row in records),
               "extracted_answers": sum(row.get("extracted_answer") is not None for row in records),
               "status": "complete" if halted is None else "incomplete", "repeats": repeat_check}
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if halted is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
