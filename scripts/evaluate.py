"""Run the nine conditions on development, or on the frozen 500-question test."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.baselines import load_development
from s1small.benchmark import sealed_test_rows
from s1small.budget_forcing import CONDITIONS, condition_budget, run_structured
from s1small.data import digest, file_sha256, save_json
from s1small.experiment import PredictionStore
from s1small.generate import build_prompt
from s1small.readout import extract_readout
from s1small.mlx_decode import load_decoder, runtime_signature, s1_prompt, token_protocol, unstructured_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["development", "test"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol-config", default="budget_protocol_v2.json")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        parser.error("Use a simple run ID")
    if Path(args.protocol_config).name != args.protocol_config or not args.protocol_config.endswith(".json"):
        parser.error("Use a protocol JSON filename in configs/")
    import mlx.core as mx
    root = Path(__file__).resolve().parents[1]
    training = json.loads((root / "manifests/training_freeze.json").read_text())
    templates = json.loads((root / "configs/prompts/baseline_v1.json").read_text())
    if args.scope == "test":
        rows, freeze = sealed_test_rows(root)
        protocol = freeze["protocol"]
        protocol_path = root / freeze.get("protocol_config_path", "configs/budget_protocol_v1.json")
        adapter = root / freeze["final_adapter_directory"]
        destination = root / "results/test" / args.run_id
    else:
        protocol_path = root / "configs" / args.protocol_config
        protocol = json.loads(protocol_path.read_text())
        # Use the selected 896-row model: these development targets were held
        # out from that model, even if all-data final retraining already ended.
        rows = load_development(root, 5)
        selected = json.loads((root / "results/training" / training["selected_run_id"] / "summary.json").read_text())
        adapter = root / selected["adapter_directory"]
        destination = root / "results/protocol_dev" / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    import fcntl
    run_lock = (destination / ".run.lock").open("w")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This evaluation run already has an active writer")
    manifest = {"scope": args.scope, "run_id": args.run_id, "protocol": protocol, "question_ids": [row["id"] for row in rows],
                "protocol_config_path": str(protocol_path.relative_to(root)), "protocol_config_sha256": file_sha256(protocol_path),
                "generation_runtime": runtime_signature(),
                "training_freeze_sha256": file_sha256(root / "manifests/training_freeze.json"),
                "adapter_directory": str(adapter.relative_to(root)), "adapter_weight_sha256": file_sha256(adapter / "adapters.safetensors"),
                "prompts_sha256": file_sha256(root / "configs/prompts/baseline_v1.json"),
                "evaluation_freeze_sha256": file_sha256(root / "manifests/evaluation_freeze.json") if args.scope == "test" else None,
                "condition_order": "question-order fixed; cyclic rotation of nine conditions by question index",
                "maximum_infrastructure_attempts": 2,
                "memory_accounting": "Per-prediction MLX active peak includes both resident base/adapted models; do not call it dedicated VRAM.",
                "code_sha256": {str(path.relative_to(root)): file_sha256(path) for path in [Path(__file__).resolve(),
                   *[root / "src/s1small" / name for name in ("budget_forcing.py", "mlx_decode.py", "experiment.py", "grading.py", "readout.py")]]}}
    save_json(destination / "manifest.json", manifest)
    store = PredictionStore(destination, [(row["id"], condition) for row in rows for condition in CONDITIONS], digest(manifest))
    started = time.perf_counter()
    base, base_tokenizer, conversion = load_decoder(root, training["training_recipe"])
    tuned, tokenizer, _ = load_decoder(root, training["training_recipe"], adapter)
    base.prefill_step_size = tuned.prefill_step_size = protocol["prefill_step_size"]
    readout_prefix = protocol.get("answer_readout_prefix", "")
    tokens = token_protocol(tokenizer, tuned.eos_ids, readout_prefix)
    save_json(destination / "token_protocol.json", {"reasoning_end": tokens.reasoning_end, "eos_ids": list(tokens.eos_ids),
              "answer_header": list(tokens.answer_header), "answer_readout_prefix": readout_prefix,
              "newline": list(tokens.newline), "wait": list(tokens.wait)})
    # Compile/warm both model paths on a development prompt, outside measured
    # benchmark generation and without consulting benchmark answers.
    warm = load_development(root, 1)[0]["question"]
    for decoder, prompt in ((base, build_prompt(warm, "A", templates, base_tokenizer)), (tuned, s1_prompt(warm, tokenizer))):
        decoder.generate(decoder.tokenizer.encode(prompt, add_special_tokens=False), 16, set(decoder.eos_ids), time.perf_counter() + 120)
    for sweep in range(2):
        for index, row in enumerate(rows):
            rotation = index % len(CONDITIONS)
            for condition in CONDITIONS[rotation:] + CONDITIONS[:rotation]:
                if not store.pending(row["id"], condition):
                    continue
                decoder = base if condition in ("A", "B") else tuned
                prompt = (build_prompt(row["question"], condition, templates, base_tokenizer) if condition in ("A", "B")
                          else s1_prompt(row["question"], tokenizer))
                input_ids = decoder.tokenizer.encode(prompt, add_special_tokens=False)
                metadata = {"scope": args.scope, "prompt": prompt, "prompt_sha256": digest(prompt),
                            "input_tokens": len(input_ids), "gold_answer": row.get("gold_answer"),
                            "model_source": conversion["source"], "quantization": conversion["quantization"],
                            "source_weight_sha256": conversion["source_weight_sha256"],
                            "adapter_weight_sha256": manifest["adapter_weight_sha256"] if decoder is tuned else None,
                            "dataset_revision": json.loads((root / "configs/sources.json").read_text())["test_dataset" if args.scope == "test" else "training_dataset"]["revision"],
                            "is_correct": None, "grading_status": "pending" if args.scope == "test" else "not_scored"}
                metadata.update(activation_dtype="float32", base_format="mlx4", decoder_seed=training["training_recipe"]["seed"],
                                protocol_id=digest(protocol), prefill_step_size=protocol["prefill_step_size"])
                attempt_started = time.perf_counter()
                mx.reset_peak_memory()
                decoder.unmapped.clear()
                try:
                    if condition in ("A", "B"):
                        result = unstructured_answer(decoder, input_ids,
                            protocol["natural_reasoning_ceiling"] + protocol["answer_budget"],
                            protocol["absolute_safety_ceiling"], protocol["timeout_seconds"])
                    else:
                        budget = condition_budget(condition, protocol["natural_reasoning_ceiling"], protocol["answer_budget"], protocol["absolute_safety_ceiling"])
                        result = run_structured(decoder, input_ids, budget, tokens, protocol["timeout_seconds"])
                    result.update(extract_readout(result["answer_text"], readout_prefix if condition not in ("A", "B") else ""))
                    result.update(peak_mlx_active_bytes=mx.get_peak_memory(), unmapped_token_ids=sorted(decoder.unmapped))
                    decoder.unmapped.clear()
                    saved = store.save_attempt(row["id"], condition, {**metadata, **result})
                    print(json.dumps({"id": row["id"], "condition": condition, "tokens": result["total_generated_tokens"],
                                      "seconds": result["latency_seconds"], "extraction": result["extraction_status"],
                                      "waits": result["forced_wait_count"]}), flush=True)
                except Exception as error:
                    store.save_attempt(row["id"], condition, {**metadata, "status": "infrastructure_failure",
                        "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                        "latency_seconds": time.perf_counter() - attempt_started})
                    print(json.dumps({"id": row["id"], "condition": condition, "infrastructure_failure": str(error)}), flush=True)
                mx.clear_cache()
    records = store.records()
    outputs = sum(row["status"] == "ok" for row in records)
    summary = {"status": "complete" if outputs == len(records) else "partial", "scope": args.scope,
               "planned_records": len(records), "successful_model_outputs": outputs,
               "extracted_answers": sum(row.get("extracted_answer") is not None for row in records),
               "question_count": len(rows), "test_scored": False,
               "wall_seconds_this_invocation": time.perf_counter() - started,
               "marker_stops": sum(e.get("removed_stop_token_id") == tokens.reasoning_end for row in records for e in row.get("segments", [])),
               "eos_reasoning_stops": sum(bool(row.get("early_reasoning_eos")) for row in records),
               "wait_triggered_outputs": sum(row.get("forced_wait_count", 0) > 0 for row in records)}
    # Derived status may be refreshed after bounded infrastructure retries.
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if outputs != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
