"""Independent, resumable vLLM evaluation of the reconstructed frozen Q4 model."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import fcntl
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.baselines import load_development
from s1small.budget_forcing import CONDITIONS, Budget, condition_budget, run_structured
from s1small.data import digest, file_sha256, save_json
from s1small.experiment import PredictionStore, atomic_json, verify_freeze
from s1small.generate import build_prompt
from s1small.mlx_decode import s1_prompt, token_protocol, unstructured_answer
from s1small.readout import extract_readout
from s1small.vllm_export import export, verify_export

DEV_RUN = "runpod-vllm-dev-v1"
TEST_RUN = "math500-runpod-vllm-v1"
EXPORT = "artifacts/vllm-export"
NEW_FILES = ("scripts/runpod_vllm.py", "src/s1small/vllm_backend.py", "src/s1small/vllm_export.py",
             "configs/runpod_vllm_v1.json", "configs/vllm_prompt_reference_v1.json", "requirements-vllm.txt")


def verify_bundle(root):
    bundle = json.loads((root / "colab_bundle_manifest.json").read_text())
    for name, expected in bundle["files_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("Transferred frozen file differs: " + name)
    launcher = json.loads((root / "vllm_launcher_manifest.json").read_text())
    if set(launcher["files_sha256"]) != set(NEW_FILES):
        raise ValueError("Incomplete vLLM launcher manifest")
    for name, expected in launcher["files_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("vLLM launcher file differs: " + name)
    parent = json.loads((root / "manifests/evaluation_freeze_macos.json").read_text())
    verify_freeze(root, parent)
    return parent


def activate_identity():
    import s1small.mlx_decode as original
    from s1small.vllm_backend import runtime_signature
    # sealed_test_rows checks this hook before opening any benchmark answers.
    original.runtime_signature = runtime_signature
    return runtime_signature()


def verify_prompts(root, tokenizer=None):
    from transformers import AutoTokenizer
    sources = json.loads((root / "configs/sources.json").read_text())
    base = root / "artifacts/base-mlx4" / (sources["model"]["revision"] + "-g64")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    reference = json.loads((root / "configs/vllm_prompt_reference_v1.json").read_text())
    if file_sha256(base / "tokenizer.json") != reference["tokenizer_source_sha256"]:
        raise ValueError("Tokenizer differs from the prompt reference")
    templates = json.loads((root / "configs/prompts/baseline_v1.json").read_text())
    rows = {row["id"]: row for row in load_development(root, 5)}
    if len(reference["rows"]) != 15:
        raise ValueError("All 15 held-out prompt variants must be checked")
    for item in reference["rows"]:
        row, condition = rows[item["question_id"]], item["condition"]
        prompt = build_prompt(row["question"], condition, templates, tokenizer) if condition in ("A", "B") else s1_prompt(row["question"], tokenizer)
        if digest(prompt) != item["prompt_sha256"] or tokenizer.encode(prompt, add_special_tokens=False) != item["token_ids"]:
            raise ValueError("Prompt or input token IDs changed with the vLLM tokenizer runtime")


def require_smoke(root, runtime):
    for path in (root / "results/vllm_smoke").glob("run-*.json"):
        value = json.loads(path.read_text())
        if (value.get("status") == "complete" and value.get("runtime") == runtime
                and value.get("stop_token_probe") and value.get("exported_files_unchanged")):
            return
    raise ValueError("A successful vLLM smoke on this exact runtime/export is required first")


def directory_for(root, scope):
    return root / ("results/protocol_dev/" + DEV_RUN if scope == "development" else "results/test/" + TEST_RUN)


def export_outputs(directory, records):
    path = directory / "predictions.jsonl"
    temporary = path.with_suffix(".jsonl.incomplete")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
    temporary.replace(path)


async def smoke(root, parent, runtime):
    from s1small.vllm_backend import AsyncBridge, VLLMDecoder, create_engine
    from vllm import SamplingParams
    settings = runtime["engine_settings"]
    engine, tokenizer, eos, lora = create_engine(root, settings)
    bridge = AsyncBridge(engine, asyncio.get_running_loop(), settings)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        row = load_development(root, 1)[0]
        prompt = s1_prompt(row["question"], tokenizer)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # Force a known stop only to test API accounting. This is distinct
        # from the following genuine model-output smoke, not an accuracy test.
        forced = []
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, stop_token_ids=[eos[0]],
                                allowed_token_ids=[eos[0]], detokenize=False, skip_special_tokens=False)
        async for output in engine.generate(prompt={"prompt_token_ids": ids}, sampling_params=params,
                request_id="counted-stop-probe", lora_request=lora):
            forced = list(output.outputs[0].token_ids)
        if forced != [eos[0]]:
            raise ValueError("vLLM stripped or changed a stop token; do not evaluate with incorrect budgets")
        decoder = VLLMDecoder(bridge, tokenizer, eos, lora)
        prefix = parent["protocol"].get("answer_readout_prefix", "")
        tokens = token_protocol(tokenizer, eos, prefix)
        result = await asyncio.get_running_loop().run_in_executor(pool, partial(run_structured,
            decoder, ids, Budget("D256", 256, 256, 1024), tokens, 120))
        if not result["answer_text"].strip() or result["timed_out"]:
            raise ValueError("vLLM smoke failed to complete a nonempty readout")
        result.update(extract_readout(result["answer_text"], prefix))
        verified = verify_export(root, root / EXPORT)
        record = {"status": "complete", "scope": "vllm_development_smoke_not_accuracy", "runtime": runtime,
            "stop_token_probe": True, "exported_files_unchanged": True,
            "original_adapter_tensor_sha256": verified["adapter"]["original_adapter_tensor_sha256"],
            "export_manifest_sha256": file_sha256(root / EXPORT / "export_manifest.json"),
            "result": result, "optimizer_updates": 0}
        path = root / "results/vllm_smoke" / f"run-{time.time_ns()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, record)
        print(json.dumps({"smoke": "complete", "stop_token_probe": True, "seconds": result["latency_seconds"],
                          "answer": result["answer_text"], "saved": str(path), "optimizer_updates": 0}), flush=True)
    finally:
        bridge.close()
        pool.shutdown(wait=True, cancel_futures=True)
        engine.shutdown()


async def evaluate(root, parent, runtime, scope):
    from s1small.vllm_backend import AsyncBridge, VLLMDecoder, create_engine
    settings = runtime["engine_settings"]
    if scope == "test":
        from s1small.benchmark import sealed_test_rows
        (root / "data/raw/math500").mkdir(parents=True, exist_ok=True)
        rows, freeze = sealed_test_rows(root)
        protocol = freeze["protocol"]
    else:
        rows, freeze = load_development(root, 5), None
        protocol = parent["protocol"]
        if [r["id"] for r in rows] != parent["development_question_ids"]:
            raise ValueError("Development IDs differ from the original held-out plan")
    directory = directory_for(root, scope)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {"scope": scope, "run_id": DEV_RUN if scope == "development" else TEST_RUN,
        "protocol": protocol, "question_ids": [r["id"] for r in rows], "generation_runtime": runtime,
        "protocol_config_path": parent["protocol_config_path"],
        "protocol_config_sha256": file_sha256(root / parent["protocol_config_path"]),
        "training_freeze_sha256": file_sha256(root / "manifests/training_freeze.json"),
        "adapter_directory": EXPORT + "/adapter",
        "adapter_weight_sha256": file_sha256(root / EXPORT / "adapter/adapter_model.safetensors"),
        "export_manifest_sha256": file_sha256(root / EXPORT / "export_manifest.json"),
        "evaluation_freeze_sha256": file_sha256(root / "manifests/evaluation_freeze.json") if freeze else None,
        "prompts_sha256": file_sha256(root / "configs/prompts/baseline_v1.json"),
        "condition_order": "Same question order and cyclic condition submission rotation, up to eight concurrent independent prediction state machines.",
        "maximum_infrastructure_attempts": 2, "latency_interpretation": settings["latency_interpretation"],
        "source_code_sha256": runtime["code_sha256"]}
    save_json(directory / "manifest.json", manifest)
    planned = [(r["id"], condition) for r in rows for condition in CONDITIONS]
    store = PredictionStore(directory, planned, digest(manifest))
    jobs = [(r, c) for i, r in enumerate(rows) for c in CONDITIONS[i % 9:] + CONDITIONS[:i % 9]]
    templates = json.loads((root / "configs/prompts/baseline_v1.json").read_text())
    conversion = json.loads((root / EXPORT / "export_manifest.json").read_text())
    sources = json.loads((root / "configs/sources.json").read_text())
    started = time.perf_counter()
    newly_generated = []
    engine = bridge = pool = None
    try:
        engine, tokenizer, eos, lora = create_engine(root, settings)
        loop = asyncio.get_running_loop()
        bridge = AsyncBridge(engine, loop, settings)
        pool = ThreadPoolExecutor(max_workers=settings["concurrency"])
        prefix = protocol.get("answer_readout_prefix", "")
        tokens = token_protocol(tokenizer, eos, prefix)
        save_json(directory / "token_protocol.json", {"reasoning_end": tokens.reasoning_end, "eos_ids": list(tokens.eos_ids),
            "answer_header": list(tokens.answer_header), "newline": list(tokens.newline), "wait": list(tokens.wait),
            "answer_readout_prefix": prefix})
        # Both base and adapted paths warm on held-out development only.
        warm = load_development(root, 1)[0]["question"]
        for request, prompt in ((None, build_prompt(warm, "A", templates, tokenizer)), (lora, s1_prompt(warm, tokenizer))):
            await bridge.collect(tokenizer.encode(prompt, add_special_tokens=False), 16, set(eos), time.perf_counter() + 120, request)
        inference_started = time.perf_counter()
        def generate(row, condition):
            decoder = VLLMDecoder(bridge, tokenizer, eos, None if condition in ("A", "B") else lora)
            prompt = build_prompt(row["question"], condition, templates, tokenizer) if condition in ("A", "B") else s1_prompt(row["question"], tokenizer)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            metadata = {"scope": scope, "prompt": prompt, "prompt_sha256": digest(prompt), "input_tokens": len(ids),
                "gold_answer": row.get("gold_answer"), "model_source": conversion["source"],
                "base_format": "dense_fp16_reconstructed_from_frozen_mlx_q4", "activation_dtype": "float16",
                "quantization": {"source_bits": 4, "source_group_size": 64, "runtime": "dequantized_fp16"},
                "export_manifest_sha256": manifest["export_manifest_sha256"],
                "adapter_weight_sha256": manifest["adapter_weight_sha256"] if condition not in ("A", "B") else None,
                "dataset_revision": sources["test_dataset" if scope == "test" else "training_dataset"]["revision"],
                "is_correct": None, "grading_status": "pending" if scope == "test" else "not_scored",
                "decoder_seed": settings["seed"], "protocol_id": digest(protocol),
                "latency_interpretation": settings["latency_interpretation"], "peak_mlx_active_bytes": None}
            attempted = time.perf_counter()
            try:
                if condition in ("A", "B"):
                    result = unstructured_answer(decoder, ids, protocol["natural_reasoning_ceiling"] + protocol["answer_budget"],
                        protocol["absolute_safety_ceiling"], protocol["timeout_seconds"])
                else:
                    budget = condition_budget(condition, protocol["natural_reasoning_ceiling"], protocol["answer_budget"], protocol["absolute_safety_ceiling"])
                    result = run_structured(decoder, ids, budget, tokens, protocol["timeout_seconds"])
                result.update(extract_readout(result["answer_text"], prefix if condition not in ("A", "B") else ""))
                result["unmapped_token_ids"] = sorted(decoder.unmapped)
                return {**metadata, **result}
            except Exception as error:
                return {**metadata, "status": "infrastructure_failure", "error_type": type(error).__name__,
                    "error": str(error), "traceback": traceback.format_exc(), "latency_seconds": time.perf_counter() - attempted}
        for sweep in range(2):
            queue = asyncio.Queue()
            for row, condition in jobs:
                if store.pending(row["id"], condition):
                    queue.put_nowait((row, condition))
            async def worker():
                while not queue.empty():
                    row, condition = queue.get_nowait()
                    values = await loop.run_in_executor(pool, generate, row, condition)
                    saved = store.save_attempt(row["id"], condition, values)
                    # An fsynced append log plus atomic per-slot records retain
                    # each answer even if the process ends before stage summary.
                    with (directory / "outputs.jsonl").open("a") as stream:
                        stream.write(json.dumps(saved, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush(); os.fsync(stream.fileno())
                    newly_generated.append(saved)
                    print(json.dumps({"id": row["id"], "condition": condition, "status": saved["status"],
                        "tokens": saved.get("total_generated_tokens"), "seconds": saved["latency_seconds"],
                        "extraction": saved.get("extraction_status"), "attempt": saved["attempt"]}), flush=True)
                    queue.task_done()
            await asyncio.gather(*(worker() for _ in range(settings["concurrency"])))
        inference_wall = time.perf_counter() - inference_started
        records = store.records()
        count = sum(r["status"] == "ok" for r in records)
        emitted = sum(r.get("total_generated_tokens", 0) for r in newly_generated)
        summary = {"status": "complete" if count == len(planned) else "partial", "scope": scope,
            "question_count": len(rows), "planned_records": len(planned), "successful_model_outputs": count,
            "extracted_answers": sum(r.get("extracted_answer") is not None for r in records),
            "test_scored": False, "new_attempts_this_invocation": len(newly_generated),
            "emitted_tokens_this_invocation": emitted, "measured_generation_wall_seconds": inference_wall,
            "aggregate_tokens_per_second_this_invocation": emitted / inference_wall if inference_wall else None,
            "wall_seconds_including_engine_setup": time.perf_counter() - started,
            "concurrency": settings["concurrency"], "latency_interpretation": settings["latency_interpretation"]}
        atomic_json(directory / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        if count != len(planned):
            raise RuntimeError("Partial coverage retained; inspect infrastructure failures before freezing")
    finally:
        if bridge is not None:
            bridge.close()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if engine is not None:
            engine.shutdown()
        export_outputs(directory, store.records())
        lock.close()


def freeze(root, parent, runtime):
    from s1small.grading import grader_manifest
    from s1small.protocol_audit import audit_development, require_readout
    directory = directory_for(root, "development")
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["question_ids"] != parent["development_question_ids"] or manifest["protocol"] != parent["protocol"]
            or manifest["generation_runtime"] != runtime
            or manifest["export_manifest_sha256"] != file_sha256(root / EXPORT / "export_manifest.json")):
        raise ValueError("Development is not the same frozen protocol/runtime/export")
    planned = [(key, condition) for key in parent["development_question_ids"] for condition in CONDITIONS]
    records = PredictionStore(directory, planned, digest(manifest)).records()
    if len(records) != 45 or any(r["status"] != "ok" or r.get("gold_answer") is not None for r in records):
        raise ValueError("All 45 ungraded development outputs are required")
    audit = audit_development(records); require_readout(audit)
    reference = {(r["question_id"], r["condition"]): r for r in json.loads((root / "colab_macos_development_reference.json").read_text())}
    parity = sum([e["token_ids"] for e in r["segments"] if e["origin"] == "model"] == reference[(r["question_id"], r["condition"])]["model_token_segments"] for r in records)
    exported = verify_export(root, root / EXPORT)
    if grader_manifest() != parent["grader"]:
        raise ValueError("Grader changed; do not open benchmark answers")
    files = {**parent["files_sha256"], **{name: file_sha256(root / name) for name in NEW_FILES},
             "vllm_launcher_manifest.json": file_sha256(root / "vllm_launcher_manifest.json"),
             "manifests/evaluation_freeze_macos.json": file_sha256(root / "manifests/evaluation_freeze_macos.json"),
             EXPORT + "/export_manifest.json": file_sha256(root / EXPORT / "export_manifest.json"),
             **{EXPORT + "/" + name: value for name, value in exported["files_sha256"].items()}}
    frozen = {**parent, "generation_runtime": runtime, "development_run_id": DEV_RUN,
        "final_adapter_directory": EXPORT + "/adapter", "selected_development_adapter_directory": EXPORT + "/adapter",
        "development_manifest_sha256": file_sha256(directory / "manifest.json"), "development_outputs_sha256": digest(records),
        "development_protocol_diagnostics": json.loads((directory / "summary.json").read_text()),
        "vllm_readout_audit": audit, "files_sha256": files,
        "parent_macos_freeze_sha256": file_sha256(root / "manifests/evaluation_freeze_macos.json"),
        "backend_change": "Independent RunPod vLLM replication: frozen Q4 base reconstructed to dense FP16, exact FP32 PEFT adapter transposes evaluated in FP16, concurrent requests, native chunked prefill and prefix caching. Logical prompts, budgets, stops and Wait state machine retained. No retraining or benchmark-driven selection.",
        "macos_development_parity": {"compared_outputs": 45, "exact_model_token_segment_matches": parity,
            "note": "FP16 kernels and batching may change tokens. Separate accuracy and throughput results; do not merge predictions/timings."}}
    save_json(root / "manifests/evaluation_freeze.json", frozen)
    print(json.dumps({"vllm_frozen": True, "development_outputs": 45, "exact_token_matches_with_mac": parity,
                      "readout": audit["status"], "run_id": TEST_RUN}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "convert", "smoke", "development", "freeze", "test", "grade", "analyze", "status"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (root / "VLLM_ISOLATED_WORKSPACE").is_file():
        raise ValueError("Use the separate RunPod vLLM export, never the active Mac or MLX workspace")
    parent = verify_bundle(root)
    if args.mode == "verify":
        verify_prompts(root)
        print("Frozen bundle and vLLM launcher hashes verified; GPU not yet validated", flush=True)
        return
    if args.mode == "convert":
        record = export(root, root / EXPORT)
        print(json.dumps({"conversion": record["status"], "dense_weight_bytes": record["dense_weight_bytes"],
                          "adapter": record["adapter"], "optimizer_updates": 0}), flush=True)
        return
    if args.mode == "status":
        for scope in ("development", "test"):
            counts = {}
            for path in (directory_for(root, scope) / "slots").glob("*.json"):
                row = json.loads(path.read_text()); counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(json.dumps({"scope": scope, "statuses": counts}), flush=True)
        return
    verify_export(root, root / EXPORT)
    if args.mode in ("smoke", "development", "freeze", "test"):
        runtime = activate_identity()
    if args.mode in ("development", "freeze", "test"):
        require_smoke(root, runtime)
    if args.mode == "smoke":
        asyncio.run(smoke(root, parent, runtime))
    elif args.mode in ("development", "test"):
        asyncio.run(evaluate(root, parent, runtime, args.mode))
    elif args.mode == "freeze":
        freeze(root, parent, runtime)
    else:
        script = "grade_results.py" if args.mode == "grade" else "analyze_results.py"
        sys.argv = [str(root / "scripts" / script), "--run-id", TEST_RUN]
        runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Interrupted; saved prediction slots remain resumable")
    signal.signal(signal.SIGTERM, interrupted)
    main()
