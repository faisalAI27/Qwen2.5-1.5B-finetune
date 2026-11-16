"""Run an isolated CUDA replication using unchanged frozen inference sources."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import sys
import time
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import digest, file_sha256, save_json
from s1small.experiment import PredictionStore, verify_freeze

DEV_RUN = "colab-cuda-dev-v1"
TEST_RUN = "math500-colab-cuda-v1"


def verify_bundle(root):
    manifest = json.loads((root / "colab_bundle_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Transferred file differs: {name}")
    parent = json.loads((root / "manifests/evaluation_freeze_macos.json").read_text())
    verify_freeze(root, parent)
    return parent


def run_script(root, script, arguments):
    sys.argv = [str(root / "scripts" / script), *arguments]
    runpy.run_path(sys.argv[0], run_name="__main__")


def freeze_cuda(root, parent):
    from s1small.colab_cuda import runtime_signature
    from s1small.protocol_audit import audit_development, require_readout
    directory = root / "results/protocol_dev" / DEV_RUN
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = parent["development_question_ids"]
    if (manifest["scope"] != "development" or manifest["question_ids"] != expected
            or manifest["protocol"] != json.loads((root / parent["protocol_config_path"]).read_text())
            or manifest["generation_runtime"] != runtime_signature()
            or manifest["adapter_directory"] != parent["selected_development_adapter_directory"]
            or manifest["adapter_weight_sha256"] != file_sha256(root / manifest["adapter_directory"] / "adapters.safetensors")):
        raise ValueError("Complete the unchanged protocol on the same held-out IDs first")
    planned = [(key, condition) for key in expected for condition in parent["planned_conditions"]]
    records = PredictionStore(directory, planned, digest(manifest)).records()
    if len(records) != 45 or any(r["status"] != "ok" or r.get("gold_answer") is not None for r in records):
        raise ValueError("All 45 CUDA development outputs are required without benchmark answers")
    audit = audit_development(records)
    require_readout(audit)
    references = json.loads((root / "colab_macos_development_reference.json").read_text())
    reference = {(r["question_id"], r["condition"]): r for r in references}
    matches = sum([e["token_ids"] for e in r["segments"] if e["origin"] == "model"] ==
                  reference[(r["question_id"], r["condition"])]["model_token_segments"] for r in records)
    parity = {"compared_outputs": len(records), "exact_model_token_segment_matches": matches,
              "macos_mean_generation_seconds": statistics.mean(r["latency_seconds"] for r in references),
              "cuda_mean_generation_seconds": statistics.mean(r["latency_seconds"] for r in records),
              "note": "CUDA and Metal may differ numerically. This replication is scored independently; do not mix Mac and CUDA predictions or latency."}
    freeze = {**parent, "generation_runtime": runtime_signature(),
              "parent_macos_freeze_sha256": file_sha256(root / "manifests/evaluation_freeze_macos.json"),
              "backend_change": "Explicit Linux/CUDA replication; identical packed model, adapter, precision and protocol; no training or test-based selection.",
              "development_run_id": DEV_RUN, "development_manifest_sha256": file_sha256(directory / "manifest.json"),
              "development_outputs_sha256": digest(records), "development_protocol_diagnostics": json.loads((directory / "summary.json").read_text()),
              "cuda_readout_audit": audit, "macos_development_parity": parity,
              "files_sha256": {**parent["files_sha256"], **{name: file_sha256(root / name) for name in
                  ("src/s1small/colab_cuda.py", "scripts/colab_inference.py", "requirements-colab.txt",
                   "manifests/evaluation_freeze_macos.json", "colab_macos_development_reference.json")}}}
    save_json(root / "manifests/evaluation_freeze.json", freeze)
    print(json.dumps({"cuda_frozen": True, "parity": parity, "readout": audit["status"]}, indent=2))


def smoke(root, parent):
    import mlx.core as mx
    from s1small.baselines import load_development
    from s1small.colab_cuda import runtime_signature
    from s1small.mlx_backend import adapter_fingerprint
    from s1small.mlx_decode import load_decoder, s1_prompt, token_protocol
    from s1small.budget_forcing import Budget, run_structured
    from s1small.readout import extract_readout
    training = json.loads((root / "manifests/training_freeze.json").read_text())
    selected = json.loads((root / "results/training" / training["selected_run_id"] / "summary.json").read_text())
    decoder, tokenizer, _ = load_decoder(root, training["training_recipe"], root / selected["adapter_directory"])
    initial = adapter_fingerprint(decoder.model)
    if initial != selected["adapter_tensor_sha256"]:
        raise ValueError("CUDA adapter tensors differ from the verified trained checkpoint")
    row = load_development(root, 1)[0]
    prompt = s1_prompt(row["question"], tokenizer)
    prefix = parent["protocol"].get("answer_readout_prefix", "")
    mx.reset_peak_memory()
    result = run_structured(decoder, tokenizer.encode(prompt, add_special_tokens=False),
                            Budget("D256", 256, 256, 1024), token_protocol(tokenizer, decoder.eos_ids, prefix), 120)
    if not result["answer_text"].strip():
        raise ValueError("CUDA smoke produced empty readout; do not start the benchmark")
    result.update(extract_readout(result["answer_text"], prefix))
    record = {"status": "complete", "scope": "cuda_development_smoke_not_benchmark", "runtime": runtime_signature(),
              "adapter_tensor_sha256": initial, "model_parameters_unchanged": adapter_fingerprint(decoder.model) == initial,
              "peak_mlx_active_bytes": mx.get_peak_memory(), "result": result, "optimizer_updates": 0}
    if not record["model_parameters_unchanged"]:
        raise ValueError("Smoke changed adapter tensors")
    output = root / "results/colab_smoke" / f"run-{time.time_ns()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, record)
    print(json.dumps(record, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "smoke", "development", "freeze", "test", "grade", "analyze", "status"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (root / "COLAB_ISOLATED_WORKSPACE").exists():
        raise ValueError("Run only in the extracted Colab bundle, never the active Mac workspace")
    parent = verify_bundle(root)
    if args.mode == "verify":
        print("Transferred source/model/adapter hashes verified; CUDA has not been tested yet")
        return
    if args.mode == "status":
        for relative in (f"results/protocol_dev/{DEV_RUN}", f"results/test/{TEST_RUN}"):
            counts = {}
            for path in (root / relative / "slots").glob("*.json"):
                row = json.loads(path.read_text())
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(json.dumps({"run": relative, "statuses": counts}))
        return
    if args.mode in ("smoke", "development", "freeze", "test"):
        from s1small.colab_cuda import activate
        activate()
    if args.mode in ("development", "freeze", "test"):
        from s1small.colab_cuda import runtime_signature
        smokes = [json.loads(p.read_text()) for p in (root / "results/colab_smoke").glob("run-*.json")]
        if not any(r["status"] == "complete" and r["model_parameters_unchanged"] and r["runtime"] == runtime_signature() for r in smokes):
            raise ValueError("A successful smoke on this exact CUDA runtime is required first")
    if args.mode == "smoke":
        smoke(root, parent)
    elif args.mode == "development":
        run_script(root, "evaluate.py", ["--scope", "development", "--run-id", DEV_RUN,
                    "--protocol-config", "budget_protocol_v2.json"])
    elif args.mode == "freeze":
        freeze_cuda(root, parent)
    elif args.mode == "test":
        run_script(root, "evaluate.py", ["--scope", "test", "--run-id", TEST_RUN])
    elif args.mode == "grade":
        run_script(root, "grade_results.py", ["--run-id", TEST_RUN])
    elif args.mode == "analyze":
        run_script(root, "analyze_results.py", ["--run-id", TEST_RUN])


if __name__ == "__main__":
    main()
