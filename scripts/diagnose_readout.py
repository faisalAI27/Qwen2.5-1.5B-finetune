"""Development-only readout probes on the accepted adapter; no training or gold."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.baselines import load_development
from s1small.budget_forcing import Budget, run_structured
from s1small.data import digest, file_sha256, save_json
from s1small.experiment import atomic_json
from s1small.grading import extract_final_answer
from s1small.mlx_decode import load_decoder, runtime_signature, s1_prompt, token_protocol


def variants(tokenizer, protocol):
    # Prefixes are common readout interventions, never condition-specific clues.
    return [(name, prefix, replace(protocol, answer_header=protocol.answer_header +
             tuple(tokenizer.encode(prefix, add_special_tokens=False))))
            for name, prefix in (("training_header", ""), ("answer_prefix", "Answer: "),
                                 ("final_answer_prefix", "Final Answer: "))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pipeline-pid", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    status_path = root / "results/runtime/readout_diagnostic.json"
    state = {"status": "running", "stage": "waiting_for_protocol_checks", "controller_pid": os.getpid(),
             "training_updates": 0, "test_answers_used": False}
    def update(stage, **values):
        state.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(status_path, state)
    try:
        update("waiting_for_protocol_checks")
        if args.wait_for_pipeline_pid:
            while True:
                if (root / "manifests/evaluation_freeze.json").exists():
                    update("not_run_after_test_freeze", status="not_run")
                    return
                process = subprocess.run(["/bin/ps", "-p", str(args.wait_for_pipeline_pid), "-o", "command="], capture_output=True, text=True)
                if process.returncode or "scripts/complete_project.py" not in process.stdout:
                    break
                time.sleep(30)
        if (root / "manifests/evaluation_freeze.json").exists() or (root / "data/raw/math500/test_with_answers.jsonl").exists():
            raise ValueError("Readout development must precede benchmark freeze/access")
        audit = json.loads((root / "results/protocol_dev/protocol-dev-v1/compatibility_audit.json").read_text())
        if audit["status"] != "readout_not_established":
            raise ValueError("This diagnostic requires the completed failed native readout gate")
        # The main pipeline has exited, and its subprocess must also be gone.
        processes = subprocess.check_output(["/bin/ps", "-axo", "state=,command="], text=True)
        if any("Z" not in line.split(None, 1)[0] and any("scripts/" + script in line
               for script in ("evaluate.py", "train_full.py")) for line in processes.splitlines() if line.strip()):
            raise ValueError("Another model job is active; refusing concurrent GPU loading")
        import mlx.core as mx
        from s1small.mlx_backend import adapter_fingerprint
        training = json.loads((root / "manifests/training_freeze.json").read_text())
        summary = json.loads((root / "results/training" / training["selected_run_id"] / "summary.json").read_text())
        decoder, tokenizer, _ = load_decoder(root, training["training_recipe"], root / summary["adapter_directory"])
        initial = adapter_fingerprint(decoder.model)
        if initial != summary["adapter_tensor_sha256"]:
            raise ValueError("Diagnostic adapter differs from the completed verified run")
        protocol = token_protocol(tokenizer, decoder.eos_ids)
        rows = load_development(root, 5)
        output = root / "results/diagnostics/readout-v1"
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"scope": "development_readout_diagnostic_not_benchmark", "question_ids": [r["id"] for r in rows],
                    "adapter_sha256": file_sha256(root / summary["adapter_directory"] / "adapters.safetensors"),
                    "training_freeze_sha256": file_sha256(root / "manifests/training_freeze.json"),
                    "runtime": runtime_signature(), "code_sha256": file_sha256(Path(__file__).resolve()),
                    "reasoning_cap": 512, "answer_cap": 256, "absolute_ceiling": 1024, "timeout_seconds": 120,
                    "interpretation": "Prefix-assisted extraction is explicitly labeled and is not mathematical correctness or evidence of spontaneous learned formatting."}
        save_json(output / "manifest.json", manifest)
        update("bounded_readout_probes", planned=15)
        records = []
        for row in rows:
            prompt = s1_prompt(row["question"], tokenizer)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            for name, prefix, variant in variants(tokenizer, protocol):
                path = output / (digest([row["id"], name])[:24] + ".json")
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["manifest_sha256"] != digest(manifest):
                        raise ValueError("Existing diagnostic belongs to a different recipe")
                else:
                    record = run_structured(decoder, ids, Budget("D512", 512, 256, 1024), variant, 120)
                    record.update(question_id=row["id"], variant=name, prompt=prompt, manifest_sha256=digest(manifest),
                                  injected_readout_prefix=prefix, generated_only_extraction=extract_final_answer(record["answer_text"]),
                                  prefix_assisted_extraction=extract_final_answer(prefix + record["answer_text"]),
                                  accuracy=None, grading_status="not_scored", scope=manifest["scope"])
                    save_json(path, record)
                records.append(record)
                print(json.dumps({"question_id":row["id"], "variant":name, "answer_excerpt":record["answer_text"][:180],
                                  "termination":record["termination_reason"]}), flush=True)
                update("bounded_readout_probes", completed=len(records), planned=15)
                mx.clear_cache()
        if adapter_fingerprint(decoder.model) != initial:
            raise ValueError("Inference diagnostic changed adapter parameters")
        report = {"status": "complete", "model_parameters_unchanged": True, "optimizer_updates": 0,
                  "test_answers_used": False, "outputs": len(records),
                  "interpretation": manifest["interpretation"],
                  "variants": {name: {"outputs": sum(r["variant"]==name for r in records),
                      "nonempty_answers": sum(r["variant"]==name and bool(r["answer_text"].strip()) for r in records),
                      "generated_only_extractions": sum(r["variant"]==name and bool(r["generated_only_extraction"]["extracted_answer"]) for r in records),
                      "prefix_assisted_extractions": sum(r["variant"]==name and bool(r["prefix_assisted_extraction"]["extracted_answer"]) for r in records)}
                      for name, _, _ in variants(tokenizer, protocol)}}
        save_json(output / "summary.json", report)
        save_json(root / "docs/readout_diagnostic_v1.json", report)
        update("readout_diagnostic_complete", status="complete", report=report)
    except BaseException as error:
        update(state["stage"], status="failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
