"""Finish first-trial verification and held-out base comparison, then await review.

This controller never starts another training trial, freezes a recipe, or opens
benchmark answers. The original scheduler must remain suspended at this boundary.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import canonical_json, file_sha256
from s1small.experiment import atomic_json


def process(pid):
    result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "state=,command="], capture_output=True, text=True)
    if result.returncode or not result.stdout.strip():
        return None
    state, command = result.stdout.strip().split(None, 1)
    return {"state": state, "command": command}


def validate_completed_trial(summary, manifest, eligibility):
    expected_dev = [row["id"] for row in eligibility["development"]]
    if (summary["status"] != "complete" or not summary["fresh_process_reload_identical"]
            or summary["scope"] != "development_training"
            or summary["training_examples"] != len(eligibility["train"])
            or summary["training_cursor"]["examples_committed"] != len(eligibility["train"])
            or summary["development_ids"] != expected_dev
            or summary["development_examples"] != len(expected_dev)
            or manifest["development_ids"] != expected_dev
            or summary["initial_adapter_tensor_sha256"] != manifest["initial_adapter_tensor_sha256"]
            or not math.isfinite(summary["development_loss"])):
        raise ValueError("First-trial review requires complete training, all held-out targets and verified reload")


def compare_validation(trained, baseline):
    if (baseline["status"] != "complete" or baseline["development_ids"] != trained["development_ids"]
            or baseline["development_supervised_tokens"] != trained["development_supervised_tokens"]
            or not math.isfinite(baseline["development_loss"])):
        raise ValueError("Base/adapted validation must use identical complete IDs and target counts")
    base_loss, tuned_loss = baseline["development_loss"], trained["development_loss"]
    return {"base_development_nll": base_loss, "adapted_development_nll": tuned_loss,
            "nll_difference_adapted_minus_base": tuned_loss - base_loss,
            "relative_nll_reduction": (base_loss - tuned_loss) / base_loss if base_loss > 0 else None,
            "interpretation": "Lower teacher-target NLL indicates better prediction of held-out supplied solutions; it is not mathematical answer accuracy or proof correctness."}


def validate_base(root, run_id):
    import mlx.core as mx
    from s1small.corpus import TrainingCorpus
    from s1small.mlx_backend import adapter_fingerprint, causal_loss
    from s1small.mlx_decode import runtime_signature
    from s1small.mlx_runtime import load_local_model
    from s1small.training import supervised_count
    source = root / "results/training" / run_id
    manifest = json.loads((source / "manifest.json").read_text())
    trained = json.loads((source / "summary.json").read_text())
    eligibility = json.loads((root / "manifests/s1k_eligibility.json").read_text())
    validate_completed_trial(trained, manifest, eligibility)
    for name, expected in manifest["identity"]["code_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("Training implementation changed before matched base validation")
    identity = {"training_manifest_sha256": file_sha256(source / "manifest.json"),
                "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json"),
                "code_sha256": file_sha256(Path(__file__).resolve()), "runtime": runtime_signature()}
    output = root / "results/review" / run_id
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "base_validation.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        if prior["identity"] != identity or file_sha256(root / prior["raw_losses_path"]) != prior["raw_losses_sha256"]:
            raise ValueError("Existing base validation differs; retain it for review")
        compare_validation(trained, prior)
        print("Retaining verified base validation", flush=True)
        return
    recipe = manifest["recipe"]
    model, tokenizer, _, _ = load_local_model(root, recipe)
    model.eval()
    initial = adapter_fingerprint(model)
    if initial != manifest["initial_adapter_tensor_sha256"]:
        raise ValueError("Base validation did not start from the identical zero-output adapter initialization")
    # This is the same original-derived Q4 base and FP32 path. All LoRA B
    # matrices are their original zeros, dropout is off, and no optimizer exists.
    corpus = TrainingCorpus(root, tokenizer, max_tokens=recipe["max_training_tokens"])
    ids = corpus.ids("development")
    attempt = output / f"base-attempt-{time.time_ns()}"
    attempt.mkdir()
    raw = attempt / "losses.jsonl"
    total_loss, targets = 0.0, 0
    started = time.perf_counter()
    with raw.open("x") as stream:
        for identifier in ids:
            row = corpus.example(identifier)
            count = supervised_count(row["encoded"])
            value = causal_loss(model, mx.array([row["encoded"]["input_ids"]]),
                                mx.array([row["encoded"]["labels"]]), recipe["loss_chunk_tokens"]).item()
            if not math.isfinite(value):
                raise ValueError("Nonfinite base development loss")
            record = {"id": identifier, "loss": value, "supervised_tokens": count}
            stream.write(canonical_json(record) + "\n"); stream.flush()
            total_loss += value * count
            targets += count
            print(json.dumps({"phase": "base_development_validation", **record}), flush=True)
            mx.clear_cache()
    if adapter_fingerprint(model) != initial:
        raise ValueError("Read-only base validation modified adapter tensors")
    summary = {"status": "complete", "scope": "held_out_original_base_teacher_target_nll",
               "identity": identity, "development_ids": ids, "development_examples": len(ids),
               "development_supervised_tokens": targets, "development_loss": total_loss / targets,
               "initial_adapter_tensor_sha256": initial, "adapter_unchanged": True,
               "optimizer_updates": 0, "test_scored": False,
               "validation_seconds": time.perf_counter() - started,
               "peak_mlx_active_bytes": mx.get_peak_memory(),
               "raw_losses_path": str(raw.relative_to(root)), "raw_losses_sha256": file_sha256(raw)}
    compare_validation(trained, summary)
    atomic_json(summary_path, summary)


def write_review(root, run_id):
    source = root / "results/training" / run_id
    trained = json.loads((source / "summary.json").read_text())
    baseline = json.loads((root / "results/review" / run_id / "base_validation.json").read_text())
    comparison = compare_validation(trained, baseline)
    result = {"status": "awaiting_user_decision", "run_id": run_id,
              "training_examples": trained["training_examples"], "development_examples": trained["development_examples"],
              "learning_rate": trained["learning_rate"], "fresh_process_reload_identical": True,
              "comparison": comparison, "second_trial_started": False, "test_used_for_selection": False,
              "paper_reference": {"url": "https://arxiv.org/html/2501.19393v1", "model_parameters_billions": 32,
                  "math500_accuracy_without_budget_forcing": 0.926, "math500_accuracy_with_budget_forcing": 0.93,
                  "comparability": "Published 32B benchmark accuracy is contextual only; local 1.5B held-out teacher NLL is a different metric and dataset."},
              "source_hashes": {"trial_summary": file_sha256(source / "summary.json"),
                  "base_validation": file_sha256(root / "results/review" / run_id / "base_validation.json")}}
    atomic_json(root / "results/review" / run_id / "review.json", result)
    atomic_json(root / "docs/first_trial_review.json", result)
    text = ["# First development trial review", "", "Status: **awaiting a decision before further training**.", "",
            f"The complete {trained['training_examples']}-example trial and {trained['development_examples']}-example held-out validation finished, and fresh-process tensor/generation reload matched.", "",
            "| Matched local condition | Held-out teacher-target NLL (lower is better) |", "|---|---:|",
            f"| Original-derived Q4 1.5B base, FP32 activations | {comparison['base_development_nll']:.6f} |",
            f"| First-trial adapted 1.5B model | {comparison['adapted_development_nll']:.6f} |", "",
            comparison["interpretation"], "",
            "The [original s1 paper, Table 1](https://arxiv.org/html/2501.19393v1) reports 92.6% MATH-500 accuracy for its 32B model without budget forcing and 93.0% with it. Those figures are contextual references, not directly comparable to this NLL or a stopping threshold for our learning-rate search. No MATH-500 answers were used for this decision.", "",
            "The second trial, recipe freeze, fresh final training and benchmark remain on hold. A single-trial choice can save compute, but must be reported as a resource constraint; equality with an unrun learning rate cannot be established.", ""]
    (root / "docs/first_trial_review.md").write_text("\n".join(text))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("wait", "baseline"), default="wait")
    parser.add_argument("--run-id", default="local-study-v1-trial-1")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        parser.error("Use a simple run ID")
    root = Path(__file__).resolve().parents[1]
    if args.mode == "baseline":
        validate_base(root, args.run_id)
        return
    runtime = root / "results/runtime"
    lock = (runtime / "first_trial_review.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit("First-trial review controller already active")
    request = json.loads((runtime / "first_trial_review_request.json").read_text())
    if request["run_id"] != args.run_id or request["second_trial_authorized_to_start"]:
        raise ValueError("Review boundary differs from the current request")
    state = {"status": "running", "stage": "waiting_for_first_trial", "run_id": args.run_id,
             "controller_pid": os.getpid(), "worker_pid": request["training_worker_pid"]}
    def update(stage, **values):
        state.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(runtime / "first_trial_review.json", state)
    try:
        update("waiting_for_first_trial")
        while True:
            parent = process(request["suspended_training_orchestrator_pid"])
            if (not (runtime / "PAUSE").exists() or parent is None or "T" not in parent["state"]
                    or "scripts/run_development.py" not in parent["command"]):
                raise ValueError("Review hold changed; refusing to launch overlapping model work")
            worker = process(request["training_worker_pid"])
            if worker is None or "Z" in worker["state"] or "scripts/train_full.py" not in worker["command"]:
                break
            update("waiting_for_first_trial")
            time.sleep(30)
        # The current worker itself completes its 100-row validation and export.
        source = root / "results/training" / args.run_id
        trained = json.loads((source / "training_summary.json").read_text())
        eligibility = json.loads((root / "manifests/s1k_eligibility.json").read_text())
        if trained["training_cursor"]["examples_committed"] != len(eligibility["train"]) or not (source / "generation_before_reload.json").exists():
            raise ValueError("First worker ended before full training/validation/export; evidence retained")
        update("fresh_process_reload_verification")
        subprocess.run([sys.executable, str(root / "scripts/train_full.py"), "--mode", "verify", "--run-id", args.run_id], cwd=root, check=True)
        summary = json.loads((source / "summary.json").read_text())
        manifest = json.loads((source / "manifest.json").read_text())
        validate_completed_trial(summary, manifest, eligibility)
        update("matched_base_development_validation")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mode", "baseline", "--run-id", args.run_id], cwd=root, check=True)
        result = write_review(root, args.run_id)
        update("first_trial_review_ready", status="awaiting_user_decision", review=result)
        print(json.dumps(state, indent=2), flush=True)
    except BaseException as error:
        update(state["stage"], status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
