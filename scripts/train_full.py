"""Resumable local development/final LoRA training; never opens MATH-500."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.baselines import load_development
from s1small.corpus import TrainingCorpus
from s1small.data import canonical_json, digest, file_sha256, save_json
from s1small.generate import build_prompt
from s1small.mlx_backend import adapter_fingerprint, causal_loss, enable_checkpointing, gradient_statistics
from s1small.mlx_runtime import load_local_model, restore_checkpoint, save_checkpoint
from s1small.training import accumulation_windows, supervised_count
from s1small.training_plan import dropout_seed, epoch_order


def repository_commit(root):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def generation_check(model, tokenizer, root):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    row = load_development(root, 1)[0]
    templates = json.loads((root / "configs/prompts/baseline_v1.json").read_text())
    prompt = build_prompt(row["question"], "A", templates, tokenizer)
    model.eval()
    tokens = []
    for token, _ in generate_step(mx.array(tokenizer.encode(prompt, add_special_tokens=False)), model,
                                  max_tokens=32, sampler=lambda logits: mx.argmax(logits, axis=-1)):
        tokens.append(int(token))
        if token == tokenizer.eos_token_id:
            break
    return {"question_id": row["id"], "prompt_sha256": digest(prompt), "prompt": prompt,
            "generated_token_ids": tokens, "raw_output": tokenizer.decode(tokens, skip_special_tokens=False),
            "accuracy": None, "max_new_tokens": 32, "decoder": "greedy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "development", "final", "verify"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        parser.error("Use a simple run directory name")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    import yaml
    root = Path(__file__).resolve().parents[1]
    output = root / "results/training" / args.run_id
    artifact = root / "artifacts/training" / args.run_id
    if args.mode == "verify":
        manifest = json.loads((output / "manifest.json").read_text())
        recipe = manifest["recipe"]
        model, tokenizer, _, _ = load_local_model(root, recipe, artifact / "adapter")
        trained = json.loads((output / "training_summary.json").read_text())
        before = json.loads((output / "generation_before_reload.json").read_text())
        after = generation_check(model, tokenizer, root)
        fingerprint = adapter_fingerprint(model)
        matches = (fingerprint == trained["adapter_tensor_sha256"]
                   and before["prompt_sha256"] == after["prompt_sha256"]
                   and before["generated_token_ids"] == after["generated_token_ids"])
        save_json(output / "generation_after_reload.json", after)
        summary = {**trained, "status": "complete" if matches else "reload_mismatch", "run_id": args.run_id,
                   "fresh_process_reload_identical": matches, "reloaded_adapter_tensor_sha256": fingerprint}
        save_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        if not matches:
            raise SystemExit(1)
        return
    acceptance = json.loads((root / "results/training_pilot/mlx-stage5-qlora-v1-acceptance.json").read_text())
    if acceptance["status"] != "complete":
        raise ValueError("Actual Stage 5 acceptance evidence is required")
    if args.mode == "final":
        freeze = json.loads((root / "manifests/training_freeze.json").read_text())
        recipe = {**freeze["training_recipe"], "scope": "final_training"}
        if file_sha256(root / "manifests/s1k_eligibility.json") != freeze["eligibility_sha256"]:
            raise ValueError("Eligibility changed after training freeze")
        if args.learning_rate is not None:
            parser.error("Final learning rate must come from the actual development freeze")
    else:
        recipe = yaml.safe_load((root / "configs/train_development.yaml").read_text())
        if args.learning_rate not in recipe.pop("learning_rates"):
            parser.error("Choose a predeclared development learning rate")
        recipe["learning_rate"] = args.learning_rate
        recipe["scope"] = "technical_smoke" if args.mode == "smoke" else "development_training"
        if args.mode == "smoke":
            recipe["gradient_accumulation"] = 2
    if (recipe["micro_batch_size"] != 1 or recipe["epochs"] != 1
            or recipe["learning_rate_schedule"] != "constant" or recipe["optimizer"] != "AdamW"):
        raise ValueError("Unsupported explicit training recipe")
    code = {str(p.relative_to(root)): file_sha256(p) for p in [Path(__file__).resolve(), *[root / "src/s1small" / name
             for name in ("corpus.py", "data.py", "formatting.py", "mlx_backend.py", "mlx_runtime.py", "training.py", "training_plan.py")]]}
    identity = {"recipe": recipe, "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json"),
                "format_config_sha256": file_sha256(root / "configs/formatting.json"), "code_sha256": code,
                "conversion_provenance_sha256": file_sha256(root / "artifacts/base-mlx4" /
                    (json.loads((root / "configs/sources.json").read_text())["model"]["revision"] + "-g64") / "provenance.json")}
    recipe_sha = digest(identity)
    if output.exists():
        if not args.resume:
            raise ValueError("Existing run is retained; use --resume after inspecting it")
        old = json.loads((output / "manifest.json").read_text())
        if old["identity"] != identity:
            raise ValueError("Code/data/config changed; cannot silently resume")
    else:
        if args.resume:
            raise ValueError("Resume requires an existing run manifest")
        output.mkdir(parents=True)
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f"attempt-{len(list(attempts.iterdir())):04d}"
    attempt.mkdir()
    cursor = {"completed_steps": 0, "epoch": 0, "next_offset": 0, "examples_committed": 0,
              "input_tokens_committed": 0, "supervised_tokens_committed": 0, "training_seconds_committed": 0.0}
    started_attempt = time.perf_counter()
    try:
        model, tokenizer, adapter_config, conversion = load_local_model(root, recipe)
        corpus = TrainingCorpus(root, tokenizer, max_tokens=recipe["max_training_tokens"])
        partition = "final_training" if args.mode == "final" else "train"
        ids = corpus.ids(partition)
        dev_ids = corpus.ids("development") if args.mode != "final" else []
        if args.mode == "smoke":
            ids = ids[:4]
            dev_ids = dev_ids[:2]
        lengths = {row["id"]: row for row in corpus.validate_lengths(partition)}
        initial_fingerprint = adapter_fingerprint(model)
        if args.mode == "final" and initial_fingerprint != freeze["initial_adapter_tensor_sha256"]:
            raise ValueError("Final model was not freshly initialized with the frozen base/adapter seed")
        manifest = {"scope": recipe["scope"], "recipe": recipe, "identity": identity, "recipe_sha256": recipe_sha,
                    "git_commit": old.get("git_commit") if args.resume else repository_commit(root),
                    "initial_adapter_tensor_sha256": initial_fingerprint, "ordered_partition_ids": ids,
                    "development_ids": dev_ids, "conversion_provenance": conversion,
                    "packages": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "mlx-metal", "numpy", "transformers", "pyarrow")}}
        save_json(output / "manifest.json", manifest)
        optimizer = optim.AdamW(learning_rate=recipe["learning_rate"], betas=recipe["adam_betas"],
                               eps=recipe["adam_eps"], weight_decay=recipe["weight_decay"],
                               bias_correction=recipe["adam_bias_correction"])
        checkpoint_directory = artifact / "checkpoints"
        if args.resume and (checkpoint_directory / "latest.json").exists():
            restored, _ = restore_checkpoint(checkpoint_directory, model, optimizer, recipe_sha)
            cursor.update({key: restored[key] for key in cursor})
            if "last_step" in restored:
                save_json(attempt / "recovered_last_committed_step.json", restored["last_step"])
        save_json(attempt / "start.json", {"resume_cursor": cursor.copy(), "date_utc": datetime.now(timezone.utc).isoformat()})
        if args.resume and (output / "training_summary.json").exists():
            trained = json.loads((output / "training_summary.json").read_text())
            if (trained["adapter_tensor_sha256"] != adapter_fingerprint(model)
                    or cursor["examples_committed"] != len(ids) * recipe["epochs"]):
                raise ValueError("Pending export does not match the durable training checkpoint")
            if not (output / "generation_before_reload.json").exists():
                save_json(output / "generation_before_reload.json", generation_check(model, tokenizer, root))
            print("Retained completed training; generation export is ready for fresh-process verification", flush=True)
            return
        enable_checkpointing(model)
        model.train()
        loss_grad = nn.value_and_grad(model, lambda m, x, y: causal_loss(m, x, y, recipe["loss_chunk_tokens"]))
        mx.reset_peak_memory()
        with (attempt / "steps.jsonl").open("x") as log, (attempt / "microbatches.jsonl").open("x") as micro_log:
            for epoch in range(cursor["epoch"], recipe["epochs"]):
                ordered = epoch_order(ids, recipe["seed"], epoch)
                offset = cursor["next_offset"] if epoch == cursor["epoch"] else 0
                for window_ids in accumulation_windows(ordered[offset:], recipe["gradient_accumulation"]):
                    step_started = time.perf_counter()
                    examples = [corpus.example(identifier) for identifier in window_ids]
                    denominator = sum(supervised_count(row["encoded"]) for row in examples)
                    mean = 0.0
                    accumulated = None
                    for row in examples:
                        mx.random.seed(dropout_seed(recipe["seed"], epoch, row["id"]))
                        loss, gradients = loss_grad(model, mx.array([row["encoded"]["input_ids"]]), mx.array([row["encoded"]["labels"]]))
                        mx.eval(loss, gradients)
                        value = loss.item()
                        if not math.isfinite(value):
                            raise ValueError(f"Nonfinite loss at {row['id']}; no example is dropped")
                        weight = supervised_count(row["encoded"]) / denominator
                        mean += value * weight
                        accumulated = (tree_map(lambda g: g * weight, gradients) if accumulated is None
                                       else tree_map(lambda a, g: a + g * weight, accumulated, gradients))
                        mx.eval(accumulated)
                        del loss, gradients
                        mx.clear_cache()
                        entry = {"epoch": epoch, "id": row["id"], "pending_step": cursor["completed_steps"] + 1,
                                 "loss": value, "input_tokens": len(row["encoded"]["input_ids"])}
                        micro_log.write(canonical_json(entry) + "\n"); micro_log.flush()
                        print(json.dumps(entry), flush=True)
                    stats = gradient_statistics(accumulated)
                    if stats["gradient_norm_before_clip"] > recipe["maximum_acceptable_preclip_norm"]:
                        raise ValueError("Implausible gradient magnitude; optimizer update aborted")
                    scale = min(1.0, recipe["gradient_clip_norm"] / (stats["gradient_norm_before_clip"] + 1e-6))
                    optimizer.update(model, tree_map(lambda g: g * scale, accumulated))
                    mx.eval(model.trainable_parameters(), optimizer.state)
                    adapter_fingerprint(model)
                    elapsed = time.perf_counter() - step_started
                    offset += len(examples)
                    updated = {"completed_steps": cursor["completed_steps"] + 1,
                               "epoch": epoch + 1 if offset == len(ordered) else epoch,
                               "next_offset": 0 if offset == len(ordered) else offset,
                               "examples_committed": cursor["examples_committed"] + len(examples),
                               "input_tokens_committed": cursor["input_tokens_committed"] + sum(lengths[key]["tokens"] for key in window_ids),
                               "supervised_tokens_committed": cursor["supervised_tokens_committed"] + denominator,
                               "training_seconds_committed": cursor["training_seconds_committed"] + elapsed}
                    step_entry = {"step": updated["completed_steps"], "ids": window_ids, "loss": mean,
                                  "seconds": elapsed, "supervised_tokens": denominator, "clip_scale": scale, **stats}
                    if shutil.disk_usage(root).free < recipe["minimum_free_disk_bytes"]:
                        raise OSError("Free disk below the declared checkpoint allowance; previous checkpoint remains resumable")
                    saved = save_checkpoint(checkpoint_directory, model, optimizer,
                                            {**updated, "recipe_sha256": recipe_sha, "last_step": step_entry},
                                            adapter_config, keep_last=recipe["keep_last_checkpoints"])
                    cursor = updated
                    log.write(canonical_json({**step_entry, "checkpoint": str(saved.relative_to(root))}) + "\n"); log.flush()
                    print(json.dumps({"phase": "checkpoint_committed", **cursor, "loss": mean}), flush=True)
        if cursor["examples_committed"] != len(ids) * recipe["epochs"]:
            raise ValueError("Training denominator differs from the declared complete partition")
        # Evaluate every held-out target with dropout disabled; no adapter update.
        model.eval()
        total_loss, total_targets = 0.0, 0
        with (attempt / "development.jsonl").open("x") as stream:
            for identifier in dev_ids:
                row = corpus.example(identifier)
                count = supervised_count(row["encoded"])
                value = causal_loss(model, mx.array([row["encoded"]["input_ids"]]), mx.array([row["encoded"]["labels"]]), recipe["loss_chunk_tokens"]).item()
                if not math.isfinite(value):
                    raise ValueError("Nonfinite development loss; do not select an incomplete trial")
                total_loss += value * count
                total_targets += count
                stream.write(canonical_json({"id": identifier, "loss": value, "supervised_tokens": count}) + "\n"); stream.flush()
                mx.clear_cache()
                print(json.dumps({"phase": "development", "id": identifier, "loss": value}), flush=True)
        latest = json.loads((checkpoint_directory / "latest.json").read_text())
        source = checkpoint_directory / latest["directory"]
        final_adapter = artifact / "adapter"
        if not final_adapter.exists():
            staging_adapter = artifact / "adapter-incomplete"
            if staging_adapter.exists() and args.resume:
                shutil.rmtree(staging_adapter)
            staging_adapter.mkdir()
            for name in ("adapters.safetensors", "adapter_config.json"):
                shutil.copyfile(source / name, staging_adapter / name)
            save_json(staging_adapter / "source_provenance.json", {"recipe": recipe, "identity": identity,
                      "initial_adapter_tensor_sha256": initial_fingerprint, "conversion_provenance": conversion})
            staging_adapter.rename(final_adapter)
        fingerprint = adapter_fingerprint(model)
        summary = {"status": "trained_reload_pending", "scope": recipe["scope"], "learning_rate": recipe["learning_rate"],
                   "training_examples": len(ids), "development_examples": len(dev_ids), "development_ids": dev_ids,
                   "development_loss": total_loss / total_targets if total_targets else None,
                   "development_supervised_tokens": total_targets, "training_cursor": cursor,
                   "peak_mlx_active_bytes": mx.get_peak_memory(), "adapter_tensor_sha256": fingerprint,
                   "initial_adapter_tensor_sha256": initial_fingerprint, "adapter_directory": str(final_adapter.relative_to(root)),
                   "attempt_wall_seconds": time.perf_counter() - started_attempt, "test_scored": False}
        save_json(output / "training_summary.json", summary)
        save_json(output / "generation_before_reload.json", generation_check(model, tokenizer, root))
        save_json(attempt / "finish.json", {"status": "complete", "wall_seconds": time.perf_counter() - started_attempt})
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        save_json(attempt / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                  "last_durable_cursor": cursor, "wall_seconds": time.perf_counter() - started_attempt,
                  "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
