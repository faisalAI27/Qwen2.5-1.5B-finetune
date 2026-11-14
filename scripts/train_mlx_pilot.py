"""Native MLX complete-example profile/pilot and independent-process reload check."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.baselines import load_development
from s1small.corpus import TrainingCorpus
from s1small.data import canonical_json, digest, file_sha256, read_jsonl, save_json
from s1small.generate import build_prompt
from s1small.mlx_backend import (adapter_fingerprint, attach_adapter, causal_loss, enable_checkpointing,
                                 enable_float32_activations, gradient_statistics)
from s1small.model import verify_checkpoint
from s1small.training import accumulation_windows, supervised_count


def greedy_check(model, tokenizer, root):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    question = load_development(root, 1)[0]
    templates = json.loads((root / "configs/prompts/baseline_v1.json").read_text())
    prompt = build_prompt(question["question"], "A", templates, tokenizer)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    model.eval()
    tokens = []
    started = time.perf_counter()
    for token, _ in generate_step(mx.array(prompt_ids), model, max_tokens=32,
                                  sampler=lambda probabilities: mx.argmax(probabilities, axis=-1)):
        tokens.append(int(token))
        if token == tokenizer.eos_token_id:
            break
    return {"question_id": question["id"], "prompt": prompt, "prompt_sha256": digest(prompt),
            "generated_token_ids": tokens, "raw_output": tokenizer.decode(tokens, skip_special_tokens=False),
            "max_new_tokens": 32, "seconds": time.perf_counter() - started, "accuracy": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["profile", "pilot", "verify"], default="profile")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-target-tokens", type=int)
    parser.add_argument("--float32-activations", action="store_true", help="Recorded compute-precision override; preserve original base storage")
    parser.add_argument("--base-format", choices=["original", "mlx4"], default="original")
    parser.add_argument("--profile-partition", choices=["train", "development"], default="train",
                        help="Development is allowed only for a zero-update memory/numerical profile")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        parser.error("Use a simple run directory name")
    if args.profile_target_tokens is not None and (args.mode != "profile" or args.profile_target_tokens < 1):
        parser.error("A positive target is allowed in profile mode only")
    if args.profile_partition != "train" and args.mode != "profile":
        parser.error("Development examples cannot be used for pilot optimizer updates")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.utils import load_model
    from mlx_lm.tuner.utils import load_adapters
    from transformers import AutoTokenizer
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/train_mlx_pilot.yaml").read_text())
    config["activation_dtype"] = "float32" if args.float32_activations else "bfloat16"
    config["base_format"] = args.base_format
    if args.base_format == "mlx4":
        config["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine"}
    if config["micro_batch_size"] != 1 or config["epochs"] != 1:
        raise ValueError("This command is restricted to the single-example/epoch technical pilot")
    if not mx.is_available(mx.gpu):
        raise RuntimeError("Native Apple GPU access required; no silent CPU fallback")
    mx.set_default_device(mx.gpu)
    device = mx.device_info()
    memory_limit = int(device["max_recommended_working_set_size"] * config["memory_fraction"])
    mx.set_memory_limit(memory_limit)
    mx.set_cache_limit(config["cache_limit_bytes"])
    mx.random.seed(config["seed"])
    np.random.seed(config["seed"])
    sources = json.loads((root / "configs/sources.json").read_text())
    checkpoint = root / "artifacts/base" / sources["model"]["revision"]
    provenance = verify_checkpoint(checkpoint, sources)
    weight_hash = next(row["sha256"] for row in provenance["files"] if row["name"] == "model.safetensors")
    conversion = None
    if args.base_format == "mlx4":
        checkpoint = root / "artifacts/base-mlx4" / (sources["model"]["revision"] + "-g64")
        conversion = json.loads((checkpoint / "provenance.json").read_text())
        if (conversion["source"] != sources["model"] or conversion["tokenizer_source"] != sources["tokenizer"]
                or conversion["source_weight_sha256"] != weight_hash or conversion["quantization"] != config["quantization"]):
            raise ValueError("Converted base does not derive from the declared original weights/settings")
        for row in conversion["files"]:
            if file_sha256(checkpoint / row["name"]) != row["sha256"]:
                raise ValueError(f"Converted checkpoint checksum differs: {row['name']}")
    baseline = json.loads((root / "docs/baseline_smoke_summary.json").read_text())
    if baseline["status"] != "complete" or weight_hash != baseline["weights_sha256"]:
        raise ValueError("Verified baseline checkpoint identity is required")
    output = root / "results/training_pilot" / args.run_id
    artifact = root / "artifacts/pilots" / args.run_id
    if args.mode == "verify":
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["config"] != config or manifest["weights_sha256"] != weight_hash:
            raise ValueError("Pilot config/base changed before reload verification")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "manifest.json", {"mode": args.mode, "scope": "technical_feasibility_only", "config": config,
                  "profile_partition": args.profile_partition if args.mode == "profile" else None,
                  "sources": sources, "weights_sha256": weight_hash,
                  "conversion_provenance": conversion,
                  "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json"),
                  "packages": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "mlx-lm", "transformers", "numpy", "pyarrow")},
                  "memory_limit_bytes": memory_limit, "device": {k: v for k, v in device.items() if k in ("device_name", "architecture", "max_recommended_working_set_size", "memory_size")},
                  "code_sha256": {str(p.relative_to(root)): file_sha256(p) for p in [Path(__file__).resolve(), *sorted((root / "src/s1small").glob("*.py"))]}})
    updates = 0
    try:
        load_started = time.perf_counter()
        model, model_config = load_model(checkpoint, lazy=False, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        if args.base_format == "original":
            if model.model.embed_tokens.weight.dtype != mx.bfloat16 or model_config.get("quantization") or model_config.get("quantization_config"):
                raise ValueError("Expected the unchanged original BF16 base, without quantization")
        elif model_config.get("quantization") != config["quantization"]:
            raise ValueError("Loaded quantization differs from verified conversion metadata")
        if args.float32_activations:
            enable_float32_activations(model)
        if args.mode == "verify":
            model = load_adapters(model, str(artifact))
            fingerprint = adapter_fingerprint(model)
            after = greedy_check(model, tokenizer, root)
            before = json.loads((output / "generation_before_reload.json").read_text())
            trained = json.loads((output / "training_summary.json").read_text())
            identical = (fingerprint == trained["adapter_tensor_sha256"]
                         and after["prompt_sha256"] == before["prompt_sha256"]
                         and after["generated_token_ids"] == before["generated_token_ids"])
            save_json(output / "generation_after_reload.json", after)
            summary = {**trained, "status": "complete" if identical else "reload_mismatch",
                       "reload_tokens_identical": identical, "reload_adapter_tensor_sha256": fingerprint,
                       "verification_process": "fresh_native_process", "accuracy": None}
            save_json(output / "summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
            if not identical:
                raise ValueError("Adapter fingerprint or generation mismatch after fresh-process reload")
            return
        info = attach_adapter(model, config["lora_rank"], config["lora_scale"], config["lora_dropout"])
        if info["trainable_parameters"] != 4358144 or config["lora_scale"] != config["lora_alpha"] / config["lora_rank"]:
            raise ValueError("Unexpected adapter parameter count/scaling")
        mx.eval(model.parameters())
        save_json(output / "model.json", {"load_seconds": time.perf_counter() - load_started,
                  "base_storage_dtype": str(model.model.embed_tokens.weight.dtype), "quantization": config["quantization"], "adapter": info})
        enable_checkpointing(model)
        model.train()
        corpus = TrainingCorpus(root, tokenizer, max_tokens=config["max_training_tokens"])
        lengths = {row["id"]: row["total"] for row in read_jsonl(root / "results/data_audit/lengths/per_example.jsonl")}
        ids = corpus.ids(args.profile_partition if args.mode == "profile" else "train")
        if args.mode == "profile" and args.profile_target_tokens is not None:
            selected = [min(ids, key=lambda identifier: (abs(lengths[identifier] - args.profile_target_tokens), identifier))]
        else:
            selected = sorted(ids, key=lambda identifier: (lengths[identifier], identifier))[:1 if args.mode == "profile" else config["pilot_examples"]]
        examples = [corpus.example(identifier) for identifier in selected]
        if any(len(row["encoded"]["input_ids"]) != lengths[row["id"]] for row in examples):
            raise ValueError("Example lengths differ from the immutable audit")
        save_json(output / "examples.json", [{"id": row["id"], "tokens": len(row["encoded"]["input_ids"]),
                  "supervised_tokens": supervised_count(row["encoded"])} for row in examples])
        loss_and_grad = nn.value_and_grad(model, lambda m, x, y: causal_loss(m, x, y, config["loss_chunk_tokens"]))
        mx.reset_peak_memory()
        started = time.perf_counter()
        if args.mode == "profile":
            row = examples[0]
            loss, gradients = loss_and_grad(model, mx.array([row["encoded"]["input_ids"]]), mx.array([row["encoded"]["labels"]]))
            mx.eval(loss, gradients)
            value = loss.item()
            stats = gradient_statistics(gradients)
            save_json(output / "gradient_diagnostic.json", {"loss": value, **stats})
            if not math.isfinite(value) or stats["gradient_norm_before_clip"] > config["maximum_acceptable_preclip_norm"]:
                raise ValueError("Invalid loss or implausible gradient magnitude")
            summary = {"status": "complete", "mode": "single_forward_backward_no_optimizer_update", "loss": value,
                       "input_tokens": len(row["encoded"]["input_ids"]), "supervised_tokens": supervised_count(row["encoded"]),
                       "seconds": time.perf_counter() - started, "optimizer_updates": 0, **stats,
                       "peak_mlx_active_bytes": mx.get_peak_memory(), "memory_note": "MLX allocator active peak, not dedicated VRAM"}
            save_json(output / "summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
            return
        random.Random(config["seed"]).shuffle(examples)
        optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
        total_input = sum(len(row["encoded"]["input_ids"]) for row in examples)
        with (output / "steps.jsonl").open("x") as stream:
            for window in accumulation_windows(examples, config["gradient_accumulation"]):
                denominator = sum(supervised_count(row["encoded"]) for row in window)
                accumulated = None
                mean = 0.0
                step_started = time.perf_counter()
                for row in window:
                    loss, gradients = loss_and_grad(model, mx.array([row["encoded"]["input_ids"]]), mx.array([row["encoded"]["labels"]]))
                    mx.eval(loss, gradients)
                    value = loss.item()
                    if not math.isfinite(value):
                        raise ValueError("Nonfinite training loss")
                    weight = supervised_count(row["encoded"]) / denominator
                    mean += value * weight
                    accumulated = (tree_map(lambda g: g * weight, gradients) if accumulated is None
                                   else tree_map(lambda a, g: a + g * weight, accumulated, gradients))
                    mx.eval(accumulated)
                    del loss, gradients
                    mx.clear_cache()
                    print(json.dumps({"phase": "microbatch_backward_complete", "id": row["id"], "loss": value,
                                      "pending_optimizer_step": updates + 1}), flush=True)
                stats = gradient_statistics(accumulated)
                save_json(output / f"gradient-step-{updates + 1}.json", stats)
                if stats["gradient_norm_before_clip"] > config["maximum_acceptable_preclip_norm"]:
                    raise ValueError("Implausible preclip gradient norm; no optimizer update attempted")
                scale = min(1.0, config["gradient_clip_norm"] / (stats["gradient_norm_before_clip"] + 1e-6))
                optimizer.update(model, tree_map(lambda g: g * scale, accumulated))
                mx.eval(model.trainable_parameters(), optimizer.state)
                adapter_fingerprint(model)  # also checks all updated parameters are finite
                updates += 1
                entry = {"optimizer_step": updates, "loss": mean, "supervised_tokens": denominator,
                         "seconds": time.perf_counter() - step_started, "clip_scale": scale, **stats}
                stream.write(canonical_json(entry) + "\n"); stream.flush()
                print(json.dumps(entry), flush=True)
        elapsed = time.perf_counter() - started
        artifact.mkdir(parents=True, exist_ok=False)
        mx.save_safetensors(str(artifact / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        save_json(artifact / "adapter_config.json", info)
        save_json(artifact / "source_provenance.json", {"sources": sources, "config": config, "weights_sha256": weight_hash,
                  "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json")})
        summary = {"status": "trained_reload_pending", "examples": len(examples), "optimizer_updates": updates,
                   "training_seconds": elapsed, "input_tokens": total_input, "input_tokens_per_second": total_input / elapsed,
                   "peak_mlx_active_bytes": mx.get_peak_memory(), "adapter_directory": str(artifact.relative_to(root)),
                   "adapter_bytes": sum(p.stat().st_size for p in artifact.iterdir()), "adapter_tensor_sha256": adapter_fingerprint(model)}
        save_json(output / "training_summary.json", summary)
        save_json(output / "generation_before_reload.json", greedy_check(model, tokenizer, root))
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        save_json(output / ("verify_failure.json" if args.mode == "verify" else "failure.json"),
                  {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                   "optimizer_updates_completed": updates, "memory": {"peak_mlx_active_bytes": mx.get_peak_memory()},
                   "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
