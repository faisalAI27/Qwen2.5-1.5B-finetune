"""Measured LoRA feasibility only: a complete-example backward probe or 20-row pilot."""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.baselines import load_development
from s1small.data import canonical_json, digest, file_sha256, read_jsonl, save_json
from s1small.environment import collect_environment
from s1small.formatting import collate_training_examples, make_training_example
from s1small.generate import generate_one
from s1small.model import load_base_model, synchronize, verify_checkpoint
from s1small.training import (MemorySampler, accumulation_windows, adapter_fingerprint, attach_lora,
                              chunked_causal_loss, clip_gradients_cpu_norm, gradient_report, supervised_count)


def select_examples(root, tokenizer, config, mode, target):
    import pyarrow.parquet as parquet
    provenance = json.loads((root / "manifests/data_provenance.json").read_text())
    path = root / "data/raw/s1k/data/train-00000-of-00001.parquet"
    expected = next(row["sha256"] for row in provenance["files"] if row["path"] == str(path.relative_to(root)))
    if file_sha256(path) != expected:
        raise ValueError("Dataset differs from Stage 2 provenance")
    eligibility = json.loads((root / "manifests/s1k_eligibility.json").read_text())
    if (file_sha256(root / "manifests/s1k_split.json") != eligibility["original_split_sha256"]
            or file_sha256(root / "configs/data_exclusions.json") != eligibility["exclusion_config_sha256"]):
        raise ValueError("Eligibility inputs changed")
    lengths = {row["id"]: row["total"] for row in read_jsonl(root / "results/data_audit/lengths/per_example.jsonl")}
    candidates = eligibility["train"]
    if mode == "profile" and target is not None:
        chosen = [min(candidates, key=lambda row: (abs(lengths[row["id"]] - target), row["id"]))]
    else:
        chosen = sorted(candidates, key=lambda row: (lengths[row["id"]], row["id"]))[:1 if mode == "profile" else config["pilot_examples"]]
    rows = parquet.read_table(path).to_pylist()
    output = []
    for reference in chosen:
        row = rows[reference["source_row"]]
        if digest(row) != reference["row_sha256"]:
            raise ValueError("Training row differs from immutable manifest")
        encoded = make_training_example(row, tokenizer, max_length=config["max_training_tokens"])
        if len(encoded["input_ids"]) != lengths[reference["id"]]:
            raise ValueError("Training example length differs from the Stage 2 audit")
        output.append({"id": reference["id"], "encoded": encoded})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["profile", "pilot"], default="profile")
    parser.add_argument("--profile-target-tokens", type=int, help="Profile the closest complete eligible training example; never truncate to this value")
    parser.add_argument("--baseline-prefix", default="stage3-local-v1")
    parser.add_argument("--run-id")
    parser.add_argument("--loss-implementation", choices=["standard", "chunked"])
    parser.add_argument("--loss-chunk-tokens", type=int)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], help="Explicit profiling override, recorded in the effective config")
    parser.add_argument("--attention-implementation", choices=["sdpa", "eager", "s1_chunked"], help="Explicit diagnostic override, recorded in the effective config")
    args = parser.parse_args()
    if args.loss_chunk_tokens is not None and args.loss_chunk_tokens < 1:
        parser.error("Loss chunk size must be positive")
    if args.profile_target_tokens is not None and (args.mode != "profile" or args.profile_target_tokens < 1):
        parser.error("A positive target is allowed in profile mode only")
    import torch
    import yaml
    from peft import PeftModel
    root = Path(__file__).resolve().parents[1]
    baseline_audit = root / "results/dev_baselines" / (args.baseline_prefix + "-main") / "audit_summary.json"
    baseline = json.loads(baseline_audit.read_text())
    if baseline["status"] != "complete":
        raise ValueError("Complete the actual baseline smoke workflow before profiling training")
    format_audit = json.loads((root / "docs/training_format_summary.json").read_text())
    if file_sha256(root / "configs/formatting.json") != format_audit["format_config_sha256"]:
        raise ValueError("Training format changed after the mask audit")
    config = yaml.safe_load((root / "configs/train_pilot.yaml").read_text())
    if args.dtype:
        config["dtype"] = args.dtype
    if args.attention_implementation:
        config["attention_implementation"] = args.attention_implementation
    config["loss_implementation"] = args.loss_implementation or config.get("loss_implementation", "standard")
    config["loss_chunk_tokens"] = args.loss_chunk_tokens or config.get("loss_chunk_tokens", 128)
    if config["scope"] != "technical_feasibility_pilot_only" or config["micro_batch_size"] != 1 or config["epochs"] != 1:
        raise ValueError("This command implements a single-epoch technical pilot with micro-batch 1")
    if not config["gradient_checkpointing"]:
        raise ValueError("This memory-constrained pilot requires its declared gradient-checkpointing setting")
    sources = json.loads((root / "configs/sources.json").read_text())
    checkpoint = root / "artifacts/base" / sources["model"]["revision"]
    provenance = verify_checkpoint(checkpoint, sources)
    weight_hash = next(row["sha256"] for row in provenance["files"] if row["name"] == "model.safetensors")
    if weight_hash != baseline["weights_sha256"]:
        raise ValueError("Training base differs from the baseline checkpoint")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("pilot-%Y%m%dT%H%M%S%fZ")
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        parser.error("Use letters, digits, underscores and hyphens for --run-id")
    output = root / "results/training_pilot" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", {"scope": "technical_feasibility_only", "mode": args.mode,
              "config": config, "config_sha256": digest(config), "sources": sources,
              "loss_implementation": config["loss_implementation"], "loss_chunk_tokens": config["loss_chunk_tokens"],
              "weights_sha256": weight_hash, "format_audit_sha256": file_sha256(root / "docs/training_format_summary.json"),
              "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json"),
              "environment": collect_environment(root),
              "code_sha256": {str(path.relative_to(root)): file_sha256(path) for path in [Path(__file__).resolve(), *sorted((root / "src/s1small").glob("*.py"))]}})
    sampler = MemorySampler(config["device"])
    updates = 0
    try:
        model, tokenizer, load_info = load_base_model(checkpoint, config)
        model, adapter_info = attach_lora(model, config)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        def compute_loss(batch):
            return (chunked_causal_loss(model, batch, config["loss_chunk_tokens"])
                    if config["loss_implementation"] == "chunked" else model(**batch).loss)
        examples = select_examples(root, tokenizer, config, args.mode, args.profile_target_tokens)
        pad_id = tokenizer.convert_tokens_to_ids("<|fim_pad|>")
        save_json(output / "examples.json", [{"id": row["id"], "tokens": len(row["encoded"]["input_ids"]),
                                              "supervised_tokens": supervised_count(row["encoded"])} for row in examples])
        save_json(output / "model.json", {"load": load_info, "adapter": adapter_info,
                                           "gradient_checkpointing": model.is_gradient_checkpointing})
        started = time.perf_counter()
        with sampler:
            if args.mode == "profile":
                batch = collate_training_examples([examples[0]["encoded"]], pad_id, return_tensors=True)
                batch = {key: value.to(config["device"]) for key, value in batch.items()}
                # Keep only the loss: retaining CausalLMOutput would also keep
                # its large full-vocabulary logits alive during backward.
                loss = compute_loss(batch)
                if not bool(torch.isfinite(loss).item()):
                    raise ValueError("Nonfinite supervised-token loss")
                value = loss.item()
                save_json(output / "forward.json", {"loss": value, "elapsed_seconds": time.perf_counter() - started})
                print(json.dumps({"phase": "forward_complete", "loss": value, "input_tokens": len(examples[0]["encoded"]["input_ids"])}), flush=True)
                loss.backward()
                gradients = gradient_report(model)
                gradients.update(clip_gradients_cpu_norm([p for p in model.parameters() if p.requires_grad], config["gradient_clip_norm"]))
                if gradients["gradient_norm_before_clip"] > config["maximum_acceptable_preclip_norm"]:
                    raise ValueError(f"Numerically implausible preclip gradient norm: {gradients['gradient_norm_before_clip']}")
                synchronize(torch, config["device"])
                elapsed = time.perf_counter() - started
                save_json(output / "summary.json", {"status": "complete", "mode": "single_forward_backward_no_optimizer_update",
                          "example_id": examples[0]["id"], "input_tokens": len(examples[0]["encoded"]["input_ids"]),
                          "supervised_tokens": supervised_count(examples[0]["encoded"]), "loss": value,
                          "elapsed_seconds": elapsed, "optimizer_updates": 0, "gradients": gradients})
                print(json.dumps({"status": "complete", "mode": "profile", "loss": value, "seconds": elapsed}), flush=True)
            else:
                if len(examples) != config["pilot_examples"]:
                    raise ValueError("Insufficient eligible pilot examples")
                random.Random(config["seed"]).shuffle(examples)
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"],
                                              betas=(config["adam_beta1"], config["adam_beta2"]), foreach=False)
                input_tokens = 0
                target_tokens = 0
                with (output / "steps.jsonl").open("x") as log:
                    for window in accumulation_windows(examples, config["gradient_accumulation"]):
                        optimizer.zero_grad(set_to_none=True)
                        denominator = sum(supervised_count(row["encoded"]) for row in window)
                        window_loss = 0.0
                        step_start = time.perf_counter()
                        for row in window:
                            batch = collate_training_examples([row["encoded"]], pad_id, return_tensors=True)
                            batch = {key: value.to(config["device"]) for key, value in batch.items()}
                            loss = compute_loss(batch)
                            if not bool(torch.isfinite(loss).item()):
                                raise ValueError("Nonfinite training loss")
                            count = supervised_count(row["encoded"])
                            micro_loss = loss.item()
                            window_loss += micro_loss * count / denominator
                            (loss * (count / denominator)).backward()
                            input_tokens += len(row["encoded"]["input_ids"])
                            target_tokens += count
                            del loss, batch
                            synchronize(torch, config["device"])
                            print(json.dumps({"phase": "microbatch_backward_complete", "example_id": row["id"],
                                              "loss": micro_loss, "input_tokens": len(row["encoded"]["input_ids"]),
                                              "pending_optimizer_step": updates + 1}), flush=True)
                            if config["device"] == "mps" and config["clear_mps_cache_each_microbatch"]:
                                torch.mps.empty_cache()
                        gradients = gradient_report(model)
                        norm_info = clip_gradients_cpu_norm(parameters, config["gradient_clip_norm"])
                        save_json(output / f"gradient-step-{updates + 1}.json", norm_info)
                        if norm_info["gradient_norm_before_clip"] > config["maximum_acceptable_preclip_norm"]:
                            raise ValueError(f"Numerically implausible preclip gradient norm: {norm_info['gradient_norm_before_clip']}")
                        optimizer.step()
                        synchronize(torch, config["device"])
                        if not all(bool(torch.isfinite(p.detach().cpu()).all().item()) for p in parameters):
                            raise ValueError("Nonfinite adapter parameter after optimizer step")
                        updates += 1
                        entry = {"optimizer_step": updates, "loss": window_loss, "micro_batches": len(window),
                                 "supervised_tokens": denominator, "elapsed_seconds": time.perf_counter() - step_start,
                                 **norm_info, **gradients}
                        log.write(canonical_json(entry) + "\n"); log.flush()
                        print(json.dumps(entry), flush=True)
                elapsed = time.perf_counter() - started
        save_json(output / "memory.json", sampler.report())
        if args.mode == "profile":
            return
        artifact = root / "artifacts/pilots" / run_id
        model.peft_config["default"].base_model_name_or_path = sources["model"]["id"]
        model.peft_config["default"].revision = sources["model"]["revision"]
        model.save_pretrained(artifact, safe_serialization=True)
        fingerprint = adapter_fingerprint(model)
        save_json(artifact / "source_provenance.json", {"sources": sources, "config": config,
                  "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json"), "weights_sha256": weight_hash})
        prompts = json.loads((root / "configs/prompts/baseline_v1.json").read_text())
        generation = yaml.safe_load((root / "configs/baseline.yaml").read_text())
        generation.update(dtype=config["dtype"], max_new_tokens=32)
        question = load_development(root, 1)[0]
        model.eval()
        before = generate_one(model, tokenizer, question["question"], "A", prompts, generation)
        save_json(output / "generation_before_reload.json", before)
        del model, optimizer, parameters
        gc.collect()
        torch.mps.empty_cache() if config["device"] == "mps" else None
        clean_base, clean_tokenizer, _ = load_base_model(checkpoint, config)
        reloaded = PeftModel.from_pretrained(clean_base, artifact, is_trainable=False)
        reloaded.eval()
        reloaded_fingerprint = adapter_fingerprint(reloaded)
        after = generate_one(reloaded, clean_tokenizer, question["question"], "A", prompts, generation)
        save_json(output / "generation_after_reload.json", after)
        identical = before["generated_token_ids"] == after["generated_token_ids"] and fingerprint == reloaded_fingerprint
        summary = {"status": "complete" if identical else "reload_mismatch", "examples": len(examples),
                   "optimizer_updates": updates, "input_tokens_seen": input_tokens, "supervised_tokens_seen": target_tokens,
                   "training_seconds": elapsed, "input_tokens_per_second": input_tokens / elapsed,
                   "supervised_tokens_per_second": target_tokens / elapsed,
                   "adapter_bytes": sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file()),
                   "adapter_directory": str(artifact.relative_to(root)), "reload_tokens_identical": identical,
                   "adapter_tensor_sha256_before_reload": fingerprint,
                   "adapter_tensor_sha256_after_reload": reloaded_fingerprint,
                   "question_id_for_reload_check": question["id"], "accuracy_result": None}
        save_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        if not identical:
            raise SystemExit(1)
    except Exception as error:
        sampler.stop.set()
        save_json(output / "failure.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "optimizer_updates_completed": updates, "memory": sampler.report(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
