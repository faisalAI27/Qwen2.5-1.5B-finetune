"""Verified converted-base loading and atomic resumable LoRA checkpoints."""

import json
from pathlib import Path
import shutil

from s1small.data import file_sha256, save_json
from s1small.mlx_backend import adapter_fingerprint, attach_adapter, enable_float32_activations


def load_local_model(root, config, adapter_path=None):
    import mlx.core as mx
    from mlx_lm.utils import load_model
    from mlx_lm.tuner.utils import load_adapters
    from transformers import AutoTokenizer
    root = Path(root)
    sources = json.loads((root / "configs/sources.json").read_text())
    path = root / "artifacts/base-mlx4" / (sources["model"]["revision"] + "-g64")
    provenance = json.loads((path / "provenance.json").read_text())
    baseline = json.loads((root / "docs/baseline_smoke_summary.json").read_text())
    if (provenance["source"] != sources["model"] or provenance["tokenizer_source"] != sources["tokenizer"]
            or provenance["source_weight_sha256"] != baseline["weights_sha256"]
            or provenance["quantization"] != {"bits": 4, "group_size": 64, "mode": "affine"}
            or config["activation_dtype"] != "float32"):
        raise ValueError("The recipe differs from the validated local conversion/precision")
    for row in provenance["files"]:
        if file_sha256(path / row["name"]) != row["sha256"]:
            raise ValueError(f"Converted base file changed: {row['name']}")
    if not mx.is_available(mx.gpu):
        raise RuntimeError("Native Apple GPU required")
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(int(mx.device_info()["max_recommended_working_set_size"] * config["memory_guideline_fraction"]))
    mx.set_cache_limit(config["cache_limit_bytes"])
    mx.random.seed(config["seed"])
    model, loaded = load_model(path, lazy=False, trust_remote_code=False)
    if loaded["quantization"] != provenance["quantization"]:
        raise ValueError("Loaded quantization differs from conversion provenance")
    enable_float32_activations(model)
    if adapter_path is None:
        info = attach_adapter(model, config["lora_rank"], config["lora_alpha"] / config["lora_rank"], config["lora_dropout"])
    else:
        model = load_adapters(model, str(adapter_path))
        info = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return model, tokenizer, info, provenance


def save_checkpoint(directory, model, optimizer, metadata, adapter_config, keep_last=2):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step-{metadata['completed_steps']:06d}"
    destination = directory / name
    temporary = directory / (name + "-incomplete")
    if destination.exists() or temporary.exists():
        raise ValueError("Checkpoint destination exists; retain it for review")
    temporary.mkdir()
    mx.eval(model.trainable_parameters(), optimizer.state)
    mx.save_safetensors(str(temporary / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(temporary / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    save_json(temporary / "adapter_config.json", adapter_config)
    record = {**metadata, "adapter_tensor_sha256": adapter_fingerprint(model),
              "files": [{"name": p.name, "sha256": file_sha256(p)} for p in sorted(temporary.iterdir())]}
    save_json(temporary / "checkpoint.json", record)
    temporary.rename(destination)
    pointer = directory / "latest.json.incomplete"
    pointer.write_text(json.dumps({"directory": name, "checkpoint_sha256": file_sha256(destination / "checkpoint.json")}) + "\n")
    pointer.replace(directory / "latest.json")
    # Only prune this function's completed, matching-recipe checkpoints.
    previous = []
    for candidate in directory.glob("step-*"):
        if candidate.is_dir() and (candidate / "checkpoint.json").exists():
            info = json.loads((candidate / "checkpoint.json").read_text())
            if info["recipe_sha256"] == metadata["recipe_sha256"]:
                previous.append(candidate)
    for candidate in sorted(previous)[:-max(2, keep_last)]:
        shutil.rmtree(candidate)
    return destination


def restore_checkpoint(directory, model, optimizer, recipe_sha256):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    directory = Path(directory)
    pointer = json.loads((directory / "latest.json").read_text())
    name = pointer["directory"]
    if Path(name).name != name or not name.startswith("step-"):
        raise ValueError("Invalid checkpoint pointer")
    path = directory / name
    if file_sha256(path / "checkpoint.json") != pointer["checkpoint_sha256"]:
        raise ValueError("Checkpoint pointer checksum differs")
    metadata = json.loads((path / "checkpoint.json").read_text())
    if metadata["recipe_sha256"] != recipe_sha256:
        raise ValueError("Resume recipe differs from the checkpoint")
    for row in metadata["files"]:
        if file_sha256(path / row["name"]) != row["sha256"]:
            raise ValueError("Checkpoint file checksum differs")
    weights = mx.load(str(path / "adapters.safetensors"))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in weights):
        raise ValueError("Resume adapter keys/shapes differ")
    model.load_weights(list(weights.items()), strict=False)
    optimizer.state = tree_unflatten(mx.load(str(path / "optimizer.safetensors")))
    mx.eval(model.trainable_parameters(), optimizer.state)
    if adapter_fingerprint(model) != metadata["adapter_tensor_sha256"]:
        raise ValueError("Reloaded checkpoint adapter differs")
    return metadata, path
