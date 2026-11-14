"""Explicit local base-model loading with provenance and backend checks."""

import json
from pathlib import Path
import time

from s1small.data import file_sha256


def verify_checkpoint(directory, sources):
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["source"] != sources["model"] or provenance["tokenizer_source"] != sources["tokenizer"]:
        raise ValueError("Local checkpoint model/tokenizer does not match pinned sources")
    for row in provenance["files"]:
        if file_sha256(directory / row["name"]) != row["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {row['name']}")
    if "model.safetensors" not in {row["name"] for row in provenance["files"]}:
        raise ValueError("Missing checkpoint weight provenance")
    return provenance


def synchronize(torch, device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def load_base_model(directory, config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if config["attention_implementation"] == "s1_chunked":
        from s1small.attention import register_chunked_attention
        register_chunked_attention()
    device = config["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process. Use an approved native run; no silent CPU fallback is performed.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device not in ("cpu", "cuda", "mps") or config["dtype"] not in ("float16", "bfloat16", "float32"):
        raise ValueError("Unsupported explicit device or dtype")
    torch.set_num_threads(config["cpu_threads"])
    set_seed(config["seed"])
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(config["mps_memory_fraction"])
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False,
        dtype=getattr(torch, config["dtype"]), attn_implementation=config["attention_implementation"],
        use_safetensors=True,
    )
    model.to(device)
    if config["attention_implementation"] == "s1_chunked":
        if model.config.model_type != "qwen2" or model.config.attention_dropout != 0.0:
            raise ValueError("Chunked attention is audited for Qwen2 with zero attention dropout only")
        model.config.s1_attention_chunk_tokens = config.get("attention_chunk_tokens", 128)
    model.eval()
    synchronize(torch, device)
    return model, tokenizer, {"load_seconds": time.perf_counter() - started,
                              "parameter_count": sum(p.numel() for p in model.parameters()),
                              "actual_dtype": str(next(model.parameters()).dtype),
                              "actual_device": str(next(model.parameters()).device),
                              "quantization": None, "adapter": None}
