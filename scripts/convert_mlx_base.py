"""Locally quantize only the checksum-verified pinned base; record every output."""

import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256, save_json
from s1small.model import verify_checkpoint


def main():
    import mlx.core as mx
    from mlx_lm.utils import load_model, quantize_model, save_model, save_config
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "configs/sources.json").read_text())
    original = root / "artifacts/base" / sources["model"]["revision"]
    provenance = verify_checkpoint(original, sources)
    output = root / "artifacts/base-mlx4" / (sources["model"]["revision"] + "-g64")
    staging = output.with_name(output.name + "-incomplete")
    if output.exists() or staging.exists():
        raise ValueError("Existing conversion is retained; inspect it instead of overwriting")
    if not mx.is_available(mx.gpu):
        raise RuntimeError("Native Apple GPU required")
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024 * 1024)
    started = time.perf_counter()
    model, config = load_model(original, lazy=False, trust_remote_code=False)
    if config.get("quantization") or model.model.embed_tokens.weight.dtype != mx.bfloat16:
        raise ValueError("Expected the original unquantized BF16 weights")
    model, config = quantize_model(model, config, group_size=64, bits=4, mode="affine")
    mx.eval(model.parameters())
    save_model(staging, model)
    save_config(config, staging / "config.json")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copyfile(original / name, staging / name)
    files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)}
             for p in sorted(staging.iterdir()) if p.is_file()]
    record = {"source": sources["model"], "tokenizer_source": sources["tokenizer"],
              "source_weight_sha256": next(row["sha256"] for row in provenance["files"] if row["name"] == "model.safetensors"),
              "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
              "reason": "Original-base FP32 activation profile had a stable norm but 9348822140 peak MLX active bytes on an 8GB Mac; low-precision pretrained profiles were numerically unstable.",
              "packages": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "mlx-lm")},
              "conversion_code_sha256": file_sha256(Path(__file__).resolve()), "files": files,
              "seconds": time.perf_counter() - started, "total_file_bytes": sum(row["bytes"] for row in files)}
    save_json(staging / "provenance.json", record)
    staging.rename(output)
    print(json.dumps({"output": str(output.relative_to(root)), "bytes": record["total_file_bytes"],
                      "seconds": record["seconds"], "quantization": record["quantization"]}, indent=2))


if __name__ == "__main__":
    main()
