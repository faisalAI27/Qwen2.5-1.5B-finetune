"""Explicit CUDA runtime adapter; leaves every frozen Mac source file unchanged."""

from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from s1small.data import file_sha256


@lru_cache(maxsize=1)
def runtime_signature():
    if platform.system() != "Linux":
        raise RuntimeError("This adapter is for a separate Linux/CUDA replication")
    expected = {"mlx": "0.32.4", "mlx-cuda-12": "0.32.4", "mlx-lm": "0.32.2",
                "transformers": "5.10.1", "numpy": "2.4.3"}
    packages = {name: importlib.metadata.version(name) for name in expected}
    if packages != expected:
        raise ValueError("Install the pinned Colab requirements in an isolated environment")
    line = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                                    "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(line) != 1:
        raise ValueError("Select exactly one NVIDIA GPU")
    name, driver, memory, capability = [value.strip() for value in line[0].split(",")]
    if float(capability) < 7.5 or tuple(map(int, driver.split("."))) < (550, 54, 14):
        raise RuntimeError("MLX CUDA 12 requires SM >= 7.5 and driver >= 550.54.14")
    root = Path(__file__).resolve().parents[2]
    return {"backend": "mlx_cuda12", "packages": packages, "python": platform.python_version(),
            "architecture": platform.machine(), "os": platform.system(), "kernel": platform.release(),
            "libc": list(platform.libc_ver()), "gpu": {"name": name, "driver": driver,
            "memory_total_mib": float(memory), "compute_capability": capability},
            "runtime_adapter_sha256": file_sha256(Path(__file__).resolve()),
            "parent_macos_freeze_sha256": file_sha256(root / "manifests/evaluation_freeze_macos.json")}


def load_cuda_model(root, config, adapter_path=None):
    import mlx.core as mx
    from mlx_lm.utils import load_model
    from mlx_lm.tuner.utils import load_adapters
    from transformers import AutoTokenizer
    from s1small.mlx_backend import attach_adapter, enable_float32_activations
    signature = runtime_signature()
    root = Path(root)
    sources = json.loads((root / "configs/sources.json").read_text())
    path = root / "artifacts/base-mlx4" / (sources["model"]["revision"] + "-g64")
    provenance = json.loads((path / "provenance.json").read_text())
    baseline = json.loads((root / "docs/baseline_smoke_summary.json").read_text())
    if (provenance["source"] != sources["model"] or provenance["tokenizer_source"] != sources["tokenizer"]
            or provenance["source_weight_sha256"] != baseline["weights_sha256"]
            or provenance["quantization"] != {"bits": 4, "group_size": 64, "mode": "affine"}
            or config["activation_dtype"] != "float32"):
        raise ValueError("CUDA must preserve the exact source-traced quantized base and FP32 recipe")
    for row in provenance["files"]:
        if file_sha256(path / row["name"]) != row["sha256"]:
            raise ValueError("Converted model file changed")
    if not mx.is_available(mx.gpu):
        raise RuntimeError("MLX cannot access the allocated CUDA GPU")
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(int(signature["gpu"]["memory_total_mib"] * 1024**2 * 0.9))
    mx.set_cache_limit(config["cache_limit_bytes"])
    mx.random.seed(config["seed"])
    model, loaded = load_model(path, lazy=False, trust_remote_code=False)
    if loaded["quantization"] != provenance["quantization"]:
        raise ValueError("CUDA loaded a different quantization")
    enable_float32_activations(model)
    if adapter_path is None:
        info = attach_adapter(model, config["lora_rank"], config["lora_alpha"] / config["lora_rank"], config["lora_dropout"])
    else:
        model = load_adapters(model, str(adapter_path))
        info = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return model, tokenizer, info, provenance


def activate():
    """Install only the declared backend hooks before importing the evaluator."""
    import s1small.mlx_decode as decoding
    runtime_signature()
    decoding.runtime_signature = runtime_signature
    decoding.load_local_model = load_cuda_model
