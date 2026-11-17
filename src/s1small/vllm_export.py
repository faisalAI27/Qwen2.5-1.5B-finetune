"""CPU, bounded-memory export of the frozen MLX affine-Q4 base and LoRA.

The dense base is reconstructed from Q4 codes/scales/biases, not substituted
with the original unquantized checkpoint. FP16 storage/compute is a declared
backend change. The original adapter remains FP32 on disk in PEFT layout.
"""

import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np

from s1small.data import file_sha256, save_json
from s1small.experiment import atomic_json


class TensorFile:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 100_000_000:
                raise ValueError("Unexpected safetensors header size")
            self.header = json.loads(stream.read(length))
        self.offset = 8 + length

    def names(self):
        return sorted(name for name in self.header if name != "__metadata__")

    def array(self, name):
        info = self.header[name]
        dtype = {"U32": "<u4", "F32": "<f4", "F16": "<f2", "BF16": "<u2"}[info["dtype"]]
        start, end = info["data_offsets"]
        if np.prod(info["shape"], dtype=np.int64) * np.dtype(dtype).itemsize != end - start:
            raise ValueError("Tensor shape/size mismatch")
        return np.memmap(self.path, mode="r", dtype=dtype, offset=self.offset + start, shape=tuple(info["shape"]))

    def float_rows(self, name, start, end):
        value = self.array(name)[start:end]
        if self.header[name]["dtype"] == "BF16":
            return (value.astype(np.uint32) << 16).view(np.float32)
        return np.asarray(value, dtype=np.float32)


def dequantize_q4(packed, scales, biases, group_size=64):
    if packed.ndim != 2 or scales.shape != biases.shape or group_size != 64:
        raise ValueError("Only the frozen two-dimensional affine Q4/group-64 format is supported")
    width = packed.shape[1] * 8
    if scales.shape != (packed.shape[0], width // group_size) or width % group_size:
        raise ValueError("Packed weight/group shapes differ")
    codes = ((np.asarray(packed, dtype=np.uint32)[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15)
    codes = codes.reshape(packed.shape[0], width // group_size, group_size).astype(np.float32)
    return (codes * np.asarray(scales, dtype=np.float32)[..., None]
            + np.asarray(biases, dtype=np.float32)[..., None]).reshape(packed.shape[0], width)


def write_tensor(path, name, shape, dtype, chunks):
    dtype_name, numpy_dtype = {"F16": ("F16", "<f2"), "F32": ("F32", "<f4")}[dtype]
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(numpy_dtype).itemsize
    header = json.dumps({"__metadata__": {"format": "pt"}, name: {
        "dtype": dtype_name, "shape": list(shape), "data_offsets": [0, expected]}}, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    written = 0
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header))); stream.write(header)
        for chunk in chunks:
            value = np.asarray(chunk, dtype=numpy_dtype, order="C")
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite or FP16-overflowing exported weight")
            raw = value.tobytes(order="C")
            stream.write(raw); written += len(raw)
    if written != expected:
        raise ValueError("Incomplete exported tensor")
    temporary.replace(path)


def convert_adapter(source, destination, configuration, expected_fingerprint=None):
    from safetensors.numpy import save_file
    source, destination = Path(source), Path(destination)
    tensors = TensorFile(source)
    output, fingerprint = {}, hashlib.sha256()
    rank = configuration["lora_parameters"]["rank"]
    scale = configuration["lora_parameters"]["scale"]
    layers = configuration["num_layers"]
    expected = {f"model.layers.{layer}.self_attn.{projection}.lora_{kind}"
                for layer in range(layers) for projection in ("q_proj", "k_proj", "v_proj", "o_proj") for kind in ("a", "b")}
    if set(tensors.names()) != expected:
        raise ValueError("Adapter keys differ from all-layer attention-only training recipe")
    for name in tensors.names():
        info = tensors.header[name]
        if info["dtype"] != "F32" or len(info["shape"]) != 2:
            raise ValueError("Expected two-dimensional FP32 LoRA tensors")
        value = np.array(tensors.array(name), dtype=np.float32, order="C")
        if not np.isfinite(value).all() or (value.shape[1] if name.endswith("lora_a") else value.shape[0]) != rank:
            raise ValueError("Invalid trained adapter values/rank")
        fingerprint.update(name.encode()); fingerprint.update(str(tuple(value.shape)).encode()); fingerprint.update(value.tobytes())
        key = "base_model.model." + re.sub(r"\.lora_([ab])$", lambda match: ".lora_" + match[1].upper() + ".weight", name)
        output[key] = np.ascontiguousarray(value.T)
    original_fingerprint = fingerprint.hexdigest()
    if expected_fingerprint is not None and original_fingerprint != expected_fingerprint:
        raise ValueError("Adapter fingerprint differs from the completed training checkpoint")
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "adapter_model.safetensors.incomplete"
    save_file(output, str(temporary), metadata={"format": "pt"})
    temporary.replace(destination / "adapter_model.safetensors")
    config = {"base_model_name_or_path": "../base", "bias": "none", "fan_in_fan_out": False,
              "inference_mode": True, "lora_alpha": rank * scale, "lora_dropout": configuration["lora_parameters"]["dropout"],
              "peft_type": "LORA", "r": rank, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
              "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False}
    save_json(destination / "adapter_config.json", config)
    # Read back the serialized tensors and verify exact transposes, not just
    # matching shapes. A/B orientation mistakes change every learned update.
    restored = TensorFile(destination / "adapter_model.safetensors")
    if set(restored.names()) != set(output) or any(not np.array_equal(restored.array(key), value) for key, value in output.items()):
        raise ValueError("PEFT adapter did not round-trip exactly")
    return {"original_adapter_tensor_sha256": original_fingerprint, "tensors": len(output),
            "rank": rank, "alpha": rank * scale, "scale": scale, "exact_transpose_roundtrip": True}


def verify_export(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    manifest = json.loads((directory / "export_manifest.json").read_text())
    for name, expected in manifest["inputs_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("Frozen conversion input changed: " + name)
    for name, expected in manifest["files_sha256"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError("Exported model/adapter changed: " + name)
    return manifest


def export(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if (destination / "export_manifest.json").exists():
        return verify_export(root, destination)
    sources = json.loads((root / "configs/sources.json").read_text())
    base = root / "artifacts/base-mlx4" / (sources["model"]["revision"] + "-g64")
    training = json.loads((root / "manifests/training_freeze.json").read_text())
    selected_path = root / "results/training" / training["selected_run_id"] / "summary.json"
    selected = json.loads(selected_path.read_text())
    adapter = root / selected["adapter_directory"]
    config = json.loads((base / "config.json").read_text())
    if config["model_type"] != "qwen2" or config.get("quantization") != {"bits": 4, "group_size": 64, "mode": "affine"}:
        raise ValueError("Unexpected frozen model/quantization")
    inputs = [base / "model.safetensors", base / "config.json", adapter / "adapters.safetensors",
              adapter / "adapter_config.json", selected_path, root / "manifests/training_freeze.json", Path(__file__).resolve()]
    inputs.extend(p for p in base.iterdir() if p.is_file() and p.suffix in (".json", ".jinja", ".txt") and p.name != "provenance.json")
    inputs_sha = {str(p.relative_to(root)): file_sha256(p) for p in inputs}
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "export_state.json"
    state = {"inputs_sha256": inputs_sha, "shards": {}}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["inputs_sha256"] != inputs_sha:
            raise ValueError("Partial export uses different inputs; use a separate destination")
    dense = destination / "base"
    dense.mkdir(exist_ok=True)
    tensors = TensorFile(base / "model.safetensors")
    names = [name for name in tensors.names() if not name.endswith((".scales", ".biases"))]
    weight_map, total, maximum_rounding_error = {}, 0, 0.0
    for index, name in enumerate(names, 1):
        info = tensors.header[name]
        quantized = info["dtype"] == "U32"
        stem = name.removesuffix(".weight")
        if quantized and (not name.endswith(".weight") or stem + ".scales" not in tensors.header or stem + ".biases" not in tensors.header):
            raise ValueError("Packed tensor lacks affine scale/bias")
        shape = list(info["shape"])
        if quantized:
            shape[-1] *= 8
        if len(shape) not in (1, 2):
            raise ValueError("Unexpected Qwen tensor dimensions")
        filename = f"model-{index:05d}-of-{len(names):05d}.safetensors"
        path = dense / filename
        previous = state["shards"].get(filename)
        if previous and path.exists() and file_sha256(path) == previous["sha256"]:
            error = previous["max_fp16_rounding_error"]
        else:
            error_box = [0.0]
            def chunks():
                for start in range(0, shape[0], 256):
                    end = min(start + 256, shape[0])
                    if quantized:
                        value = dequantize_q4(tensors.array(name)[start:end],
                            tensors.float_rows(stem + ".scales", start, end), tensors.float_rows(stem + ".biases", start, end))
                    else:
                        value = tensors.float_rows(name, start, end)
                    rounded = value.astype(np.float16)
                    error_box[0] = max(error_box[0], float(np.max(np.abs(value - rounded.astype(np.float32)))))
                    yield rounded
            write_tensor(path, name, shape, "F16", chunks())
            error = error_box[0]
            state["shards"][filename] = {"sha256": file_sha256(path), "max_fp16_rounding_error": error}
            atomic_json(state_path, state)
        total += int(np.prod(shape, dtype=np.int64)) * 2
        maximum_rounding_error = max(maximum_rounding_error, error)
        weight_map[name] = filename
        print(json.dumps({"conversion_tensor": index, "total_tensors": len(names), "name": name}), flush=True)
    config.pop("quantization", None); config.pop("quantization_config", None)
    config["torch_dtype"] = "float16"
    config["dtype"] = "float16"
    save_json(dense / "config.json", config)
    save_json(dense / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": weight_map})
    for path in inputs:
        if path.parent == base and path.name not in ("model.safetensors", "config.json", "model.safetensors.index.json"):
            (dense / path.name).write_bytes(path.read_bytes())
    adapter_record = convert_adapter(adapter / "adapters.safetensors", destination / "adapter",
        json.loads((adapter / "adapter_config.json").read_text()), selected["adapter_tensor_sha256"])
    files = {str(p.relative_to(destination)): file_sha256(p) for folder in (dense, destination / "adapter")
             for p in sorted(folder.iterdir()) if p.is_file()}
    record = {"status": "complete", "source": sources["model"], "inputs_sha256": inputs_sha, "files_sha256": files,
              "base_origin": "Frozen MLX affine Q4/group64 reconstructed by scale*code+bias, then rounded to FP16; original full-precision weights are not substituted.",
              "computation_change": "MLX FP32 activation/adapter compute changes to vLLM FP16 base/LoRA compute; independently validated replication, not numerical parity.",
              "dense_weight_bytes": total, "maximum_absolute_fp16_rounding_error": maximum_rounding_error,
              "adapter": adapter_record, "optimizer_updates": 0}
    save_json(destination / "export_manifest.json", record)
    return record
