"""Concurrent vLLM bridge for the unchanged bounded reasoning state machine."""

import asyncio
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import threading
import time
import uuid

from s1small.budget_forcing import Segment
from s1small.data import file_sha256
from s1small.mlx_decode import MLXDecoder

CODE_FILES = ("scripts/runpod_vllm.py", "src/s1small/vllm_backend.py", "src/s1small/vllm_export.py",
              "configs/runpod_vllm_v1.json", "configs/vllm_prompt_reference_v1.json", "requirements-vllm.txt")


@lru_cache(maxsize=1)
def runtime_signature():
    if platform.system() != "Linux":
        raise ValueError("vLLM generation requires the separate Linux CUDA workspace")
    import torch
    expected = {"vllm": "0.11.2", "torch": "2.9.0", "transformers": "4.57.1", "numpy": "2.2.6",
                "tokenizers": "0.22.1", "huggingface-hub": "0.36.0"}
    packages = {name: importlib.metadata.version(name) for name in expected}
    if any(packages[name].split("+")[0] != value for name, value in expected.items()):
        raise ValueError("Install requirements-vllm.txt in its own environment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Allocate exactly one NVIDIA GPU to this replication")
    line = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(line) != 1:
        raise ValueError("One-GPU hardware identity required")
    name, driver, memory, capability = [part.strip() for part in line[0].split(",")]
    if float(capability) < 8.0:
        raise ValueError("Use an Ampere-or-newer GPU such as RTX 4090 for this vLLM route")
    root = Path(__file__).resolve().parents[2]
    return {"backend": "vllm_fp16_reconstructed_q4_lora", "provider": "runpod", "packages": packages,
        "installed_distributions": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata.get("Name"))),
        "python": platform.python_version(), "architecture": platform.machine(), "kernel": platform.release(),
        "libc": list(platform.libc_ver()), "torch_cuda": torch.version.cuda,
        "gpu": {"name": name, "driver": driver, "memory_total_mib": float(memory), "compute_capability": capability},
        "engine_settings": json.loads((root / "configs/runpod_vllm_v1.json").read_text()),
        "code_sha256": {name: file_sha256(root / name) for name in CODE_FILES},
        "export_manifest_sha256": file_sha256(root / "artifacts/vllm-export/export_manifest.json"),
        "parent_macos_freeze_sha256": file_sha256(root / "manifests/evaluation_freeze_macos.json")}


class AsyncBridge:
    """Only the asyncio loop accesses vLLM; state-machine threads await it."""
    def __init__(self, engine, loop, settings, sampling_factory=None):
        self.engine, self.loop, self.settings = engine, loop, settings
        self.futures = set()
        self.lock = threading.Lock()
        self.closed = False
        self.sampling_factory = sampling_factory

    async def collect(self, context, maximum, stops, deadline, lora):
        if maximum <= 0 or not context or len(context) + maximum > self.settings["max_model_len"]:
            raise ValueError("Nonempty context, positive allowance and no-truncation context bound required")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return Segment((), timed_out=True)
        if self.sampling_factory is None:
            from vllm import SamplingParams
            factory = SamplingParams
        else:
            factory = self.sampling_factory
        params = factory(temperature=0, top_p=1.0, max_tokens=maximum, stop_token_ids=sorted(stops),
            ignore_eos=True, skip_special_tokens=False, detokenize=False, seed=self.settings["seed"])
        identifier = "s1-" + uuid.uuid4().hex
        emitted = ()
        async def consume():
            nonlocal emitted
            async for output in self.engine.generate(prompt={"prompt_token_ids": list(context)},
                    sampling_params=params, request_id=identifier, lora_request=lora):
                if len(output.outputs) != 1:
                    raise ValueError("Greedy one-completion output required")
                current = tuple(int(t) for t in output.outputs[0].token_ids)
                if current[:len(emitted)] != emitted:
                    raise ValueError("vLLM stream must contain cumulative token IDs")
                emitted = current
        try:
            await asyncio.wait_for(consume(), timeout=remaining)
            timed_out = False
        except asyncio.TimeoutError:
            await self.engine.abort(identifier)
            timed_out = True
        if len(emitted) > maximum or any(t < 0 for t in emitted) or any(t in stops for t in emitted[:-1]):
            raise ValueError("vLLM violated the counted stop-token contract")
        if not emitted and not timed_out:
            raise ValueError("vLLM returned no counted tokens")
        return Segment(emitted, timed_out=timed_out)

    def generate(self, context, maximum, stops, deadline, lora):
        with self.lock:
            if self.closed:
                raise RuntimeError("vLLM bridge is shutting down")
            future = asyncio.run_coroutine_threadsafe(self.collect(context, maximum, stops, deadline, lora), self.loop)
            self.futures.add(future)
        try:
            return future.result()
        finally:
            with self.lock:
                self.futures.discard(future)

    def close(self):
        with self.lock:
            self.closed = True
            pending = tuple(self.futures)
        for future in pending:
            future.cancel()


class VLLMDecoder:
    decode = MLXDecoder.decode

    def __init__(self, bridge, tokenizer, eos_ids, lora=None):
        self.bridge, self.tokenizer, self.eos_ids, self.lora = bridge, tokenizer, tuple(eos_ids), lora
        self.unmapped = set()

    def generate(self, context, maximum, stop_ids, deadline):
        return self.bridge.generate(context, maximum, stop_ids, deadline, self.lora)


def create_engine(root, settings):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    root = Path(root)
    directory = root / "artifacts/vllm-export"
    args = AsyncEngineArgs(model=str(directory / "base"), tokenizer=str(directory / "base"),
        dtype=settings["dtype"], tensor_parallel_size=settings["tensor_parallel_size"],
        max_model_len=settings["max_model_len"], max_num_seqs=settings["max_num_seqs"],
        gpu_memory_utilization=settings["gpu_memory_utilization"], enable_lora=True, max_loras=1,
        max_lora_rank=settings["max_lora_rank"], lora_dtype=settings["lora_dtype"],
        enable_prefix_caching=settings["enable_prefix_caching"], enforce_eager=settings["enforce_eager"],
        generation_config=settings["generation_config"], seed=settings["seed"], trust_remote_code=False,
        enable_chunked_prefill=True, max_num_batched_tokens=8192, disable_log_stats=False)
    engine = AsyncLLM.from_engine_args(args)
    tokenizer = AutoTokenizer.from_pretrained(directory / "base", local_files_only=True, trust_remote_code=False)
    eos = json.loads((directory / "base/generation_config.json").read_text())["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    lora = LoRARequest("s1-trained", 1, str(directory / "adapter"))
    return engine, tokenizer, eos, lora
