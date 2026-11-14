"""LoRA attachment and inspection; actual pilot execution is kept separate."""

import platform
import resource
import threading
import math


def accumulation_windows(examples, size):
    if size < 1:
        raise ValueError("Accumulation size must be positive")
    for start in range(0, len(examples), size):
        yield examples[start:start + size]


def supervised_count(example):
    # Causal shifting drops label position zero, which is in the masked prompt.
    count = sum(value != -100 for value in example["labels"][1:])
    if count < 1:
        raise ValueError("No supervised next-token targets")
    return count


def chunked_causal_loss(model, batch, chunk_tokens=128):
    """Exact supervised-token mean CE, with checkpointed vocabulary projections.

    Qwen2's usual loss holds sequence × vocabulary logits and FP32 CE buffers.
    Recompute each small output-head chunk during backward instead. Decoder
    gradient checkpointing remains the caller's responsibility. This path is
    restricted to ordinary Qwen2 LoRA with a frozen output head.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    if chunk_tokens < 1:
        raise ValueError("Loss chunk size must be positive")
    base = model.get_base_model()
    if base.config.model_type != "qwen2":
        raise ValueError("Chunked decoder path is audited for Qwen2 only")
    head = base.get_output_embeddings()
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("This memory path requires a frozen output head")
    labels = batch["labels"][:, 1:]
    count = (labels != -100).sum()
    if not bool((count > 0).item()):
        raise ValueError("No supervised next-token targets")
    hidden = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False).last_hidden_state[:, :-1, :]

    def project_loss(states, targets):
        logits = head(states).float()
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                  targets.reshape(-1), ignore_index=-100,
                                                  reduction="sum")

    total = None
    for start in range(0, labels.shape[1], chunk_tokens):
        targets = labels[:, start:start + chunk_tokens]
        if not bool((targets != -100).any().item()):
            continue
        states = hidden[:, start:start + chunk_tokens, :]
        loss = checkpoint(project_loss, states, targets, use_reentrant=False) if torch.is_grad_enabled() else project_loss(states, targets)
        total = loss if total is None else total + loss
    return total / count


class MemorySampler:
    """Record sampled MPS allocator peaks, without calling them exact VRAM peaks."""

    def __init__(self, device, interval=0.05):
        import torch
        self.torch = torch
        self.device = device
        self.interval = interval
        self.stop = threading.Event()
        self.samples = 0
        self.allocated = 0
        self.driver = 0
        self.error = None

    def sample(self):
        try:
            if self.device == "mps":
                self.allocated = max(self.allocated, self.torch.mps.current_allocated_memory())
                self.driver = max(self.driver, self.torch.mps.driver_allocated_memory())
            self.samples += 1
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def collect(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=1)
        self.sample()

    def report(self):
        return {"sampling_interval_seconds": self.interval, "samples": self.samples,
                "sampled_peak_mps_allocated_bytes": self.allocated if self.device == "mps" and self.samples else None,
                "sampled_peak_mps_driver_bytes": self.driver if self.device == "mps" and self.samples else None,
                "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
                "sampling_error": self.error,
                "note": "MPS values are sampled peaks, not exact peak VRAM. RSS and MPS memory overlap on unified memory; do not add them."}


def gradient_report(model):
    import torch
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    finite = len(gradients) == len(parameters) and bool(gradients) and all(bool(torch.isfinite(value).all().item()) for value in gradients)
    nonzero = any(bool(torch.count_nonzero(value).item()) for value in gradients) if finite else False
    if not finite or not nonzero:
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite = [name for name, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None and not bool(torch.isfinite(p.grad).all().item())]
        raise ValueError(f"Invalid adapter gradients: tensors={len(parameters)}, present={len(gradients)}, "
                         f"all_finite={finite}, some_nonzero={nonzero}, missing={missing[:8]}, nonfinite={nonfinite[:8]}")
    return {"trainable_tensors": len(parameters), "tensors_with_grad": len(gradients),
            "all_present_gradients_finite": finite, "some_gradient_nonzero": nonzero}


def clip_gradients_cpu_norm(parameters, max_norm=1.0):
    """Compute the global norm in CPU FP64; scale the original device gradients.

    Avoid FP32 squared-sum overflow and retain an independently observed norm.
    Only the small LoRA gradients are copied, never base weights/activations.
    """
    import torch
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("Gradient norm limit must be positive and finite")
    parameters = list(parameters)
    norms, maximum = [], 0.0
    for parameter in parameters:
        if parameter.grad is None:
            raise ValueError("Missing adapter gradient")
        gradient = parameter.grad.detach().cpu().double()
        if not bool(torch.isfinite(gradient).all().item()):
            raise ValueError("Nonfinite adapter gradient in CPU inspection")
        norms.append(torch.linalg.vector_norm(gradient).item())
        maximum = max(maximum, gradient.abs().max().item())
    norm = math.hypot(*norms)
    if not norms or not math.isfinite(norm) or norm == 0:
        raise ValueError("Invalid global gradient norm")
    scale = min(1.0, max_norm / (norm + 1e-6))
    with torch.no_grad():
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return {"gradient_norm_before_clip": norm, "maximum_absolute_gradient_before_clip": maximum,
            "gradient_clip_scale": scale, "gradient_norm_implementation": "cpu_float64_hypot"}


def adapter_fingerprint(model):
    import hashlib
    checksum = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        checksum.update(name.encode())
        checksum.update(str(tuple(parameter.shape)).encode())
        checksum.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        count += 1
    if not count:
        raise ValueError("No adapter parameters to fingerprint")
    return checksum.hexdigest()


def attach_lora(base_model, config):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    targets = tuple(config["target_modules"])
    selected = [name for name, module in base_model.named_modules()
                if isinstance(module, torch.nn.Linear) and name.endswith(targets)]
    if len(selected) != len(targets) * base_model.config.num_hidden_layers:
        raise ValueError("Attention projection count differs from the declared layer/target configuration")
    if any(".self_attn." not in name for name in selected):
        raise ValueError("A LoRA target matched outside attention")
    adapter = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=list(targets),
        bias="none", use_rslora=False,
    )
    model = get_peft_model(base_model, adapter, autocast_adapter_dtype=True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(".lora_A." not in name and ".lora_B." not in name for name, _ in trainable):
        raise ValueError("Expected only LoRA A/B parameters to be trainable")
    scales = {float(module.scaling["default"]) for module in model.modules() if hasattr(module, "lora_A") and "default" in module.lora_A}
    if scales != {config["lora_alpha"] / config["lora_rank"]}:
        raise ValueError("LoRA scaling differs from the declared alpha/r convention")
    return model, {
        "target_module_names": selected, "target_module_count": len(selected),
        "trainable_parameter_count": sum(p.numel() for _, p in trainable),
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "effective_scaling": next(iter(scales)),
        "trainable_dtypes": sorted({str(p.dtype) for _, p in trainable}),
        "trainable_names": [name for name, _ in trainable],
        "base_parameters_frozen": True,
    }
