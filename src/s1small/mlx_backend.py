"""Explicit MLX Qwen2 LoRA, using the audited complete-example label mask."""

import hashlib
import math


def enable_float32_activations(model):
    """Keep the original BF16 base storage, promoting decoder compute to FP32."""
    import mlx.core as mx
    import mlx.nn as nn
    class Float32QuantizedEmbedding(nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding
        def __call__(self, indices):
            return self.embedding(indices).astype(mx.float32)
        def as_linear(self, states):
            return self.embedding.as_linear(states)
        @property
        def weight(self):
            return self.embedding.weight
    class Float32Embedding(nn.Embedding):
        def __call__(self, indices):
            return super().__call__(indices).astype(mx.float32)
    original = model.model.embed_tokens
    if isinstance(original, nn.QuantizedEmbedding):
        replacement = Float32QuantizedEmbedding(original)
    else:
        replacement = Float32Embedding(*original.weight.shape)
        replacement.weight = original.weight
    replacement.freeze()
    model.model.embed_tokens = replacement


def attach_adapter(model, rank=16, scale=2.0, dropout=0.05):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    if model.model_type != "qwen2":
        raise ValueError("Only the audited Qwen2 backend is supported")
    model.freeze()
    keys = ["self_attn." + name for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
    parameters = {"rank": rank, "scale": scale, "dropout": dropout, "keys": keys}
    linear_to_lora_layers(model, model.args.num_hidden_layers, parameters)
    leaves = tree_flatten(model.trainable_parameters())
    if len(leaves) != model.args.num_hidden_layers * 8 or any(name.rsplit(".", 1)[-1] not in ("lora_a", "lora_b") for name, _ in leaves):
        raise ValueError("Expected exactly the attention LoRA A/B tensors to be trainable")
    if any(value.dtype != mx.float32 for _, value in leaves):
        raise ValueError("Expected FP32 adapter parameters")
    return {"fine_tune_type": "lora", "num_layers": model.args.num_hidden_layers,
            "lora_parameters": parameters, "trainable_tensors": len(leaves),
            "trainable_parameters": sum(value.size for _, value in leaves)}


def enable_checkpointing(model):
    from mlx_lm.tuner.trainer import grad_checkpoint
    cls = type(model.layers[0])
    if not getattr(cls, "_s1_checkpointed", False):
        grad_checkpoint(model.layers[0])
        cls._s1_checkpointed = True


def causal_loss(model, input_ids, labels, chunk_tokens=128):
    import mlx.core as mx
    import mlx.nn as nn
    if chunk_tokens < 1:
        raise ValueError("Loss chunk size must be positive")
    # Micro-batch 1 has no added padding. Labels still explicitly mask prompt
    # and any supplied padding, including the first padding target after EOS.
    targets = labels[:, 1:]
    hidden = model.model(input_ids[:, :-1])
    def project_loss(states, target):
        target = mx.stop_gradient(target)
        logits = (model.model.embed_tokens.as_linear(states) if model.args.tie_word_embeddings
                  else model.lm_head(states)).astype(mx.float32)
        mask = target != -100
        safe_target = mx.where(mask, target, 0)
        ce = nn.losses.cross_entropy(logits, safe_target)
        return mx.where(mask, ce, 0).sum()
    total = mx.array(0.0, dtype=mx.float32)
    for start in range(0, targets.shape[1], chunk_tokens):
        total = total + mx.checkpoint(project_loss)(hidden[:, start:start + chunk_tokens],
                                                   targets[:, start:start + chunk_tokens])
    return total / (targets != -100).sum()


def gradient_statistics(gradients):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    leaves = tree_flatten(gradients)
    mx.eval(gradients)
    norms, maximum = [], 0.0
    for name, gradient in leaves:
        # An independent CPU float64 reduction also guards against the extreme
        # finite values seen in the previous PyTorch pilot.
        value = np.array(gradient).astype(np.float64)
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite MLX gradient: {name}")
        norms.append(float(np.linalg.norm(value)))
        maximum = max(maximum, float(np.abs(value).max()))
    norm = math.hypot(*norms)
    if not leaves or not math.isfinite(norm) or norm == 0:
        raise ValueError("Invalid or zero global adapter gradient norm")
    return {"trainable_tensors_with_grad": len(leaves), "all_gradients_finite": True,
            "gradient_norm_before_clip": norm, "maximum_absolute_gradient": maximum}


def adapter_fingerprint(model):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    checksum = hashlib.sha256()
    count = 0
    for name, value in sorted(tree_flatten(model.parameters())):
        if name.rsplit(".", 1)[-1] not in ("lora_a", "lora_b"):
            continue
        mx.eval(value)
        array = np.array(value.astype(mx.float32))
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite saved adapter parameter")
        checksum.update(name.encode()); checksum.update(str(tuple(value.shape)).encode())
        checksum.update(array.tobytes())
        count += 1
    if not count:
        raise ValueError("No adapter parameters")
    return checksum.hexdigest()
