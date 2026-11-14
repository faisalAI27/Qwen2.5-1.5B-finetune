"""Query-chunked eager Qwen2 attention: same keys/softmax, bounded score buffers."""


def chunked_attention_forward(module, query, key, value, attention_mask, scaling,
                              dropout=0.0, **kwargs):
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    if dropout != 0.0:
        raise ValueError("The audited Qwen2 attention path requires zero attention dropout")
    if kwargs.get("output_attentions", False):
        raise ValueError("Chunked attention does not materialize full attention weights")
    if attention_mask is None and query.shape[-2] > 1:
        raise ValueError("Multi-query attention requires an explicit causal mask")
    size = getattr(module.config, "s1_attention_chunk_tokens", 128)
    if size < 1:
        raise ValueError("Attention query chunk size must be positive")
    keys = repeat_kv(key, module.num_key_value_groups)
    values = repeat_kv(value, module.num_key_value_groups)

    def attend(queries, all_keys, all_values, mask):
        scores = torch.matmul(queries, all_keys.transpose(2, 3)) * scaling
        if mask is not None:
            scores = scores + mask
        probabilities = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(queries.dtype)
        return torch.matmul(probabilities, all_values)

    outputs = []
    for start in range(0, query.shape[-2], size):
        queries = query[:, :, start:start + size, :]
        mask = attention_mask[..., start:start + size, :] if attention_mask is not None else None
        # Checkpointing bounds backward's softmax buffers as well as forward's.
        result = (checkpoint(attend, queries, keys, values, mask, use_reentrant=False)
                  if torch.is_grad_enabled() else attend(queries, keys, values, mask))
        outputs.append(result)
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None


def register_chunked_attention():
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import eager_mask
    AttentionInterface.register("s1_chunked", chunked_attention_forward)
    # Preserve the exact standard eager causal/padding/sliding mask semantics.
    AttentionMaskInterface.register("s1_chunked", eager_mask)
