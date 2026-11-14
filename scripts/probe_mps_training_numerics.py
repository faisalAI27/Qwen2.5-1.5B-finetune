"""Compare CPU/MPS gradients on a random Qwen with real attention head geometry."""

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.attention import register_chunked_attention
from s1small.training import attach_lora, chunked_causal_loss
from s1small.data import save_json


def main():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Native MPS access required")
    register_chunked_attention()
    config = Qwen2Config(vocab_size=1024, hidden_size=1536, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                         attention_dropout=0.0, use_cache=False, attn_implementation="s1_chunked")
    config.s1_attention_chunk_tokens = 128
    model, _ = attach_lora(Qwen2ForCausalLM(config), {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                                                    "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.0})
    for name, p in model.named_parameters():
        if ".lora_B." in name:
            torch.nn.init.normal_(p, std=0.01)
    ids = torch.randint(0, 1024, (1, 712))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    batch["labels"][:, :67] = -100
    reports = []
    for dtype in (torch.float32, torch.bfloat16):
        cpu = copy.deepcopy(model)
        # Use exactly BF16-rounded base weights on both devices, with CPU FP32
        # arithmetic as the independent reference; preserve FP32 adapter params.
        for p in cpu.parameters():
            if not p.requires_grad:
                p.data = p.data.to(dtype).float()
        mps = copy.deepcopy(cpu).to("mps")
        for p in mps.parameters():
            if not p.requires_grad:
                p.data = p.data.to(dtype)
        for candidate in (cpu, mps):
            candidate.train()
            candidate.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            candidate.enable_input_require_grads()
        expected = chunked_causal_loss(cpu, batch, 128)
        actual = chunked_causal_loss(mps, {k: v.to("mps") for k, v in batch.items()}, 128)
        expected.backward(); actual.backward()
        left = torch.cat([p.grad.flatten().double() for p in cpu.parameters() if p.requires_grad])
        right = torch.cat([p.grad.detach().cpu().flatten().double() for p in mps.parameters() if p.requires_grad])
        finite = bool(torch.isfinite(right).all().item())
        cosine = torch.nn.functional.cosine_similarity(left, right, dim=0).item() if finite else None
        relative = (torch.linalg.vector_norm(right - left) / torch.linalg.vector_norm(left)).item() if finite else None
        report = {"dtype": str(dtype), "cpu_loss": expected.item(), "mps_loss": actual.item(),
                  "cpu_gradient_norm": torch.linalg.vector_norm(left).item(),
                  "mps_gradient_norm": torch.linalg.vector_norm(right).item() if finite else None,
                  "mps_gradients_finite": finite, "gradient_cosine": cosine, "gradient_relative_l2_error": relative,
                  "accepted": finite and cosine > 0.99 and relative < 0.10}
        reports.append(report)
        print(json.dumps(report), flush=True)
        del cpu, mps, expected, actual, left, right
        import gc
        gc.collect(); torch.mps.empty_cache()
    save_json(Path("results/environment/mps-qwen-gradient-parity.json"), {"scope": "random_architecture_numerical_probe_not_training",
              "head_geometry": "12 query heads / 2 KV heads / 128 head dimensions", "sequence_length": 712,
              "layers": 2, "attention": "s1_chunked", "loss": "chunked", "lora_dropout": 0.0,
              "note": "CPU FP32 reference uses the same rounded base weights. This does not prove full-model gradient accuracy.",
              "reports": reports})
    if not all(row["accepted"] for row in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
