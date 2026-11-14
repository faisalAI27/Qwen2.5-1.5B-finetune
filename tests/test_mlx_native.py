"""Run explicitly with RUN_MLX_NATIVE=1 in a native GPU-enabled process."""

import os
import unittest

if os.environ.get("RUN_MLX_NATIVE") == "1":
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
else:
    mx = None

from s1small.mlx_backend import (attach_adapter, causal_loss, enable_checkpointing,
                                 enable_float32_activations, gradient_statistics)


@unittest.skipIf(mx is None, "Native MLX tests require explicit opt-in")
class MLXNativeTests(unittest.TestCase):
    def test_checkpoint_resume_matches_uninterrupted_adamw_with_dropout(self):
        import tempfile
        from pathlib import Path
        import mlx.optimizers as optim
        from s1small.mlx_runtime import save_checkpoint, restore_checkpoint
        from s1small.training_plan import dropout_seed
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_a = mx.array([0.2, -0.3])
                self.lora_b = mx.array([0.1, 0.05])
                self.dropout = nn.Dropout(0.2)
            def __call__(self, inputs):
                return self.dropout(inputs) @ self.lora_a + self.lora_b.sum()
        def make():
            model = Model()
            model.train()
            return model, optim.AdamW(learning_rate=1e-3, weight_decay=0.0, bias_correction=True)
        def step(model, optimizer, index):
            mx.random.seed(dropout_seed(42, 0, str(index)))
            loss, gradients = nn.value_and_grad(model, lambda m: (m(mx.array([[1., 2.], [3., 4.]])) ** 2).mean())(model)
            optimizer.update(model, gradients)
            mx.eval(loss, model.parameters(), optimizer.state)
        original, optimizer = make()
        step(original, optimizer, 0)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            saved = save_checkpoint(directory, original, optimizer,
                                    {"recipe_sha256": "synthetic", "completed_steps": 1}, {})
            fresh, restored_optimizer = make()
            metadata, path = restore_checkpoint(directory, fresh, restored_optimizer, "synthetic")
            self.assertEqual(path, saved)
            self.assertEqual(metadata["completed_steps"], 1)
            for index in (1, 2, 3):
                step(original, optimizer, index)
                step(fresh, restored_optimizer, index)
            for (_, left), (_, right) in zip(tree_flatten(original.parameters()), tree_flatten(fresh.parameters())):
                self.assertTrue(mx.array_equal(left, right).item())
            original_state = dict(tree_flatten(optimizer.state))
            restored_state = dict(tree_flatten(restored_optimizer.state))
            self.assertEqual(set(original_state), set(restored_state))
            for name, left in original_state.items():
                self.assertTrue(mx.array_equal(left, restored_state[name]).item(), name)
            with (saved / "optimizer.safetensors").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                restore_checkpoint(directory, fresh, restored_optimizer, "synthetic")

    def test_quantized_storage_with_float32_compute_preserves_adapter_scope(self):
        from mlx_lm.models.qwen2 import Model, ModelArgs
        model = Model(ModelArgs(model_type="qwen2", hidden_size=64, num_hidden_layers=1,
                                intermediate_size=128, num_attention_heads=2, rms_norm_eps=1e-6,
                                vocab_size=64, num_key_value_heads=1))
        nn.quantize(model, group_size=64, bits=4)
        original_weight = model.model.embed_tokens.weight
        enable_float32_activations(model)
        self.assertEqual(model.model.embed_tokens.weight.dtype, mx.uint32)
        self.assertTrue(mx.array_equal(model.model.embed_tokens.weight, original_weight).item())
        ids = mx.array([[1, 2, 3, 4, 5]])
        self.assertEqual(model.model.embed_tokens(ids).dtype, mx.float32)
        info = attach_adapter(model, rank=2, scale=2.0, dropout=0.0)
        self.assertEqual(info["trainable_tensors"], 8)
        labels = mx.array([[-100, -100, 3, 4, 5]])
        loss, gradients = nn.value_and_grad(model, causal_loss)(model, ids, labels)
        mx.eval(loss, gradients)
        self.assertTrue(mx.isfinite(loss).item())
        self.assertLess(gradient_statistics(gradients)["gradient_norm_before_clip"], 1e6)

    def test_lora_scale_matches_alpha_over_rank(self):
        from mlx_lm.tuner.lora import LoRALinear
        layer = LoRALinear(3, 2, r=2, scale=2.0, dropout=0.0)
        layer.linear.weight = mx.zeros((2, 3))
        layer.lora_a = mx.array([[1., 0.], [0., 1.], [1., 1.]])
        layer.lora_b = mx.array([[1., 2.], [3., 4.]])
        x = mx.array([[1., 2., 3.]])
        expected = 2.0 * ((x @ layer.lora_a) @ layer.lora_b)
        self.assertTrue(mx.allclose(layer(x), expected).item())

    def test_full_qwen_chunked_loss_and_adapter_gradients_match_dense(self):
        from mlx_lm.models.qwen2 import Model, ModelArgs
        mx.random.seed(42)
        model = Model(ModelArgs(model_type="qwen2", hidden_size=16, num_hidden_layers=2,
                                intermediate_size=32, num_attention_heads=2, rms_norm_eps=1e-6,
                                vocab_size=41, num_key_value_heads=1))
        info = attach_adapter(model, rank=2, scale=2.0, dropout=0.0)
        self.assertEqual(info["trainable_tensors"], 16)
        for layer in model.layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                projection = getattr(layer.self_attn, name)
                projection.lora_b = mx.random.normal(projection.lora_b.shape) * 0.02
        enable_checkpointing(model)
        ids = mx.array([[1, 2, 3, 4, 5, 6, 7]])
        labels = mx.array([[-100, -100, -100, 4, 5, 6, -100]])
        def dense(candidate, inputs, targets):
            logits = candidate(inputs[:, :-1]).astype(mx.float32)
            shifted = targets[:, 1:]
            mask = shifted != -100
            ce = nn.losses.cross_entropy(logits, mx.where(mask, shifted, 0))
            return mx.where(mask, ce, 0).sum() / mask.sum()
        expected, expected_grad = nn.value_and_grad(model, dense)(model, ids, labels)
        mx.eval(expected, expected_grad)
        for size in (1, 3, 100):
            with self.subTest(chunk=size):
                actual, actual_grad = nn.value_and_grad(model, lambda m, x, y: causal_loss(m, x, y, size))(model, ids, labels)
                mx.eval(actual, actual_grad)
                self.assertAlmostEqual(actual.item(), expected.item(), places=5)
                for (name, left), (other, right) in zip(tree_flatten(expected_grad), tree_flatten(actual_grad)):
                    self.assertEqual(name, other)
                    self.assertTrue(mx.allclose(left, right, rtol=1e-4, atol=1e-6).item(), name)

    def test_prompt_and_first_padding_target_are_masked(self):
        import numpy as np
        from types import SimpleNamespace
        class Decoder:
            def __call__(self, ids):
                return mx.zeros((*ids.shape, 3))
            def __init__(self):
                self.embed_tokens = SimpleNamespace(as_linear=lambda h: mx.broadcast_to(mx.array([1., 2., 3.]), h.shape))
        model = SimpleNamespace(model=Decoder(), args=SimpleNamespace(tie_word_embeddings=True))
        ids = mx.array([[0, 1, 2, 0, 0]])
        labels = mx.array([[-100, -100, 2, 0, -100]])
        actual = causal_loss(model, ids, labels, 2).item()
        logz = np.log(np.exp(np.array([1., 2., 3.])).sum())
        self.assertAlmostEqual(actual, float(((logz - 3) + (logz - 1)) / 2), places=6)


if __name__ == "__main__":
    unittest.main()
