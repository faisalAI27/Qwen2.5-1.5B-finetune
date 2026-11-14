import copy
from types import SimpleNamespace
import unittest

try:
    import torch
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
except ImportError:
    torch = None

from s1small.attention import chunked_attention_forward, register_chunked_attention


@unittest.skipIf(torch is None, "Optional Torch/Transformers runtime absent")
class AttentionTests(unittest.TestCase):
    def test_causal_padding_outputs_and_qkv_gradients_match_eager(self):
        torch.manual_seed(42)
        for size in (1, 3, 100):
            with self.subTest(chunk=size):
                module = SimpleNamespace(config=SimpleNamespace(s1_attention_chunk_tokens=size),
                                         num_key_value_groups=2, training=True)
                tensors = [torch.randn(2, heads, 7, 4, requires_grad=True) for heads in (4, 2, 2)]
                clones = [value.detach().clone().requires_grad_(True) for value in tensors]
                mask = torch.zeros(2, 1, 7, 7)
                mask.masked_fill_(torch.ones(7, 7, dtype=torch.bool).triu(1), torch.finfo(torch.float32).min)
                mask[1, :, :, 5:] = torch.finfo(torch.float32).min
                expected, _ = eager_attention_forward(module, *tensors, mask, scaling=0.5)
                actual, weights = chunked_attention_forward(module, *clones, mask, scaling=0.5)
                self.assertIsNone(weights)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                upstream = torch.randn_like(actual)
                expected.backward(upstream); actual.backward(upstream)
                for value, clone in zip(tensors, clones):
                    torch.testing.assert_close(value.grad, clone.grad, atol=1e-6, rtol=1e-5)

    def test_actual_qwen_decoder_with_nested_checkpointing_matches_eager(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        register_chunked_attention()
        torch.manual_seed(42)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=41, hidden_size=16, intermediate_size=32,
                                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                           attention_dropout=0.0, use_cache=False,
                                           attn_implementation="eager"))
        clone = copy.deepcopy(model)
        clone.set_attn_implementation("s1_chunked")
        clone.config.s1_attention_chunk_tokens = 3
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 0, 0]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]]),
                 "labels": torch.tensor([[-100, -100, 3, 4, 5, 6, 7], [-100, -100, 4, 5, 6, -100, -100]])}
        for candidate in (model, clone):
            candidate.train()
            candidate.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        expected = model(**batch).loss
        actual = clone(**batch).loss
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        expected.backward(); actual.backward()
        for (name, p), (other_name, q) in zip(model.named_parameters(), clone.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)

    def test_rejects_unmasked_multitoken_and_attention_dropout(self):
        module = SimpleNamespace(config=SimpleNamespace(), num_key_value_groups=1)
        query = torch.randn(1, 1, 3, 4)
        with self.assertRaisesRegex(ValueError, "causal"):
            chunked_attention_forward(module, query, query, query, None, scaling=0.5)
        with self.assertRaisesRegex(ValueError, "dropout"):
            chunked_attention_forward(module, query, query, query, None, scaling=0.5, dropout=0.1)


if __name__ == "__main__":
    unittest.main()
