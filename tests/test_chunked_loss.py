import copy
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from s1small.training import chunked_causal_loss, clip_gradients_cpu_norm


@unittest.skipIf(torch is None, "Optional Torch runtime is absent")
class ChunkedLossTests(unittest.TestCase):
    def test_global_norm_and_clipping_do_not_overflow_float32(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3e20, 4e20])
        report = clip_gradients_cpu_norm([parameter], 1.0)
        self.assertAlmostEqual(report["gradient_norm_before_clip"] / 5e20, 1.0, places=6)
        torch.testing.assert_close(parameter.grad, torch.tensor([0.6, 0.8]))
        parameter.grad = torch.tensor([3.0, 4.0])
        clip_gradients_cpu_norm([parameter], 10.0)
        torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 4.0]))
        parameter.grad[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            clip_gradients_cpu_norm([parameter])

    def make_model(self):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(13, 7)
                self.adapter = torch.nn.Linear(7, 7, bias=False)
                self.embedding.weight.requires_grad_(False)

            def forward(self, input_ids, attention_mask, use_cache=False):
                return SimpleNamespace(last_hidden_state=self.adapter(self.embedding(input_ids)))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="qwen2")
                self.model = Decoder()
                self.head = torch.nn.Linear(7, 13, bias=False)
                self.head.weight.requires_grad_(False)

            def get_base_model(self):
                return self

            def get_output_embeddings(self):
                return self.head

        torch.manual_seed(42)
        return Model()

    def test_matches_dense_loss_and_gradients_with_padding_and_prompt_masks(self):
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 0, 0]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
                 "labels": torch.tensor([[-100, -100, -100, 4, 5, 6], [-100, -100, 4, 5, -100, -100]])}
        for size in (1, 2, 4, 100):
            with self.subTest(chunk=size):
                dense = self.make_model()
                chunked = copy.deepcopy(dense)
                states = dense.model(batch["input_ids"], batch["attention_mask"]).last_hidden_state
                expected = torch.nn.functional.cross_entropy(dense.head(states[:, :-1]).float().reshape(-1, 13),
                                                               batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
                actual = chunked_causal_loss(chunked, batch, size)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                expected.backward(); actual.backward()
                torch.testing.assert_close(chunked.model.adapter.weight.grad, dense.model.adapter.weight.grad,
                                           atol=1e-6, rtol=1e-6)
                self.assertIsNone(chunked.head.weight.grad)
                with torch.no_grad():
                    torch.testing.assert_close(chunked_causal_loss(chunked, batch, size), expected.detach())

    def test_rejects_empty_targets_and_trainable_head(self):
        model = self.make_model()
        batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
                 "labels": torch.full((1, 3), -100)}
        with self.assertRaisesRegex(ValueError, "No supervised"):
            chunked_causal_loss(model, batch)
        with self.assertRaisesRegex(ValueError, "positive"):
            chunked_causal_loss(model, batch, 0)
        model.head.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            chunked_causal_loss(model, batch)

    def test_real_qwen_lora_matches_standard_forward_and_gradients(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("Optional Transformers/PEFT runtime is absent")
        torch.manual_seed(42)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=41, hidden_size=16, intermediate_size=32,
                                           num_hidden_layers=2, num_attention_heads=2,
                                           num_key_value_heads=1, tie_word_embeddings=True,
                                           attention_dropout=0.0, use_cache=False))
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                                lora_dropout=0.0,
                                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        # Nonzero B exercises gradients through both adapter matrices.
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                torch.nn.init.normal_(parameter, std=0.02)
        clone = copy.deepcopy(model)
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
                 "attention_mask": torch.ones(1, 8, dtype=torch.long),
                 "labels": torch.tensor([[-100, -100, -100, 4, 5, 6, 7, 8]])}
        model.train(); clone.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        clone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads(); clone.enable_input_require_grads()
        expected = model(**batch).loss
        actual = chunked_causal_loss(clone, batch, 3)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        expected.backward(); actual.backward()
        for (name, p), (other_name, q) in zip(model.named_parameters(), clone.named_parameters()):
            self.assertEqual(name, other_name)
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
