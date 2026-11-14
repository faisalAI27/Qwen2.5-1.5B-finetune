from types import SimpleNamespace
import unittest

from s1small.mlx_decode import MLXDecoder, s1_prompt


class DecodeTests(unittest.TestCase):
    def test_unused_embedding_ids_remain_visible_and_recorded(self):
        class Tokenizer:
            def __len__(self): return 3
            def decode(self, ids, **kwargs): return "".join(str(value) for value in ids)
        model = SimpleNamespace(eval=lambda: None)
        decoder = MLXDecoder(model, Tokenizer(), [2])
        self.assertEqual(decoder.decode([1, 4, 2]), "1<|unmapped_token_4|>2")
        self.assertEqual(decoder.unmapped, {4})

    def test_s1_prompt_contains_question_and_exact_training_prefix_only(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "chat-prefix"
        tokenizer = Tokenizer()
        value = s1_prompt("A fixed question", tokenizer)
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "A fixed question"}])
        self.assertEqual(value, "chat-prefix<|im_start|>think\n")
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])
