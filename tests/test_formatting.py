import importlib.util
import unittest

from s1small.formatting import (ContextOverflowError, collate_training_examples,
                               format_example, make_training_example)


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]


def row(question="Calculate 1+1"):
    return {"question": question, "thinking_trajectories": ["One plus one makes two."], "attempt": r"\boxed{2}"}


class TrainingFormatTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch optional in the data-only environment")
    def test_tensor_batch_is_integer_and_matches_python_labels(self):
        import torch
        examples = [make_training_example(row(), CharacterTokenizer()), make_training_example(row("Longer question text"), CharacterTokenizer())]
        batch = collate_training_examples(examples, 999, return_tensors=True)
        plain = collate_training_examples(examples, 999)
        for key in ("input_ids", "attention_mask", "labels"):
            self.assertEqual(batch[key].dtype, torch.long)
            self.assertEqual(batch[key].tolist(), plain[key])

    def test_prompt_masked_and_every_assistant_token_supervised(self):
        tokenizer = CharacterTokenizer()
        encoded = make_training_example(row(), tokenizer)
        start = encoded["prompt_tokens"]
        self.assertEqual(encoded["labels"][:start], [-100] * start)
        self.assertEqual(encoded["labels"][start:], encoded["input_ids"][start:])
        supervised = "".join(chr(value) for value in encoded["labels"] if value != -100)
        self.assertTrue(supervised.startswith("<|im_start|>think\n"))
        self.assertIn("<|im_start|>answer\nAnswer: \\boxed{2}", supervised)
        self.assertTrue(supervised.endswith("<|im_end|>\n"))

    def test_loss_labels_are_not_manually_shifted(self):
        encoded = make_training_example(row(), CharacterTokenizer())
        for index in range(encoded["prompt_tokens"], len(encoded["input_ids"])):
            self.assertEqual(encoded["labels"][index], encoded["input_ids"][index])

    def test_overflow_rejects_without_losing_answer_and_exact_limit_passes(self):
        tokenizer = CharacterTokenizer()
        original = make_training_example(row(), tokenizer)
        length = len(original["input_ids"])
        self.assertEqual(original, make_training_example(row(), tokenizer, max_length=length))
        with self.assertRaises(ContextOverflowError):
            make_training_example(row(), tokenizer, max_length=length - 1)
        self.assertEqual(original, make_training_example(row(), tokenizer))

    def test_padding_masks_only_added_positions_and_preserves_end_tokens(self):
        tokenizer = CharacterTokenizer()
        first = make_training_example(row(), tokenizer)
        second = make_training_example(row("A considerably longer mathematical question"), tokenizer)
        batch = collate_training_examples([first, second], pad_token_id=999)
        length = len(first["input_ids"])
        self.assertEqual(batch["labels"][0][:length], first["labels"])
        self.assertTrue(all(value == -100 for value in batch["labels"][0][length:]))
        self.assertTrue(all(value == 0 for value in batch["attention_mask"][0][length:]))
        self.assertTrue(all(value == 999 for value in batch["input_ids"][0][length:]))
        self.assertEqual(batch["labels"][0][length - 1], ord("\n"))

    def test_collator_does_not_mask_real_tokens_by_token_id(self):
        encoded = make_training_example(row(), CharacterTokenizer())
        # A token ID that also occurs in actual content must not erase labels.
        batch = collate_training_examples([encoded], pad_token_id=ord("2"))
        self.assertIn(ord("2"), batch["labels"][0])

    def test_empty_or_misaligned_batch_rejected(self):
        with self.assertRaises(ValueError):
            collate_training_examples([], 999)
        with self.assertRaises(ValueError):
            collate_training_examples([{"input_ids": [1], "labels": [], "attention_mask": [1]}], 999)

    def test_ambiguous_prefix_tokenization_is_rejected(self):
        class BadTokenizer(CharacterTokenizer):
            def encode(self, text, **kwargs):
                values = super().encode(text, **kwargs)
                if "<|im_start|>think" in text:
                    values[0] = 999
                return values
        with self.assertRaisesRegex(ValueError, "boundary"):
            make_training_example(row(), BadTokenizer())


if __name__ == "__main__":
    unittest.main()
