import copy
import unittest

from s1small.baselines import select_development, validate_config
from s1small.data import identify_rows
from s1small.extraction import extract_answer
from s1small.generate import build_prompt, repeat_comparison, termination_reason


def configuration():
    return {"scope": "development_smoke_only", "sample_count": 5, "repeats": 2,
            "max_new_tokens": 512, "max_input_tokens": 1024, "timeout_seconds": 180,
            "mps_memory_fraction": 0.9, "do_sample": False, "num_beams": 1, "use_cache": True}


class BaselineTests(unittest.TestCase):
    def test_development_only_config_rejects_test_or_large_runs(self):
        validate_config(configuration())
        for key, value in [("scope", "test"), ("sample_count", 500), ("repeats", 0),
                           ("do_sample", True), ("num_beams", 2), ("max_new_tokens", 0),
                           ("mps_memory_fraction", 1.5), ("timeout_seconds", 0)]:
            config = configuration()
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)

    def test_selection_preserves_development_order_and_content(self):
        questions = ["Question zero?", "Question one?", "Question two?"]
        rows = identify_rows([{"question": q} for q in questions])
        split = {"train": [rows[0]], "development": [rows[2], rows[1]]}
        selected = select_development(split, split, questions, 2)
        self.assertEqual([r["question"] for r in selected], [questions[2], questions[1]])
        invalid = copy.deepcopy(split)
        invalid["development"][0] = rows[0]
        with self.assertRaises(ValueError):
            select_development(invalid, split, questions, 1)
        with self.assertRaises(ValueError):
            select_development(split, split, ["Changed", "Changed", "Changed"], 2)

    def test_prompts_share_question_and_system_but_have_distinct_instructions(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return str(messages)
        tokenizer = Tokenizer()
        templates = {"system": "Shared system", "conditions": {"A": "Only answer", "B": "Explain"}}
        first = build_prompt("Same question?", "A", templates, tokenizer)
        second = build_prompt("Same question?", "B", templates, tokenizer)
        self.assertNotEqual(first, second)
        self.assertIn("Same question?", first)
        self.assertIn("Same question?", second)
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

    def test_nested_latex_box(self):
        result = extract_answer(r"Final answer: \boxed{\frac{1}{2}}")
        self.assertEqual(result["extracted_answer"], r"\frac{1}{2}")
        self.assertEqual(result["extraction_status"], "boxed")

    def test_last_box_used_and_escaped_braces_preserved(self):
        self.assertEqual(extract_answer(r"\boxed{1} then \boxed{\{2,3\}} ")["extracted_answer"], r"\{2,3\}")

    def test_partial_or_empty_box_is_not_a_success(self):
        self.assertEqual(extract_answer(r"\boxed{\frac{1}{2}")["extraction_status"], "unclosed_box")
        self.assertEqual(extract_answer(r"\boxed{}")["extraction_status"], "empty_answer")

    def test_unfinished_reasoning_does_not_become_a_numeric_answer(self):
        self.assertIsNone(extract_answer("We calculate the intermediate value 42.")["extracted_answer"])

    def test_final_answer_line_fallback(self):
        result = extract_answer("Some reasoning.\nFinal answer: 17\n")
        self.assertEqual(result["extracted_answer"], "17")
        self.assertEqual(result["extraction_status"], "final_answer_line")

    def test_explicit_final_line_overrides_earlier_intermediate_box(self):
        self.assertEqual(extract_answer("Intermediate \\boxed{2}\nFinal answer: 3")["extracted_answer"], "3")
        self.assertIsNone(extract_answer("Intermediate \\boxed{2}\nFinal answer: ")["extracted_answer"])

    def test_termination_classification_handles_caps_and_timeouts(self):
        self.assertEqual(termination_reason([1, 2], [2], 2, True), "eos")
        self.assertEqual(termination_reason([1, 3], [2], 2, False), "max_new_tokens")
        self.assertEqual(termination_reason([1], [2], 10, True), "timeout")
        self.assertEqual(termination_reason([], [2], 10, False), "unexpected_early_stop")

    def test_repeat_check_compares_tokens_and_does_not_hide_errors(self):
        rows = [{"question_id": "q", "condition": "A", "status": "ok", "generated_token_ids": [1, 2]},
                {"question_id": "q", "condition": "A", "status": "ok", "generated_token_ids": [1, 2]}]
        self.assertTrue(repeat_comparison(rows)[0]["token_ids_identical"])
        rows[1]["generated_token_ids"] = [1, 3]
        self.assertFalse(repeat_comparison(rows)[0]["token_ids_identical"])
        rows[1]["status"] = "error"
        self.assertFalse(repeat_comparison(rows)[0]["comparable"])


if __name__ == "__main__":
    unittest.main()
