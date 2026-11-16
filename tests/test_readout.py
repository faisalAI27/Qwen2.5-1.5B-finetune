import unittest

from s1small.readout import extract_readout


class ReadoutTests(unittest.TestCase):
    def test_declared_label_does_not_create_an_answer_without_model_text(self):
        for body in ("", "\n  "):
            result = extract_readout(body, "Final Answer: ")
            self.assertIsNone(result["extracted_answer"])
        self.assertIsNone(extract_readout("13")["extracted_answer"])

    def test_contextual_extraction_preserves_control_origin_and_generated_value(self):
        result = extract_readout("13", "Final Answer: ")
        self.assertEqual(result["extracted_answer"], "13")
        self.assertEqual(result["injected_readout_prefix"], "Final Answer: ")
        self.assertEqual(result["extraction_origin"], "declared_common_label_plus_generated_answer")
        self.assertEqual(result["extraction_input"], "Final Answer: 13")
        self.assertIsNone(extract_readout(r"13 then \boxed{unfinished", "Final Answer: ")["extracted_answer"])
