import importlib.util
from pathlib import Path
import unittest

from s1small.budget_forcing import Protocol


class ReadoutDiagnosticTests(unittest.TestCase):
    def test_prefix_probes_preserve_original_protocol_and_label_injected_text(self):
        path = Path(__file__).resolve().parents[1] / "scripts/diagnose_readout.py"
        spec = importlib.util.spec_from_file_location("readout_diagnostic_for_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [ord(character) for character in text]
        original = Protocol(90, (99,), (90, 91, 10), (10,), (10, 87))
        choices = module.variants(Tokenizer(), original)
        self.assertEqual(len(choices), 3)
        self.assertEqual(choices[0][2], original)
        self.assertEqual(original.answer_header, (90, 91, 10))
        for _, prefix, variant in choices:
            self.assertEqual(variant.answer_header, original.answer_header + tuple(ord(c) for c in prefix))
            self.assertEqual(variant.reasoning_end, original.reasoning_end)
            self.assertEqual(variant.wait, original.wait)
