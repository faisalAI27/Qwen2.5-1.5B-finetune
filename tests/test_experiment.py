import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from s1small.data import digest, file_sha256, save_json
from s1small.experiment import PredictionStore, validate_question_identity, verify_freeze
from s1small.benchmark import sealed_test_rows


class ExperimentTests(unittest.TestCase):
    def test_benchmark_gold_cannot_be_fetched_before_freeze(self):
        with tempfile.TemporaryDirectory() as temporary, patch("urllib.request.urlopen") as network:
            with self.assertRaises(FileNotFoundError):
                sealed_test_rows(temporary)
            network.assert_not_called()
    def test_fixed_slots_include_failures_and_never_reroll_a_model_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = PredictionStore(temporary, [("q", "C"), ("q", "E1")], "frozen")
            self.assertEqual(len(store.records()), 2)
            store.save_attempt("q", "C", {"status": "ok", "answer_text": "wrong", "is_correct": False})
            self.assertFalse(store.pending("q", "C"))
            with self.assertRaisesRegex(ValueError, "cannot be replaced"):
                store.save_attempt("q", "C", {"status": "ok", "answer_text": "right"})
            store.save_attempt("q", "E1", {"status": "infrastructure_failure"})
            self.assertTrue(store.pending("q", "E1"))
            store.save_attempt("q", "E1", {"status": "infrastructure_failure"})
            self.assertFalse(store.pending("q", "E1"))
            self.assertEqual(len(store.records()), 2)
            with self.assertRaises(ValueError):
                PredictionStore(temporary, [("q", "C"), ("q", "E1")], "changed")

    def test_crash_recovery_uses_durable_attempt_instead_of_rerunning(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = PredictionStore(temporary, [("q", "C")], "frozen")
            key = store.key("q", "C")
            record = {"status": "ok", "question_id": "q", "condition": "C", "manifest_id": "frozen", "attempt": 1}
            save_json(Path(temporary) / "attempts" / f"{key}-01.json", record)
            reopened = PredictionStore(temporary, [("q", "C")], "frozen")
            self.assertFalse(reopened.pending("q", "C"))
            self.assertEqual(reopened.records(), [record])

    def test_frozen_hash_and_sealed_question_order_are_enforced(self):
        rows = [{"id": "q1", "question": "x?"}, {"id": "q2", "question": "y?"}]
        manifest = {"rows": [{"id": row["id"], "question_sha256": digest(row["question"])} for row in rows]}
        validate_question_identity(rows, manifest)
        with self.assertRaises(ValueError):
            validate_question_identity(list(reversed(rows)), manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("original")
            freeze = {"scope": "frozen_test_evaluation", "test_used_for_selection": False,
                      "files_sha256": {"config.json": file_sha256(root / "config.json")}}
            verify_freeze(root, freeze)
            (root / "config.json").write_text("changed")
            with self.assertRaises(ValueError):
                verify_freeze(root, freeze)
