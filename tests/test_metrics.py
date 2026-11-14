import unittest

from s1small.metrics import paired_comparison, summarize_condition


def row(key, correct, **kwargs):
    return {"question_id": key, "status": "ok", "is_correct": correct, "extracted_answer": "1",
            "total_generated_tokens": 10, "latency_seconds": 1.0, **kwargs}


class MetricsTests(unittest.TestCase):
    def test_full_denominator_counts_extraction_failure_wrong(self):
        result = summarize_condition([row("a", True), row("b", False, extracted_answer=None)], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["extraction_failure_rate"], 0.5)
        self.assertIsNone(result["mean_reasoning_tokens"])

    def test_partial_outputs_are_never_presented_as_full_accuracy(self):
        records = [row("a", True), {"question_id": "b", "status": "infrastructure_failure", "is_correct": None}]
        result = summarize_condition(records, 2)
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["resolved_accuracy"], 1.0)
        self.assertEqual(result["confirmed_correct_fraction_of_planned"], 0.5)
        self.assertFalse(result["complete"])

    def test_paired_bootstrap_resamples_same_question_and_reports_transitions(self):
        left = [row("a", False), row("b", True), row("c", False), row("d", True)]
        right = [row("a", True), row("b", False), row("c", False), row("d", True)]
        result = paired_comparison(left, right, iterations=1000)
        self.assertEqual(result["accuracy_difference"], 0.0)
        self.assertEqual(set(result["transitions"].values()), {1})
        self.assertEqual(result, paired_comparison(left, list(reversed(right)), iterations=1000))
        same = paired_comparison(left, left, iterations=1000)
        self.assertEqual(same["confidence_interval"], [0.0, 0.0])

    def test_unresolved_pair_denominator_is_explicit_and_ids_must_match(self):
        left = [row("a", True), row("b", None)]
        right = [row("a", False), row("b", True)]
        result = paired_comparison(left, right, iterations=100)
        self.assertEqual(result["paired_resolved_denominator"], 1)
        self.assertEqual(result["excluded_unresolved_ids"], ["b"])
        with self.assertRaises(ValueError):
            paired_comparison(left, right[:1])
