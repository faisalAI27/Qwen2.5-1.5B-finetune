import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from s1small.data import (apply_eligibility, digest, find_overlaps, identify_rows, make_splits,
                          normalize_question, validate_s1k, write_immutable)
from s1small.formatting import assistant_parts, format_example, measure_example
from s1small.lengths import percentile, summarize_lengths


def example(index=0):
    return {"question": f"What is {index} plus 2?", "thinking_trajectories": ["Add two.", "Check."], "attempt": "The result is 3."}


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class DataTests(unittest.TestCase):
    def test_benchmark_projection_never_persists_gold(self):
        script = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
        spec = importlib.util.spec_from_file_location("prepare_data", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = {"unique_id": "test/1", "problem": "A synthetic question?", "answer": "SECRET", "solution": "SECRET"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO((json.dumps(source) + "\n").encode())):
                module.project_benchmark("https://example.invalid", path, 1)
            saved = path.read_text()
            self.assertNotIn("SECRET", saved)
            self.assertEqual(json.loads(saved), {"id": "test/1", "question": "A synthetic question?"})

    def test_schema_valid(self):
        self.assertEqual(validate_s1k([example()], 1)["problems"], [])

    def test_missing_empty_and_wrong_type_fields(self):
        for field, value in [("question", ""), ("attempt", None), ("thinking_trajectories", "text"), ("thinking_trajectories", [""])]:
            row = example()
            row[field] = value
            with self.subTest(field=field, value=value):
                self.assertTrue(validate_s1k([row], 1)["problems"])

    def test_wrong_count_reported(self):
        self.assertTrue(validate_s1k([example()], 1000)["problems"])

    def test_duplicate_questions_reported(self):
        report = validate_s1k([example(), example()], 2)
        self.assertEqual(report["duplicate_question_groups"], 1)

    def test_stable_unique_ids_and_content_changes(self):
        records = identify_rows([example(i) for i in range(1000)])
        self.assertEqual(len({r["id"] for r in records}), 1000)
        self.assertEqual(records, identify_rows([example(i) for i in range(1000)]))
        changed = example()
        changed["attempt"] = "Changed answer"
        self.assertNotEqual(identify_rows([example()])[0]["row_sha256"], identify_rows([changed])[0]["row_sha256"])

    def test_fixed_900_100_split_and_order_independence(self):
        records = identify_rows([example(i) for i in range(1000)])
        split = make_splits(records, "revision")
        self.assertEqual(len(split["train"]), 900)
        self.assertEqual(len(split["development"]), 100)
        self.assertFalse({r["id"] for r in split["train"]} & {r["id"] for r in split["development"]})
        self.assertEqual(split, make_splits(list(reversed(records)), "revision"))
        self.assertNotEqual(split["development"], make_splits(records, "revision", seed=43)["development"])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            make_splits([{"id": "same"}, {"id": "same"}], "revision", development_size=1)

    def test_duplicate_questions_cannot_cross_partitions(self):
        with self.assertRaisesRegex(ValueError, "cross train/development"):
            make_splits([{"id": "a", "question_sha256": "same"}, {"id": "b", "question_sha256": "same"}],
                        "revision", development_size=1)

    def test_exclusions_preserve_raw_split_and_do_not_refill_partitions(self):
        records = identify_rows([example(i) for i in range(10)])
        splits = make_splits(records, "revision", development_size=2)
        excluded = splits["train"][0]["id"]
        eligible = apply_eligibility(splits, [{"s1k_id": excluded, "reason": "test overlap"}])
        self.assertEqual(len(splits["train"]), 8)
        self.assertEqual(len(eligible["train"]), 7)
        self.assertEqual(len(eligible["development"]), 2)
        self.assertEqual(len(eligible["final_training"]), 9)
        self.assertNotIn(excluded, {row["id"] for row in eligible["final_training"]})
        with self.assertRaises(ValueError):
            apply_eligibility(splits, [{"s1k_id": "unknown"}])

    def test_immutable_outputs_allow_identical_rerun_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_immutable(path, b"original")
            write_immutable(path, b"original")
            with self.assertRaises(ValueError):
                write_immutable(path, b"changed")
            self.assertEqual(path.read_bytes(), b"original")

    def test_normalization_preserves_mathematical_differences(self):
        self.assertEqual(normalize_question("  WHAT\n is ２? "), "what is 2?")
        self.assertNotEqual(normalize_question("x+1"), normalize_question("x-1"))
        self.assertNotEqual(digest("12"), digest("21"))

    def test_exact_near_and_unrelated_question_matches(self):
        train = [{"id": "a", "question": "Compute the sum of the first ten positive integers."}]
        benchmark = [{"id": "exact", "question": "COMPUTE the sum of the first ten positive integers."},
                     {"id": "near", "question": "Compute the sum of the first eleven positive integers."},
                     {"id": "other", "question": "Find a triangle angle given its perimeter and area."}]
        report = find_overlaps(train, benchmark)
        self.assertEqual(report["exact"], [{"s1k_id": "a", "benchmark_id": "exact"}])
        self.assertEqual([r["benchmark_id"] for r in report["near_candidates"]], ["near"])
        self.assertEqual(report["exclusions_applied"], [])

    def test_short_near_matches_suppressed(self):
        report = find_overlaps([{"id": "a", "question": "1 + 2"}], [{"id": "b", "question": "1 + 3"}])
        self.assertEqual(report["near_candidates"], [])

    def test_official_answer_prefix_and_trace_join(self):
        reasoning, answer = assistant_parts(example())
        self.assertEqual(reasoning, "Add two.\nCheck.")
        self.assertEqual(answer, "Answer: The result is 3.")
        row = example()
        row["attempt"] = "  Answer: 3  "
        self.assertEqual(assistant_parts(row)[1], "Answer: 3")

    def test_format_preserves_complete_answer_and_end_marker(self):
        text = format_example(example(), CharacterTokenizer())
        self.assertIn("<|im_start|>think\nAdd two.\nCheck.\n<|im_start|>answer\nAnswer: The result is 3.", text)
        self.assertTrue(text.endswith("<|im_end|>\n"))

    def test_single_sequence_length_accounting(self):
        tokenizer = CharacterTokenizer()
        row = example()
        lengths = measure_example(row, tokenizer)
        self.assertEqual(lengths["total"], len(format_example(row, tokenizer)))
        self.assertEqual(lengths["reasoning"], len("Add two.\nCheck."))
        self.assertEqual(lengths["total"], sum(lengths[k] for k in ("prompt", "reasoning", "answer", "structure")))

    def test_exact_context_limit_is_eligible_and_histogram_adds_up(self):
        rows = [{"total": n, "prompt": 1, "reasoning": n-3, "answer": 1, "structure": 1, "boundary_crossing_tokens": 0} for n in [10, 20, 30]]
        summary = summarize_lengths(rows, [10, 20])
        self.assertEqual(summary["context_limits"][0]["fits_complete_examples"], 1)
        self.assertEqual(summary["context_limits"][1]["exceeds"], 1)
        self.assertEqual(sum(b["count"] for b in summary["histogram"]), 3)
        self.assertEqual(percentile([10, 20, 30], 0.95), 29)


if __name__ == "__main__":
    unittest.main()
