import importlib.util
import unittest

from s1small.grading import equivalent, extract_final_answer, strip_math_environment


class ExtractionTests(unittest.TestCase):
    def test_last_explicit_claim_wins_without_guessing_an_intermediate_number(self):
        self.assertEqual(extract_final_answer("Intermediate \\boxed{2}\nAnswer: 3")["extracted_answer"], "3")
        self.assertIsNone(extract_final_answer("We might obtain 42 after more reasoning.")["extracted_answer"])
        self.assertIsNone(extract_final_answer("Earlier \\boxed{2}\nAnswer: ")["extracted_answer"])
        self.assertEqual(extract_final_answer("Answer:\n17")["extracted_answer"], "17")

    def test_last_incomplete_box_never_falls_back_to_earlier_claim(self):
        result = extract_final_answer("Answer: 2\nThen \\boxed{3")
        self.assertIsNone(result["extracted_answer"])
        self.assertEqual(result["extraction_status"], "unclosed_box")
        self.assertEqual(extract_final_answer(r"\boxed{\frac{1}{2}}")["extracted_answer"], r"\frac{1}{2}")

    def test_environment_stripping_preserves_inner_expression(self):
        for value in ("$x$", "$$x$$", r"\(x\)", r"\[x\]"):
            self.assertEqual(strip_math_environment(value), "x")


@unittest.skipIf(importlib.util.find_spec("math_verify") is None, "Separate grader runtime absent")
class EquivalenceTests(unittest.TestCase):
    def test_fractions_radicals_powers_and_polynomial_forms(self):
        for gold, prediction in [(r"\frac{1}{2}", "0.5"), (r"\sqrt{8}", r"2\sqrt{2}"),
                                 ("1000", "10^3"), ("(x+1)^2", "x^2+2x+1")]:
            with self.subTest(gold=gold, prediction=prediction):
                self.assertTrue(equivalent(gold, prediction)["is_correct"])

    def test_tuple_order_sets_intervals_and_sign_are_not_conflated(self):
        for gold, prediction in [("(1,2)", "(2,1)"), ("[0,1]", "(0,1)"), ("-2", "2"), ("1", "1.0001")]:
            with self.subTest(gold=gold, prediction=prediction):
                self.assertFalse(equivalent(gold, prediction)["is_correct"])
        self.assertTrue(equivalent(r"\{1,2\}", r"\{2,1\}")["is_correct"])

    def test_unfinished_or_unparseable_answer_does_not_become_correct(self):
        self.assertFalse(equivalent("42", None)["is_correct"])
        self.assertIsNot(equivalent("42", r"\frac{")["is_correct"], True)
        self.assertTrue(equivalent(r"\text{positive}", r"\text{positive}")["is_correct"])

    def test_distinct_symbols_and_inaccurate_decimals_are_rejected(self):
        self.assertFalse(equivalent("x", "y")["is_correct"])
        self.assertFalse(equivalent(r"\frac{1}{3}", "0.3333")["is_correct"])
        self.assertTrue(equivalent("2", "2+0")["is_correct"])
