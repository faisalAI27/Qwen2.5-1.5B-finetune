import unittest

from s1small.budget_forcing import Budget, CONDITIONS, Protocol, Segment, condition_budget, run_structured


class Decoder:
    def __init__(self, programs):
        self.programs = list(programs)
        self.calls = []
    def decode(self, tokens):
        mapping = {10: "\n", 90: "<start>", 91: "answer", 87: "Wait", 99: "<end>"}
        return "".join(mapping.get(t, str(t)) for t in tokens)
    def generate(self, context, maximum, stops, deadline):
        self.calls.append((context, maximum))
        item = self.programs.pop(0)
        if isinstance(item, Segment):
            return item
        result = []
        for token in item[:maximum]:
            result.append(token)
            if token in stops:
                break
        return Segment(tuple(result))


class BudgetForcingTests(unittest.TestCase):
    protocol = Protocol(90, (99,), (90, 91, 10), (10,), (10, 87))
    def run_case(self, programs, condition="C", reasoning=10, answer=4, absolute=50, waits=0):
        decoder = Decoder(programs)
        result = run_structured(decoder, [1, 10], Budget(condition, reasoning, answer, absolute, waits), self.protocol, 10)
        self.assertEqual(result["total_generated_tokens"], sum(len(e["token_ids"]) for e in result["segments"] if e["origin"] == "model"))
        self.assertEqual(result["injected_tokens"], sum(len(e["token_ids"]) for e in result["segments"] if e["origin"] == "injected"))
        self.assertLessEqual(result["total_new_tokens_consumed"], absolute)
        return result, decoder

    def test_natural_stop_and_common_answer_readout(self):
        result, decoder = self.run_case([[11, 12, 90], [31, 99]])
        self.assertEqual(result["reasoning_tokens"], 2)
        self.assertEqual(result["reasoning_phase_generated_tokens"], 3)
        self.assertEqual(result["forced_wait_count"], 0)
        self.assertEqual(result["answer_text"], "31")
        self.assertFalse(result["reasoning_cap_hit"])
        self.assertEqual(decoder.calls[1][0][-4:], [10, 90, 91, 10])

    def test_cap_transitions_to_answer_instead_of_cutting_all_generation(self):
        result, decoder = self.run_case([[11, 12, 13, 14], [31, 99]], "D256", reasoning=2)
        self.assertEqual(result["reasoning_tokens"], 2)
        self.assertTrue(result["reasoning_cap_hit"])
        self.assertFalse(result["safety_cap_hit"])
        self.assertEqual(result["answer_text"], "31")
        self.assertEqual(len(decoder.calls), 2)

    def test_one_and_two_waits_require_actual_stop_events(self):
        for waits in (1, 2):
            with self.subTest(waits=waits):
                result, _ = self.run_case([[11, 90]] * (waits + 1) + [[31, 99]], f"E{waits}", waits=waits)
                self.assertEqual(result["forced_wait_count"], waits)
                self.assertEqual(result["reasoning_tokens"], waits + 1)
                self.assertEqual(sum(e["phase"] == "wait" for e in result["segments"]), waits)
                self.assertTrue(all(e["removed_stop_token_id"] == 90 for e in result["segments"] if e["phase"] == "reasoning"))

    def test_no_stop_no_wait_and_natural_safety_cap_is_recorded(self):
        result, _ = self.run_case([[11] * 20, [31, 99]], "E2", reasoning=5, waits=2)
        self.assertEqual(result["forced_wait_count"], 0)
        self.assertTrue(result["safety_cap_hit"])
        self.assertEqual(result["answer_text"], "31")

    def test_early_eos_is_visible_and_can_trigger_a_declared_continuation(self):
        result, _ = self.run_case([[99], [11, 90], [31, 99]], "E1", waits=1)
        self.assertTrue(result["early_reasoning_eos"])
        self.assertEqual(result["forced_wait_count"], 1)

    def test_zero_content_stop_segments_are_bounded(self):
        result, decoder = self.run_case([[90], [90], [90], [99]], "E2", waits=2)
        self.assertEqual(result["reasoning_tokens"], 0)
        self.assertEqual(result["forced_wait_count"], 2)
        self.assertEqual(len(decoder.calls), 4)
        self.assertEqual(result["answer_text"], "")

    def test_answer_overflow_and_deadline_are_explicit(self):
        result, _ = self.run_case([[11, 90], [31, 32, 33, 34]], answer=2)
        self.assertTrue(result["answer_cap_hit"])
        result, decoder = self.run_case([Segment((11,), timed_out=True)])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["termination_reason"], "timeout_before_answer")
        self.assertEqual(len(decoder.calls), 1)

    def test_remaining_budget_denies_wait_but_preserves_answer_allowance(self):
        result, decoder = self.run_case([[90], [31, 99]], "E2", reasoning=10, answer=2, absolute=8, waits=2)
        self.assertEqual(result["forced_wait_count"], 0)
        self.assertTrue(result["wait_denied_exhausted_budget"])
        self.assertEqual(decoder.calls[-1][1], 2)

    def test_no_reasoning_room_forces_readout_without_looping(self):
        result, decoder = self.run_case([[31, 99]], reasoning=0)
        self.assertTrue(result["reasoning_cap_hit"])
        self.assertEqual(result["answer_text"], "31")
        self.assertEqual(len(decoder.calls), 1)

    def test_stop_at_exact_cap_is_natural_and_no_wait_after_exhaustion(self):
        result, _ = self.run_case([[11, 90], [31, 99]], "E1", reasoning=2, waits=1)
        self.assertEqual(result["reasoning_termination"], "reasoning_marker")
        self.assertFalse(result["reasoning_cap_hit"])
        self.assertTrue(result["wait_denied_exhausted_budget"])

    def test_empty_or_out_of_contract_decoder_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "empty"):
            self.run_case([[]])
        with self.assertRaisesRegex(ValueError, "contract"):
            self.run_case([Segment((11, 90, 12))])

    def test_all_nine_conditions_and_fixed_caps(self):
        self.assertEqual(len(CONDITIONS), 9)
        specs = {name: condition_budget(name, 8192, 1536, 12288) for name in CONDITIONS}
        self.assertIsNone(specs["A"])
        self.assertIsNone(specs["B"])
        self.assertEqual([specs[f"D{n}"].reasoning for n in (256, 512, 1024, 2048)], [256, 512, 1024, 2048])
        self.assertEqual((specs["E1"].waits, specs["E2"].waits), (1, 2))
