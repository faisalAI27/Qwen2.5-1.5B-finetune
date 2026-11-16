import unittest

from s1small.budget_forcing import CONDITIONS
from s1small.protocol_audit import audit_development, require_readout


class ProtocolAuditTests(unittest.TestCase):
    def records(self):
        return [{"condition": condition, "question_id": f"synthetic-{index}", "status": "ok",
                 "answer_text": "unfinished model text", "extracted_answer": "1" if condition == "A" else None}
                for index in range(5) for condition in CONDITIONS]

    def test_direct_answers_alone_do_not_validate_structured_readout(self):
        audit = audit_development(self.records())
        self.assertEqual(audit["completed_outputs"], 45)
        self.assertEqual(audit["conditions"]["A"]["extracted_answers"], 5)
        self.assertEqual(audit["structured_extracted_answers"], 0)
        with self.assertRaisesRegex(ValueError, "before freezing"):
            require_readout(audit)

    def test_observed_readout_is_a_compatibility_check_without_correctness_scoring(self):
        rows = self.records()
        next(row for row in rows if row["condition"] == "C")["extracted_answer"] = "synthetic-answer"
        audit = audit_development(rows)
        require_readout(audit)
        self.assertEqual(audit["status"], "readout_observed")
        self.assertNotIn("accuracy", audit)
        rows[-1]["status"] = "not_attempted"
        with self.assertRaises(ValueError):
            require_readout(audit_development(rows))
