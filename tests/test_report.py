import importlib.util
from pathlib import Path
import unittest

from s1small.budget_forcing import CONDITIONS


class ReportTests(unittest.TestCase):
    def test_draft_never_claims_full_accuracy_for_partial_data(self):
        path = Path(__file__).resolve().parents[1] / "scripts/report_project.py"
        spec = importlib.util.spec_from_file_location("report_cli_for_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scores = {key: {"accuracy": None, "correct": 0, "resolved_grades": 0, "mean_reasoning_tokens": None,
                        "mean_total_generated_tokens": None, "mean_latency_seconds": None, "extraction_failures": 0}
                  for key in CONDITIONS}
        pairs = {key: {"accuracy_difference": None, "confidence_interval": None, "paired_resolved_denominator": 0,
                       "transitions": dict.fromkeys(("wrong_to_right", "right_to_wrong", "wrong_to_wrong", "right_to_right"), 0)}
                 for key in CONDITIONS if key != "C"}
        report = {"status": "partial", "conditions": scores, "paired_vs_C": pairs, "figures": []}
        row = {"run_id": "synthetic", "training_examples": 4,
               "training_cursor": {"training_seconds_committed": 10, "input_tokens_committed": 100}, "peak_mlx_active_bytes": 100}
        stage6 = {"selected_learning_rate": 1e-4, "trials": [], "final_training": row}
        freeze = {"protocol": {"natural_reasoning_ceiling": 8192, "answer_budget": 1536, "absolute_safety_ceiling": 12288},
                  "development_protocol_diagnostics": {"marker_stops": 0}}
        paper = module.make_paper(report, stage6, freeze, "synthetic", "Synthetic hardware fixture")
        self.assertIn("Coverage is incomplete", paper)
        self.assertNotIn("condition achieved", paper)
        self.assertIn("No internal reasoning role-marker stop", paper)
        self.assertIn("Appendix A. Hardware", paper)
        existing = {**stage6, "completion_scope": "existing_adapter_selected_for_evaluation",
                    "evaluation_model": {**row, "training_examples": 896}, "final_training": None,
                    "trials": [{**row, "training_examples": 896}]}
        single_run_paper = module.make_paper(report, existing, freeze, "synthetic", "Synthetic hardware fixture")
        self.assertIn("existing 896-example adapter is used directly", single_run_paper)
        self.assertIn("retraining were deferred by user decision", single_run_paper)
        self.assertNotIn("Two predeclared one-epoch development trials", single_run_paper)
        self.assertNotIn("Final training starts from fresh", single_run_paper)
        self.assertEqual(single_run_paper.count("| synthetic |"), 1)
