import importlib.util
from pathlib import Path
import tempfile
import unittest

from s1small.budget_forcing import CONDITIONS


class AnalysisTests(unittest.TestCase):
    def test_raw_record_analysis_and_plot_export_use_all_conditions(self):
        if importlib.util.find_spec("matplotlib") is None or importlib.util.find_spec("numpy") is None:
            self.skipTest("Optional reporting dependencies absent")
        path = Path(__file__).resolve().parents[1] / "scripts/analyze_results.py"
        spec = importlib.util.spec_from_file_location("analysis_cli_for_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = []
        for condition in CONDITIONS:
            for index in range(4):
                records.append({"condition": condition, "question_id": f"synthetic-{index}", "status": "ok",
                                "is_correct": index % 2 == 0, "extracted_answer": "1", "gold_answer": "1",
                                "total_generated_tokens": 10 + index, "latency_seconds": 1.0 + index,
                                "reasoning_tokens": None if condition in ("A", "B") else 5 + index})
        settings = {"bootstrap_seed": 42, "bootstrap_iterations": 1000, "confidence_level": 0.95}
        report = module.analyze(records, 4, settings)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["paired_vs_C"]), 8)
        self.assertEqual(report["conditions"]["C"]["accuracy"], 0.5)
        self.assertTrue(all(row["accuracy_difference"] == 0 for row in report["paired_vs_C"].values()))
        with tempfile.TemporaryDirectory() as temporary:
            files = module.plot_results(report, Path(temporary))
            self.assertEqual(len(files), 10)
            self.assertTrue(all((Path(temporary) / name).stat().st_size > 1000 for name in files))
