import importlib.util
from pathlib import Path
import unittest


class TimedCheckTests(unittest.TestCase):
    def test_continuation_policy_preserves_pause_failures_and_existing_jobs(self):
        path = Path(__file__).resolve().parents[1] / "scripts/check_pipeline_later.py"
        spec = importlib.util.spec_from_file_location("timed_check_for_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        choose = module.next_action
        self.assertEqual(choose("running", True, True, False, True), "workflow_already_continuing")
        self.assertEqual(choose("running", False, True, False, True), "wait_for_existing_worker")
        self.assertEqual(choose("running", False, False, True, True), "honor_pause")
        self.assertEqual(choose("running", False, False, False, False), "configuration_changed_no_restart")
        self.assertEqual(choose("failed", False, False, False, True), "failure_requires_investigation")
        self.assertEqual(choose("complete", False, False, False, True), "handoff_finished")
        self.assertEqual(choose("running", False, False, False, True), "resume_same_evaluation_pipeline_once")
