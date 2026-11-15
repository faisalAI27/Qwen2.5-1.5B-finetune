import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/review_first_trial.py"
    spec = importlib.util.spec_from_file_location("first_trial_review_for_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture():
    eligibility = {"train": [{"id": "t1"}, {"id": "t2"}], "development": [{"id": "d1"}, {"id": "d2"}]}
    trained = {"status": "complete", "scope": "development_training", "fresh_process_reload_identical": True,
               "training_examples": 2, "training_cursor": {"examples_committed": 2},
               "development_ids": ["d1", "d2"], "development_examples": 2,
               "development_supervised_tokens": 50, "development_loss": 0.6,
               "initial_adapter_tensor_sha256": "synthetic", "learning_rate": 5e-5}
    manifest = {"development_ids": ["d1", "d2"], "initial_adapter_tensor_sha256": "synthetic"}
    baseline = {"status": "complete", "development_ids": ["d1", "d2"],
                "development_supervised_tokens": 50, "development_loss": 0.8}
    return eligibility, trained, manifest, baseline


class FirstTrialReviewTests(unittest.TestCase):
    def test_original_orchestrator_refuses_new_training_during_review(self):
        path = Path(__file__).resolve().parents[1] / "scripts/run_development.py"
        spec = importlib.util.spec_from_file_location("development_review_guard_for_test", path)
        development = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(development)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "results/runtime"
            runtime.mkdir(parents=True)
            (runtime / "first_trial_review_request.json").write_text(json.dumps({
                "run_id": "study-trial-1", "second_trial_authorized_to_start": False}))
            with patch.object(development, "__file__", str(root / "scripts/run_development.py")), \
                    patch.object(development, "execute") as execute, \
                    patch("sys.argv", ["run_development.py", "--prefix", "study"]):
                with self.assertRaisesRegex(SystemExit, "review is active"):
                    development.main()
                execute.assert_not_called()

    def test_incomplete_trial_and_mismatched_base_denominator_are_rejected(self):
        review = module()
        eligible, trained, manifest, baseline = fixture()
        review.validate_completed_trial(trained, manifest, eligible)
        with self.assertRaises(ValueError):
            review.validate_completed_trial({**trained, "training_cursor": {"examples_committed": 1}}, manifest, eligible)
        with self.assertRaises(ValueError):
            review.compare_validation(trained, {**baseline, "development_supervised_tokens": 49})
        with self.assertRaises(ValueError):
            review.compare_validation(trained, {**baseline, "development_ids": ["d2", "d1"]})
        self.assertAlmostEqual(review.compare_validation(trained, baseline)["relative_nll_reduction"], 0.25)

    def test_review_runs_verification_and_read_only_baseline_then_stops_before_more_training(self):
        review = module()
        eligible, trained, manifest, baseline = fixture()
        run_id = "study-trial-1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "results/training" / run_id
            runtime = root / "results/runtime"
            for folder in (source, runtime, root / "manifests", root / "docs", root / "scripts"):
                folder.mkdir(parents=True, exist_ok=True)
            (runtime / "PAUSE").write_text("User review requested")
            (runtime / "first_trial_review_request.json").write_text(json.dumps({"run_id": run_id,
                "second_trial_authorized_to_start": False, "training_worker_pid": 2, "suspended_training_orchestrator_pid": 1}))
            for name, value in (("training_summary.json", trained), ("manifest.json", manifest), ("generation_before_reload.json", {})):
                (source / name).write_text(json.dumps(value))
            (root / "manifests/s1k_eligibility.json").write_text(json.dumps(eligible))
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                if command[command.index("--mode") + 1] == "verify":
                    (source / "summary.json").write_text(json.dumps(trained))
                else:
                    target = root / "results/review" / run_id
                    target.mkdir(parents=True)
                    (target / "base_validation.json").write_text(json.dumps(baseline))
                return SimpleNamespace(returncode=0)
            with patch.object(review, "__file__", str(root / "scripts/review_first_trial.py")), \
                    patch.object(review, "process", side_effect=lambda pid: {"state": "T", "command": "python scripts/run_development.py"} if pid == 1 else None), \
                    patch.object(review.subprocess, "run", side_effect=run), \
                    patch("sys.argv", ["review_first_trial.py", "--run-id", run_id]), patch("builtins.print"):
                review.main()
            self.assertEqual([command[command.index("--mode") + 1] for command in commands], ["verify", "baseline"])
            state = json.loads((runtime / "first_trial_review.json").read_text())
            self.assertEqual(state["status"], "awaiting_user_decision")
            self.assertFalse(state["review"]["second_trial_started"])
            self.assertTrue((runtime / "PAUSE").exists())
            self.assertFalse((root / "manifests/training_freeze.json").exists())
            self.assertFalse((root / "data/raw/math500/test_with_answers.jsonl").exists())
