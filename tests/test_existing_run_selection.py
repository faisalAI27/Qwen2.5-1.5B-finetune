import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from s1small.data import file_sha256, save_json


class ExistingRunSelectionTests(unittest.TestCase):
    def fixture(self, root):
        run_id = "study-trial-1"
        source = root / "results/training" / run_id
        source.mkdir(parents=True)
        eligibility = {"train": [{"id": "t1"}, {"id": "t2"}], "development": [{"id": "d1"}, {"id": "d2"}],
                       "final_training": [{"id": value} for value in ("t1", "t2", "d1", "d2")]}
        save_json(root / "manifests/s1k_eligibility.json", eligibility)
        (root / "configs").mkdir()
        (root / "configs/train_development.yaml").write_text("original two-trial declaration\n")
        summary = {"status": "complete", "scope": "development_training", "run_id": run_id,
            "fresh_process_reload_identical": True, "training_examples": 2,
            "training_cursor": {"examples_committed": 2}, "development_ids": ["d1", "d2"],
            "development_examples": 2, "development_supervised_tokens": 50, "development_loss": 0.6,
            "initial_adapter_tensor_sha256": "initial", "learning_rate": 5e-5,
            "adapter_directory": f"artifacts/training/{run_id}/adapter"}
        manifest = {"development_ids": ["d1", "d2"], "initial_adapter_tensor_sha256": "initial",
            "recipe": {"learning_rate": 5e-5}, "recipe_sha256": "recipe",
            "identity": {"eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json"),
                         "code_sha256": {}, "conversion_provenance_sha256": "conversion"}}
        save_json(source / "summary.json", summary)
        save_json(source / "manifest.json", manifest)
        baseline_path = root / "results/review" / run_id / "base_validation.json"
        save_json(baseline_path, {"status": "complete", "development_ids": ["d1", "d2"],
            "development_supervised_tokens": 50, "development_loss": 0.8})
        save_json(root / "docs/first_trial_review.json", {"run_id": run_id,
            "source_hashes": {"trial_summary": file_sha256(source / "summary.json"), "base_validation": file_sha256(baseline_path)}})
        save_json(root / "configs/single_run_decision.json", {"strategy": "evaluate_existing_adapter", "study_prefix": "study",
            "selected_run_id": run_id, "additional_training_authorized": False, "test_used_for_decision": False,
            "review_sha256": file_sha256(root / "docs/first_trial_review.json")})
        checkpoint = root / "artifacts/training" / run_id / "checkpoints/step-000001"
        adapter = root / summary["adapter_directory"]
        checkpoint.mkdir(parents=True)
        adapter.mkdir(parents=True)
        for name in ("adapters.safetensors", "adapter_config.json", "optimizer.safetensors"):
            (checkpoint / name).write_bytes(b"synthetic-only")
            if name != "optimizer.safetensors":
                (adapter / name).write_bytes(b"synthetic-only")
        (adapter / "source_provenance.json").write_text("{}")
        save_json(checkpoint / "checkpoint.json", {"examples_committed": 2, "recipe_sha256": "recipe", "files": [
            {"name": path.name, "sha256": file_sha256(path)} for path in checkpoint.iterdir()]})
        save_json(checkpoint.parent / "latest.json", {"directory": checkpoint.name,
            "checkpoint_sha256": file_sha256(checkpoint / "checkpoint.json")})
        return run_id

    def module(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location("existing_run_selection_for_test", scripts / "freeze_existing_run.py")
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec.loader.exec_module(module)
        return module

    def test_existing_adapter_is_selected_without_claiming_or_starting_more_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = self.fixture(root)
            result = self.module().prepare(root)
            self.assertIsNone(result["final_training"])
            self.assertEqual(result["evaluation_model"]["run_id"], run_id)
            self.assertFalse(result["training_deviation"]["hyperparameter_search_performed"])
            self.assertFalse(result["training_deviation"]["additional_training_authorized"])
            self.assertEqual(self.module().prepare(root), result)
            self.assertFalse((root / "artifacts/training/study-final").exists())
            self.assertFalse((root / "results/training/study-trial-2").exists())
            self.assertFalse((root / "manifests/evaluation_freeze.json").exists())

    def test_changed_export_or_prior_benchmark_access_blocks_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = self.fixture(root)
            path = root / "artifacts/training" / run_id / "adapter/adapters.safetensors"
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Exported adapter"):
                self.module().prepare(root)
            path.write_bytes(b"synthetic-only")
            gold = root / "data/raw/math500/test_with_answers.jsonl"
            gold.parent.mkdir(parents=True)
            gold.write_text("synthetic inaccessible benchmark placeholder")
            with self.assertRaisesRegex(ValueError, "benchmark access"):
                self.module().prepare(root)
