import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from s1small.data import digest, file_sha256, save_json
from s1small.budget_forcing import CONDITIONS


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ColabHandoffTests(unittest.TestCase):
    def test_asset_export_excludes_code_environments_gold_and_live_mac_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ["README.md", "pyproject.toml", "docs/colab_inference.md", "docs/license_metadata.json",
                "docs/hardware.md", "notebooks/colab_inference.ipynb", "manifests/data_provenance.json",
                "data/raw/s1k/data/train-00000-of-00001.parquet", "results/training/study/manifest.json",
                "results/training/study/summary.json", "scripts/evaluate.py", "artifacts/base/model.safetensors"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic-only")
            save_json(root / "manifests/training_freeze.json", {"selected_run_id": "study"})
            records = []
            for index in range(5):
                for condition in CONDITIONS:
                    row = {"question_id": str(index), "condition": condition, "status": "ok", "latency_seconds": 1.0,
                           "segments": [{"origin": "model", "token_ids": [1, 2]}]}
                    records.append(row)
                    save_json(root / f"results/protocol_dev/dev/slots/{index}-{condition}.json", row)
            parent = {"scope": "frozen_test_evaluation", "test_used_for_selection": False,
                "files_sha256": {name: file_sha256(root / name) for name in ("scripts/evaluate.py", "artifacts/base/model.safetensors")},
                "development_run_id": "dev", "development_question_ids": [str(i) for i in range(5)],
                "planned_conditions": list(CONDITIONS), "development_outputs_sha256": digest(records)}
            save_json(root / "manifests/evaluation_freeze.json", parent)
            for name in (".venv/secret", "data/raw/math500/test_with_answers.jsonl", "results/test/mac/answer.json"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("must-not-transfer")
            destination = root / "export.zip"
            result = script("build_colab_bundle.py").build(root, destination)
            self.assertFalse(result["cuda_tested"])
            with zipfile.ZipFile(destination) as archive:
                paths = set(archive.namelist())
                self.assertIn("artifacts/base/model.safetensors", paths)
                self.assertIn("manifests/evaluation_freeze_macos.json", paths)
                self.assertNotIn("manifests/evaluation_freeze.json", paths)
                self.assertNotIn("scripts/evaluate.py", paths)
                self.assertFalse(any("test_with_answers" in p or ".venv" in p or "results/test/mac" in p for p in paths))
                manifest = json.loads(archive.read("colab_bundle_manifest.json"))
                self.assertIn("scripts/evaluate.py", manifest["files_sha256"])
            self.assertEqual(json.loads((root / "manifests/evaluation_freeze.json").read_text()), parent)

    def test_checkpoint_preserves_newer_local_outputs_and_rejects_conflicts(self):
        checkpoint = script("colab_checkpoint.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir()
            (root / "colab_bundle_manifest.json").write_text("{}")
            slot = root / "results/test/cuda/slots/example.json"
            slot.parent.mkdir(parents=True)
            initial = {"manifest_id": "same", "status": "not_attempted", "attempt": 0}
            slot.write_text(json.dumps(initial))
            drive = Path(temporary) / "drive"
            checkpoint.checkpoint(root, drive)
            completed = {"manifest_id": "same", "status": "ok", "attempt": 1, "answer": "13"}
            slot.write_text(json.dumps(completed))
            checkpoint.restore(root, drive)
            self.assertEqual(json.loads(slot.read_text()), completed)
            checkpoint.checkpoint(root, drive)
            slot.write_text(json.dumps({**completed, "answer": "different"}))
            with self.assertRaisesRegex(ValueError, "Conflicting completed"):
                checkpoint.restore(root, drive)
            self.assertFalse(checkpoint.state_path("../source.py"))
            self.assertFalse(checkpoint.state_path("src/s1small/data.py"))

    def test_colab_runner_refuses_active_mac_workspace(self):
        runner = script("colab_inference.py")
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "__file__", str(Path(temporary) / "scripts/colab_inference.py")), patch("sys.argv", ["colab_inference.py", "verify"]):
            with self.assertRaisesRegex(ValueError, "never the active Mac"):
                runner.main()

    def test_notebook_code_cells_parse_without_running_cloud_actions(self):
        import ast
        notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/colab_inference.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])

    def test_cuda_runtime_checks_documented_hardware_before_loading_a_model(self):
        import s1small.colab_cuda as cuda
        versions = {"mlx": "0.32.4", "mlx-cuda-12": "0.32.4", "mlx-lm": "0.32.2",
                    "transformers": "5.10.1", "numpy": "2.4.3"}
        cuda.runtime_signature.cache_clear()
        with patch.object(cuda.platform, "system", return_value="Linux"), \
                patch.object(cuda.importlib.metadata, "version", side_effect=lambda name: versions[name]), \
                patch.object(cuda.subprocess, "check_output", return_value="Tesla T4, 550.54.15, 15109, 7.5\n"), \
                patch.object(cuda, "file_sha256", return_value="synthetic"):
            self.assertEqual(cuda.runtime_signature()["backend"], "mlx_cuda12")
            cuda.runtime_signature.cache_clear()
            with patch.object(cuda.subprocess, "check_output", return_value="Older GPU, 550.54.15, 15109, 7.0\n"):
                with self.assertRaisesRegex(RuntimeError, "SM >= 7.5"):
                    cuda.runtime_signature()
        cuda.runtime_signature.cache_clear()
