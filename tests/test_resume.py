import json
from pathlib import Path
import tempfile
import unittest

from s1small.data import file_sha256
from s1small.resume import prepare_training_resume


class ResumeTests(unittest.TestCase):
    def test_verified_newest_checkpoint_and_partial_files_are_recovered_without_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = "study-trial-1"
            metadata = root / "results/training" / run
            metadata.mkdir(parents=True)
            (metadata / "manifest.json").write_text(json.dumps({"recipe_sha256": "recipe"}))
            checkpoints = root / "artifacts/training" / run / "checkpoints"
            for step in (1, 2):
                folder = checkpoints / f"step-{step:06d}"
                folder.mkdir(parents=True)
                (folder / "weights.bin").write_bytes(b"synthetic-checkpoint")
                (folder / "checkpoint.json").write_text(json.dumps({"recipe_sha256": "recipe", "completed_steps": step,
                    "files": [{"name": "weights.bin", "sha256": file_sha256(folder / "weights.bin")}] }))
            partial = checkpoints / "step-000003-incomplete"
            partial.mkdir()
            (partial / "partial.bin").write_bytes(b"retain-me")
            events = prepare_training_resume(root, "study")
            self.assertEqual(json.loads((checkpoints / "latest.json").read_text())["directory"], "step-000002")
            self.assertTrue(any(row["action"] == "archived_without_deleting" for row in events))
            retained = next((root / "artifacts/training" / run / "interrupted_checkpoints").iterdir())
            self.assertEqual((retained / "partial.bin").read_bytes(), b"retain-me")
            self.assertTrue((checkpoints / "step-000001").exists())
            self.assertEqual(prepare_training_resume(root, "study"), [])
