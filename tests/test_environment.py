import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from s1small import environment


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda="test" if cuda else None),
        cuda=SimpleNamespace(
            is_available=lambda: cuda, device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(name="Test GPU", total_memory=4096),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_built=lambda: mps, is_available=lambda: mps)),
        mps=SimpleNamespace(recommended_max_memory=lambda: 2048),
    )


class EnvironmentTests(unittest.TestCase):
    def test_missing_torch_is_valid_cpu_environment(self):
        with patch.object(environment.importlib, "import_module", side_effect=ModuleNotFoundError(name="torch")):
            report = environment.inspect_torch([])
        self.assertEqual(report["import_status"], "not_installed")
        self.assertEqual(report["preferred_available_device"], "cpu")

    def test_broken_torch_is_reported(self):
        notes = []
        with patch.object(environment.importlib, "import_module", side_effect=OSError("broken library")):
            report = environment.inspect_torch(notes)
        self.assertEqual(report["import_status"], "error")
        self.assertIn("broken library", notes[0])

    def test_cpu_only_torch(self):
        with patch.object(environment.importlib, "import_module", return_value=fake_torch()):
            report = environment.inspect_torch([])
        self.assertFalse(report["cuda_available"])
        self.assertFalse(report["mps_available"])
        self.assertEqual(report["cuda_devices"], [])

    def test_cuda_name_and_dedicated_vram(self):
        with patch.object(environment.importlib, "import_module", return_value=fake_torch(cuda=True)):
            report = environment.inspect_torch([])
        self.assertEqual(report["preferred_available_device"], "cuda")
        self.assertEqual(report["cuda_devices"][0], {"index": 0, "name": "Test GPU", "total_vram_bytes": 4096})

    def test_mps_memory_is_not_reported_as_dedicated_vram(self):
        with patch.object(environment.importlib, "import_module", return_value=fake_torch(mps=True)):
            report = environment.inspect_torch([])
        self.assertEqual(report["preferred_available_device"], "mps")
        self.assertEqual(report["mps_recommended_working_set_bytes"], 2048)
        self.assertEqual(report["cuda_devices"], [])

    def test_sandbox_hidden_mps_is_explained(self):
        torch = fake_torch(mps=True)
        torch.backends.mps.is_available = lambda: False
        notes = []
        with patch.object(environment.importlib, "import_module", return_value=torch):
            report = environment.inspect_torch(notes)
        self.assertFalse(report["mps_available"])
        self.assertIn("sandbox", notes[0])

    def test_missing_system_command_does_not_crash(self):
        notes = []
        with patch.object(environment.subprocess, "run", side_effect=FileNotFoundError):
            self.assertIsNone(environment.run_command(["missing"], notes))
        self.assertTrue(notes)

    def test_command_timeout_does_not_crash(self):
        with patch.object(environment.subprocess, "run", side_effect=subprocess.TimeoutExpired("test", 20)):
            self.assertIsNone(environment.run_command(["test"], []))

    def test_hardware_report_excludes_identifiers(self):
        raw = {"SPHardwareDataType": [{"chip_type": "Apple M1", "serial_number": "SECRET", "platform_UUID": "SECRET"}],
               "SPDisplaysDataType": [{"sppci_model": "Apple M1", "sppci_cores": "7"}]}
        with patch.object(environment, "run_command", return_value=json.dumps(raw)):
            report = environment.inspect_apple_hardware([])
        self.assertEqual(report["chip"], "Apple M1")
        self.assertNotIn("SECRET", json.dumps(report))

    def test_invalid_hardware_json(self):
        with patch.object(environment, "run_command", return_value="invalid"):
            self.assertIsNone(environment.inspect_apple_hardware([]))

    def test_report_serializes_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(environment, "run_command", return_value=None), patch.object(environment.platform, "system", return_value="Linux"):
                report = environment.collect_environment(directory, include_torch=False)
        self.assertEqual(report["torch"]["import_status"], "skipped")
        self.assertIsNone(report["git_commit"])
        self.assertGreater(report["disk"]["total_bytes"], 0)
        json.dumps(report, allow_nan=False)

    def test_cli_saves_same_json_it_prints(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            result = subprocess.run([
                sys.executable, str(root / "scripts/check_environment.py"),
                "--skip-torch", "--output", str(output),
            ], cwd=root, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
