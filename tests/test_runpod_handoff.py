import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location("runpod_launcher_test", ROOT / "scripts/runpod_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunPodHandoffTests(unittest.TestCase):
    def test_runpod_identity_separates_provider_and_binds_launcher_bytes(self):
        module = launcher()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "launcher.py"
            path.write_text("first")
            base = {"backend": "mlx_cuda12", "gpu": {"name": "example"}}
            first = module.runpod_identity(base, path)
            self.assertEqual(first["provider"], "runpod")
            self.assertEqual(first["gpu"], base["gpu"])
            self.assertNotIn("provider", base)
            path.write_text("changed")
            self.assertNotEqual(first, module.runpod_identity(base, path))

    def test_launcher_refuses_mac_workspace_before_importing_gpu_runtime(self):
        module = launcher()
        with tempfile.TemporaryDirectory() as temporary, patch.object(module, "__file__", str(Path(temporary) / "scripts/runpod_inference.py")):
            with self.assertRaisesRegex(ValueError, "never the active Mac or Colab"):
                module.main()

    def test_notebook_parses_and_embeds_exact_launcher(self):
        notebook = json.loads((ROOT / "notebooks/runpod_inference.ipynb").read_text())
        embedded = None
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            self.assertEqual(cell["outputs"], [])
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RUNPOD_LAUNCHER" for t in node.targets):
                    embedded = ast.literal_eval(node.value)
        self.assertEqual(embedded, (ROOT / "scripts/runpod_inference.py").read_text())
