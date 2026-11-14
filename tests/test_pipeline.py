import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PipelineTests(unittest.TestCase):
    def run_fixture(self, root, failure_stage=None):
        source = Path(__file__).resolve().parents[1]
        (root / "scripts").mkdir()
        shutil.copyfile(source / "scripts/complete_project.py", root / "scripts/complete_project.py")
        for environment in (".venv-mlx", ".venv-eval"):
            target = root / environment / "bin"
            target.mkdir(parents=True)
            (target / "python").symlink_to(sys.executable)
        stage6 = root / "results/study/local-study-v1"
        stage6.mkdir(parents=True)
        (stage6 / "stage6_summary.json").write_text(json.dumps({"status": "complete"}))
        (root / "docs").mkdir()
        for name in ("evaluate.py", "freeze_evaluation.py", "grade_results.py", "analyze_results.py", "report_project.py"):
            (root / "scripts" / name).write_text(
                "import json, pathlib, sys\n"
                "root = pathlib.Path(__file__).resolve().parents[1]\n"
                "stage = pathlib.Path(__file__).name\n"
                "if stage == 'evaluate.py': stage += ':' + sys.argv[sys.argv.index('--scope')+1]\n"
                "with (root / 'events.jsonl').open('a') as stream: stream.write(json.dumps(stage)+'\\n')\n"
                f"if stage == {failure_stage!r}: raise SystemExit(1)\n"
                f"if stage == 'report_project.py': (root / 'docs/handoff.json').write_text(json.dumps({{'status': {'partial' if failure_stage == 'evaluate.py:test' else 'complete'!r}}}))\n"
            )
        result = subprocess.run([sys.executable, str(root / "scripts/complete_project.py")],
                                capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(source / "src")})
        events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        state = json.loads((root / "results/runtime/project_pipeline.json").read_text())
        return result, events, state

    def test_development_failure_halts_before_freeze_or_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, events, state = self.run_fixture(Path(temporary), "evaluate.py:development")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(events, ["evaluate.py:development"])
            self.assertEqual(state["status"], "failed")

    def test_complete_evidence_moves_through_all_stages_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, events, state = self.run_fixture(Path(temporary))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events, ["evaluate.py:development", "freeze_evaluation.py", "evaluate.py:test",
                                      "grade_results.py", "analyze_results.py", "report_project.py"])
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["stage"], "finished")

    def test_resource_failure_in_test_still_attempts_an_honest_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, events, state = self.run_fixture(Path(temporary), "evaluate.py:test")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events[-3:], ["grade_results.py", "analyze_results.py", "report_project.py"])
            self.assertEqual(state["status"], "partial")
