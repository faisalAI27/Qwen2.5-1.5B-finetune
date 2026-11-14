import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    import pyarrow as arrow
    import pyarrow.parquet as parquet
except ImportError:
    arrow = None

from s1small.corpus import TrainingCorpus
from s1small.data import digest, file_sha256


@unittest.skipIf(arrow is None, "Optional Parquet runtime absent")
class CorpusTests(unittest.TestCase):
    def prepare(self, root, change=None):
        for directory in ("data/raw/s1k/data", "manifests", "configs", "docs"):
            (root / directory).mkdir(parents=True)
        rows = [{"question": "q1", "attempt": "a1", "thinking_trajectories": ["t1"]},
                {"question": "q2", "attempt": "a2", "thinking_trajectories": ["t2"]}]
        dataset = root / "data/raw/s1k/data/train-00000-of-00001.parquet"
        parquet.write_table(arrow.Table.from_pylist(rows), dataset)
        refs = [{"id": f"r{i}", "source_row": i, "row_sha256": digest(row)} for i, row in enumerate(rows)]
        def write(path, value):
            (root / path).write_text(json.dumps(value))
        for filename in ("manifests/s1k_split.json", "configs/data_exclusions.json", "configs/formatting.json"):
            write(filename, {})
        eligible = {"train": refs[:1], "development": refs[1:], "final_training": refs,
                    "original_split_sha256": file_sha256(root / "manifests/s1k_split.json"),
                    "exclusion_config_sha256": file_sha256(root / "configs/data_exclusions.json")}
        if change:
            change(eligible)
        write("manifests/s1k_eligibility.json", eligible)
        write("manifests/data_provenance.json", {"files": [{"path": str(dataset.relative_to(root)), "sha256": file_sha256(dataset)}]})
        write("docs/training_format_summary.json", {"format_config_sha256": file_sha256(root / "configs/formatting.json"),
                                                    "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json")})

    def test_eligible_union_and_source_integrity_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            corpus = TrainingCorpus(root, tokenizer=None)
            self.assertEqual(corpus.ids("train"), ["r0"])
            self.assertEqual(corpus.ids("development"), ["r1"])
            self.assertEqual(corpus.ids("final_training"), ["r0", "r1"])
            with self.assertRaises(ValueError):
                corpus.ids("test")
            (root / "configs/data_exclusions.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "Split or exclusion"):
                TrainingCorpus(root, tokenizer=None)

    def test_overlap_incorrect_union_and_tampered_rows_are_rejected(self):
        changes = [lambda d: d.update(development=copy.deepcopy(d["train"])),
                   lambda d: d.update(final_training=d["train"]),
                   lambda d: d["final_training"][0].update(row_sha256="wrong")]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.prepare(root, change)
                with self.assertRaises(ValueError):
                    TrainingCorpus(root, tokenizer=None)


if __name__ == "__main__":
    unittest.main()
