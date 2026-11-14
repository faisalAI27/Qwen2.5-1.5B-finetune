"""Verified, lazy complete-example access to the immutable s1K partitions."""

import json
from pathlib import Path

from s1small.data import digest, file_sha256
from s1small.formatting import make_training_example


class TrainingCorpus:
    def __init__(self, root, tokenizer, max_tokens=None):
        import pyarrow.parquet as parquet
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        provenance = json.loads((self.root / "manifests/data_provenance.json").read_text())
        dataset = self.root / "data/raw/s1k/data/train-00000-of-00001.parquet"
        expected = next(row["sha256"] for row in provenance["files"]
                        if row["path"] == str(dataset.relative_to(self.root)))
        if file_sha256(dataset) != expected:
            raise ValueError("Training source differs from recorded provenance")
        self.eligibility = json.loads((self.root / "manifests/s1k_eligibility.json").read_text())
        if (file_sha256(self.root / "manifests/s1k_split.json") != self.eligibility["original_split_sha256"]
                or file_sha256(self.root / "configs/data_exclusions.json") != self.eligibility["exclusion_config_sha256"]):
            raise ValueError("Split or exclusion configuration changed")
        audit = json.loads((self.root / "docs/training_format_summary.json").read_text())
        if (file_sha256(self.root / "configs/formatting.json") != audit["format_config_sha256"]
                or file_sha256(self.root / "manifests/s1k_eligibility.json") != audit["eligibility_sha256"]):
            raise ValueError("Corpus no longer matches the complete-example format audit")
        train = self.eligibility["train"]
        dev = self.eligibility["development"]
        final = self.eligibility["final_training"]
        if {row["id"] for row in train} & {row["id"] for row in dev}:
            raise ValueError("Training and development IDs overlap")
        if sorted(train + dev, key=lambda row: row["id"]) != final:
            raise ValueError("Final-training partition differs from the eligible union")
        self.rows = parquet.read_table(dataset).to_pylist()
        self.references = {}
        for reference in final:
            if reference["id"] in self.references or digest(self.rows[reference["source_row"]]) != reference["row_sha256"]:
                raise ValueError("Duplicate ID or modified training row")
            self.references[reference["id"]] = reference

    def ids(self, partition):
        if partition not in ("train", "development", "final_training"):
            raise ValueError("Unknown training partition")
        return [row["id"] for row in self.eligibility[partition]]

    def example(self, identifier):
        reference = self.references[identifier]
        encoded = make_training_example(self.rows[reference["source_row"]], self.tokenizer,
                                        max_length=self.max_tokens)
        return {"id": identifier, "encoded": encoded}

    def validate_lengths(self, partition):
        # Tokenize one at a time: do not retain millions of Python token objects.
        lengths = []
        for identifier in self.ids(partition):
            example = self.example(identifier)
            lengths.append({"id": identifier, "tokens": len(example["encoded"]["input_ids"]),
                            "supervised_tokens": sum(label != -100 for label in example["encoded"]["labels"][1:])})
        return lengths
