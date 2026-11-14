"""Freeze checks and crash-safe, fixed-denominator prediction slots."""

import json
from pathlib import Path

from s1small.data import digest, file_sha256, save_json


def verify_freeze(root, freeze):
    root = Path(root)
    if freeze["scope"] != "frozen_test_evaluation" or freeze["test_used_for_selection"]:
        raise ValueError("A pre-test evaluation freeze is required")
    for filename, expected in freeze["files_sha256"].items():
        if file_sha256(root / filename) != expected:
            raise ValueError(f"Frozen file changed: {filename}")


def validate_question_identity(rows, manifest):
    expected = manifest["rows"]
    if len(rows) != len(expected) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Question denominator or IDs changed")
    for row, reference in zip(rows, expected):
        if row["id"] != reference["id"] or digest(row["question"]) != reference["question_sha256"]:
            raise ValueError("Question order or text differs from the sealed question manifest")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


class PredictionStore:
    def __init__(self, directory, planned, manifest_id, maximum_attempts=2):
        self.directory = Path(directory)
        self.slots = self.directory / "slots"
        self.attempts = self.directory / "attempts"
        self.slots.mkdir(parents=True, exist_ok=True)
        self.attempts.mkdir(parents=True, exist_ok=True)
        self.manifest_id = manifest_id
        self.maximum_attempts = maximum_attempts
        self.planned = list(planned)
        self.keys = {self.key(q, c): (q, c) for q, c in planned}
        if len(self.keys) != len(self.planned):
            raise ValueError("Duplicate planned prediction keys")
        for question, condition in self.planned:
            path = self.slots / (self.key(question, condition) + ".json")
            if not path.exists():
                save_json(path, {"question_id": question, "condition": condition,
                                "manifest_id": manifest_id, "status": "not_attempted", "attempt": 0})
            self.read(question, condition)

    @staticmethod
    def key(question, condition):
        return digest([question, condition])[:32]

    def read(self, question, condition):
        key = self.key(question, condition)
        if self.keys.get(key) != (question, condition):
            raise ValueError("Prediction is outside the fixed plan")
        path = self.slots / (key + ".json")
        record = json.loads(path.read_text())
        if (record["manifest_id"] != self.manifest_id or record["question_id"] != question
                or record["condition"] != condition):
            raise ValueError("Existing slot differs from this run's identity")
        # Recover an attempt that reached disk just before a process crash left
        # the slot pointer unchanged. Never regenerate a completed prediction.
        next_path = self.attempts / f"{key}-{record['attempt'] + 1:02d}.json"
        if next_path.exists():
            recovered = json.loads(next_path.read_text())
            if (recovered["manifest_id"] != self.manifest_id or recovered["question_id"] != question
                    or recovered["condition"] != condition or recovered["attempt"] != record["attempt"] + 1):
                raise ValueError("Attempt belongs to a different frozen run")
            atomic_json(path, recovered)
            record = recovered
        return record

    def pending(self, question, condition):
        record = self.read(question, condition)
        return (record["status"] == "not_attempted"
                or (record["status"] == "infrastructure_failure" and record["attempt"] < self.maximum_attempts))

    def save_attempt(self, question, condition, values):
        old = self.read(question, condition)
        if not self.pending(question, condition):
            raise ValueError("Completed outputs cannot be replaced; retries are infrastructure-only and bounded")
        if values["status"] not in ("ok", "infrastructure_failure"):
            raise ValueError("Unknown generation status")
        record = {**values, "question_id": question, "condition": condition,
                  "manifest_id": self.manifest_id, "attempt": old["attempt"] + 1}
        key = self.key(question, condition)
        attempt_path = self.attempts / f"{key}-{record['attempt']:02d}.json"
        if attempt_path.exists():
            if json.loads(attempt_path.read_text()) != record:
                raise ValueError("Refusing to overwrite a different durable attempt")
        else:
            atomic_json(attempt_path, record)
        atomic_json(self.slots / (key + ".json"), record)
        return record

    def records(self):
        return [self.read(question, condition) for question, condition in self.planned]
