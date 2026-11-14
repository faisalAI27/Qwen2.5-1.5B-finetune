"""Pinned MATH-500 acquisition permitted only after the evaluation freeze."""

import hashlib
import json
from pathlib import Path
import urllib.request

from s1small.data import file_sha256, read_jsonl, save_json
from s1small.experiment import validate_question_identity, verify_freeze


def sealed_test_rows(root):
    root = Path(root)
    freeze = json.loads((root / "manifests/evaluation_freeze.json").read_text())
    verify_freeze(root, freeze)
    from s1small.mlx_decode import runtime_signature
    if runtime_signature() != freeze["generation_runtime"]:
        raise ValueError("Generation runtime changed after freeze; do not open benchmark answers")
    sources = json.loads((root / "configs/sources.json").read_text())
    source = sources["test_dataset"]
    directory = root / "data/raw/math500"
    path = directory / "test_with_answers.jsonl"
    provenance_path = directory / "test_source.json"
    if not path.exists():
        url = f"https://huggingface.co/datasets/{source['id']}/resolve/{source['revision']}/test.jsonl"
        temporary = path.with_suffix(".jsonl.incomplete")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            etag = response.headers.get("x-linked-etag") or response.headers.get("etag", "")
            etag = etag.strip('"')
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        raw = temporary.read_bytes()
        verified = None
        if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag):
            if hashlib.sha256(raw).hexdigest() != etag:
                raise ValueError("Benchmark upstream SHA-256 differs")
            verified = "upstream_sha256"
        elif len(etag) == 40 and all(c in "0123456789abcdef" for c in etag):
            if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != etag:
                raise ValueError("Benchmark upstream Git blob differs")
            verified = "upstream_git_blob_sha1"
        temporary.replace(path)
        save_json(provenance_path, {"source": source, "url": url, "sha256": file_sha256(path),
                  "etag": etag, "upstream_verification": verified,
                  "evaluation_freeze_sha256": file_sha256(root / "manifests/evaluation_freeze.json")})
    provenance = json.loads(provenance_path.read_text())
    if (provenance["source"] != source or file_sha256(path) != provenance["sha256"]
            or provenance["evaluation_freeze_sha256"] != file_sha256(root / "manifests/evaluation_freeze.json")):
        raise ValueError("Cached benchmark source differs")
    rows = read_jsonl(path)
    projected = [{"id": str(row["unique_id"]), "question": row["problem"], "gold_answer": row["answer"]} for row in rows]
    manifest = json.loads((root / "manifests/math500_questions.json").read_text())
    validate_question_identity(projected, manifest)
    if len(projected) != 500 or any(not isinstance(row["gold_answer"], str) or not row["gold_answer"].strip() for row in projected):
        raise ValueError("Unexpected benchmark answer schema or denominator")
    return projected, freeze
