"""Stable data validation, split manifests and question-only overlap checks."""

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_immutable(path, content):
    """Permit identical reruns; refuse to replace changed experiment artifacts."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to change existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def save_json(path, value):
    write_immutable(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def save_jsonl(path, records):
    write_immutable(path, "".join(canonical_json(row) + "\n" for row in records).encode("utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_question(text):
    # Preserve digits, signs and punctuation: removing them can merge distinct math.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_s1k(rows, expected_count=1000):
    problems = []
    if len(rows) != expected_count:
        problems.append({"row": None, "error": f"Expected {expected_count} rows; found {len(rows)}"})
    for index, row in enumerate(rows):
        for field in ("question", "attempt"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                problems.append({"row": index, "field": field, "error": "Expected nonempty string"})
        traces = row.get("thinking_trajectories")
        if not isinstance(traces, list) or not traces or any(not isinstance(t, str) or not t.strip() for t in traces):
            problems.append({"row": index, "field": "thinking_trajectories", "error": "Expected nonempty list of nonempty strings"})
    questions = [normalize_question(row["question"]) for row in rows if isinstance(row.get("question"), str)]
    counts = Counter(questions)
    return {
        "rows": len(rows), "expected_rows": expected_count,
        "columns": sorted(set().union(*(row.keys() for row in rows))),
        "required_fields": ["question", "thinking_trajectories", "attempt"],
        "problems": problems,
        "duplicate_question_groups": sum(count > 1 for count in counts.values()),
        "extra_duplicate_rows": sum(count - 1 for count in counts.values()),
    }


def identify_rows(rows):
    return [{
        "id": f"s1k-{index:04d}-{digest(row['question'])[:12]}",
        "source_row": index, "row_sha256": digest(row),
        "question_sha256": digest(normalize_question(row["question"])),
    } for index, row in enumerate(rows)]


def make_splits(records, revision, seed=42, development_size=100):
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs")
    if not 0 < development_size < len(ids):
        raise ValueError("Development size must be between zero and dataset size")
    # Hash ranking is independent of input order and Python's random implementation.
    ranking = sorted(ids, key=lambda item: (digest([seed, item]), item))
    development = set(ranking[:development_size])
    # Stable ID order is the persisted question order within each partition.
    ordered = sorted(records, key=lambda row: row["id"])
    train_questions = {row["question_sha256"] for row in ordered if row["id"] not in development and "question_sha256" in row}
    dev_questions = {row["question_sha256"] for row in ordered if row["id"] in development and "question_sha256" in row}
    if train_questions & dev_questions:
        raise ValueError("Duplicate normalized questions cross train/development; a documented grouped split is required")
    return {
        "schema_version": 1, "dataset_id": "simplescaling/s1K", "dataset_revision": revision,
        "seed": seed, "algorithm": "sha256_seed_and_id_rank_first_100_dev_v1",
        "source_rows_sha256": digest(ordered),
        "train": [row for row in ordered if row["id"] not in development],
        "development": [row for row in ordered if row["id"] in development],
        "eligibility": "Original split only; no overlap/context exclusions applied",
    }


def question_tokens(question):
    text = normalize_question(question)
    return re.findall(r"\w+|[^\w\s]", text)


def apply_eligibility(splits, exclusions):
    excluded = {row["s1k_id"] for row in exclusions}
    original = splits["train"] + splits["development"]
    all_ids = {row["id"] for row in original}
    if len(excluded) != len(exclusions) or not excluded <= all_ids:
        raise ValueError("Duplicate or unknown exclusion IDs")
    train = [row for row in splits["train"] if row["id"] not in excluded]
    development = [row for row in splits["development"] if row["id"] not in excluded]
    return {"schema_version": 1, "dataset_revision": splits["dataset_revision"],
            "context_policy": "No context filtering or truncation applied",
            "exclusions": exclusions, "train": train, "development": development,
            "final_training": sorted(train + development, key=lambda row: row["id"])}


def find_overlaps(training, benchmark, threshold=0.80, minimum_tokens=8):
    """Exhaustive pairwise token-sequence similarity, followed by human review.

    Exact equality uses conservative normalization. Near matches are candidates,
    not automatic exclusions or proof of semantic equivalence.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must be in (0, 1]")
    exact, near = [], []
    benchmark_rows = [(row, normalize_question(row["question"]), question_tokens(row["question"])) for row in benchmark]
    for row in training:
        normalized = normalize_question(row["question"])
        tokens = question_tokens(row["question"])
        matcher = SequenceMatcher(autojunk=False)
        matcher.set_seq2(tokens)
        for other, other_normalized, other_tokens in benchmark_rows:
            pair = {"s1k_id": row["id"], "benchmark_id": other["id"]}
            if normalized == other_normalized:
                exact.append(pair)
            elif min(len(tokens), len(other_tokens)) >= minimum_tokens:
                matcher.set_seq1(other_tokens)
                if matcher.real_quick_ratio() >= threshold and matcher.quick_ratio() >= threshold:
                    score = matcher.ratio()
                    if score >= threshold:
                        near.append({**pair, "similarity": score})
    return {"method": "normalized equality + token SequenceMatcher (autojunk=False)",
            "threshold": threshold, "minimum_tokens": minimum_tokens,
            "exact": exact, "near_candidates": near,
            "exclusions_applied": [],
            "limitations": "Lexical screening cannot rule out paraphrases, semantic overlap or pretraining contamination."}

