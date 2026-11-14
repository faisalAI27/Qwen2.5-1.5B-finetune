"""Development-only selection and baseline configuration validation."""

import json

from s1small.data import digest, file_sha256, normalize_question


def validate_config(config):
    if config["scope"] != "development_smoke_only":
        raise ValueError("This runner is restricted to development smoke tests")
    if not 1 <= config["sample_count"] <= 5 or not 1 <= config["repeats"] <= 3:
        raise ValueError("Smoke tests allow 1-5 development questions and 1-3 repetitions")
    if not 1 <= config["max_new_tokens"] <= 1024 or config["max_input_tokens"] < 1:
        raise ValueError("Invalid smoke-test token limits")
    if config["timeout_seconds"] <= 0 or not 0 < config["mps_memory_fraction"] <= 1:
        raise ValueError("Invalid timeout or GPU memory fraction")
    if config["do_sample"] or config["num_beams"] != 1 or not config["use_cache"]:
        raise ValueError("Baseline smoke tests require greedy single-beam cached generation")


def select_development(eligibility, original_split, questions, limit):
    allowed = {row["id"] for row in original_split["development"]}
    selected = eligibility["development"][:limit]
    if len(selected) != limit or any(row["id"] not in allowed for row in selected):
        raise ValueError("Selection must contain eligible original development IDs only")
    output = []
    for row in selected:
        question = questions[row["source_row"]]
        if digest(normalize_question(question)) != row["question_sha256"]:
            raise ValueError("Question content differs from the immutable manifest")
        output.append({**row, "question": question})
    return output


def load_development(root, limit):
    import pyarrow.parquet as parquet
    provenance = json.loads((root / "manifests/data_provenance.json").read_text())
    path = root / "data/raw/s1k/data/train-00000-of-00001.parquet"
    record = next(row for row in provenance["files"] if row["path"] == str(path.relative_to(root)))
    if file_sha256(path) != record["sha256"]:
        raise ValueError("Training dataset checksum mismatch")
    eligibility = json.loads((root / "manifests/s1k_eligibility.json").read_text())
    split_path = root / "manifests/s1k_split.json"
    if file_sha256(split_path) != eligibility["original_split_sha256"]:
        raise ValueError("Original split checksum mismatch")
    if file_sha256(root / "configs/data_exclusions.json") != eligibility["exclusion_config_sha256"]:
        raise ValueError("Exclusion configuration checksum mismatch")
    # Only question text is loaded. Reference answers/trajectories are not prompts.
    questions = parquet.read_table(path, columns=["question"])["question"].to_pylist()
    return select_development(eligibility, json.loads(split_path.read_text()), questions, limit)
