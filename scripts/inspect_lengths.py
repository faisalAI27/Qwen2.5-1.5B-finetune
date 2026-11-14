"""Audit complete original-s1 formatted sequences with the pinned 1.5B tokenizer."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256, identify_rows, save_json, save_jsonl
from s1small.formatting import measure_example
from s1small.lengths import summarize_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Smoke test on the first N rows, saved separately")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    import pyarrow.parquet as parquet
    import yaml
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/dataset.yaml").read_text())
    tokenizer = AutoTokenizer.from_pretrained(root / "data/raw/tokenizer", local_files_only=True, trust_remote_code=False)
    rows = parquet.read_table(root / "data/raw/s1k" / config["training_file"]).to_pylist()
    records = identify_rows(rows)
    if args.limit:
        rows, records = rows[:args.limit], records[:args.limit]
    started = time.monotonic()
    lengths = []
    for index, (record, row) in enumerate(zip(records, rows)):
        lengths.append({"id": record["id"], **measure_example(row, tokenizer)})
        if (index + 1) % 100 == 0:
            print(f"Measured {index + 1}/{len(rows)}", file=sys.stderr, flush=True)
    summary = summarize_lengths(lengths, config["candidate_context_limits"])
    summary.update({"tokenizer_source": json.loads((root / "configs/sources.json").read_text())["tokenizer"],
                    "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                    "split_manifest_sha256": file_sha256(root / "manifests/s1k_split.json"),
                    "truncation": False, "exclusions_applied": [],
                    "phase_accounting": "Single full-sequence offset mapping; boundary-spanning tokens assigned to structure",
                    "special_tokens": {token: tokenizer.encode(token, add_special_tokens=False) for token in
                                       ["<|im_start|>", "<|im_end|>", "<|fim_pad|>", "<|im_start|>think\n", "\n<|im_start|>answer\n"]}})
    directory = root / "results/data_audit" / (f"smoke-{args.limit}" if args.limit else "lengths")
    save_jsonl(directory / "per_example.jsonl", lengths)
    save_json(directory / "summary.json", summary)
    print(json.dumps({"summary": summary, "elapsed_seconds": time.monotonic() - started}, indent=2))


if __name__ == "__main__":
    main()
