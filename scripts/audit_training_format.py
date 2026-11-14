"""Verify complete assistant-only labels with the pinned tokenizer; no training."""

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import digest, file_sha256, save_json, save_jsonl, write_immutable
from s1small.formatting import (ANSWER_START, THINK_START, ContextOverflowError,
                               collate_training_examples, format_example, make_training_example)


def main():
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/formatting.json").read_text())
    if (config["think_start"] != THINK_START or config["answer_start"] != ANSWER_START
            or config["assistant_end"] != "<|im_end|>\n" or config["ignore_index"] != -100
            or config["padding_side"] != "right"
            or config["overflow_policy"] != "reject_complete_example_without_truncation"):
        raise ValueError("Formatting configuration differs from the implemented protocol")
    if config["max_training_tokens"] is not None:
        raise ValueError("This full-corpus audit does not select or apply a hardware context limit")
    provenance = json.loads((root / "manifests/data_provenance.json").read_text())
    for record in provenance["files"]:
        if record["path"].startswith(("data/raw/tokenizer/", "data/raw/s1k/")):
            if file_sha256(root / record["path"]) != record["sha256"]:
                raise ValueError(f"Source hash mismatch: {record['path']}")
    tokenizer = AutoTokenizer.from_pretrained(root / "data/raw/tokenizer", local_files_only=True, trust_remote_code=False)
    pad_ids = tokenizer.encode(config["training_pad_token"], add_special_tokens=False)
    end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(pad_ids) != 1 or len(end_ids) != 1 or pad_ids[0] == end_ids[0] or not 0 <= pad_ids[0] < len(tokenizer):
        raise ValueError("Existing distinct pad/end token IDs are required")
    rows = parquet.read_table(root / "data/raw/s1k/data/train-00000-of-00001.parquet").to_pylist()
    eligibility = json.loads((root / "manifests/s1k_eligibility.json").read_text())
    selected = sorted(eligibility["final_training"], key=lambda row: row["source_row"])
    records, first_examples = [], []
    sample = None
    for reference in selected:
        row = rows[reference["source_row"]]
        if digest(row) != reference["row_sha256"]:
            raise ValueError("Row differs from immutable eligibility manifest")
        encoded = make_training_example(row, tokenizer)
        boundary = encoded["prompt_tokens"]
        if tokenizer.decode(encoded["input_ids"], skip_special_tokens=False) != format_example(row, tokenizer):
            raise ValueError("Decoded full example differs from the released text format")
        if encoded["labels"][:boundary] != [-100] * boundary or encoded["labels"][boundary:] != encoded["input_ids"][boundary:]:
            raise ValueError("Loss mask does not match the assistant boundary")
        if end_ids[0] not in encoded["labels"][boundary:]:
            raise ValueError("Assistant end-of-answer token lost from supervised target")
        records.append({"id": reference["id"], "total_tokens": len(encoded["input_ids"]),
                        "prompt_tokens": boundary, "supervised_tokens": len(encoded["input_ids"]) - boundary,
                        "labels_sha256": digest(encoded["labels"])})
        if len(first_examples) < 2:
            first_examples.append(encoded)
        if sample is None:
            sample = {"reference": reference, "encoded": encoded,
                      "masked_text": tokenizer.decode(encoded["input_ids"][:boundary], skip_special_tokens=False),
                      "supervised_text": tokenizer.decode(encoded["input_ids"][boundary:], skip_special_tokens=False)}
    batch = collate_training_examples(first_examples, pad_ids[0])
    for original, labels, attention in zip(first_examples, batch["labels"], batch["attention_mask"]):
        length = len(original["input_ids"])
        if labels[:length] != original["labels"] or any(value != -100 for value in labels[length:]) or any(attention[length:]):
            raise ValueError("Padding changes supervised labels or unmasked padding")
    first = sample["encoded"]
    overflow_rejected = False
    try:
        make_training_example(rows[sample["reference"]["source_row"]], tokenizer, max_length=len(first["input_ids"]) - 1)
    except ContextOverflowError:
        overflow_rejected = True
    if not overflow_rejected:
        raise ValueError("Overflow policy did not reject an incomplete context")
    directory = root / "results/training_format"
    text = (f"Sample ID: {sample['reference']['id']}\n\n[IGNORED PROMPT; labels=-100]\n"
            + sample["masked_text"] + "\n[SUPERVISED ASSISTANT; labels=input_ids, no manual shift]\n" + sample["supervised_text"])
    write_immutable(directory / "decoded_sample.txt", text.encode())
    save_jsonl(directory / "per_example.jsonl", records)
    report = {
        "eligible_examples_checked": len(records), "train_rows": len(eligibility["train"]),
        "development_rows": len(eligibility["development"]), "training_performed": False,
        "tokenizer_revision": json.loads((root / "configs/sources.json").read_text())["tokenizer"]["revision"],
        "format_config_sha256": file_sha256(root / "configs/formatting.json"),
        "eligibility_sha256": file_sha256(root / "manifests/s1k_eligibility.json"),
        "training_pad_token_id": pad_ids[0], "assistant_end_token_id": end_ids[0],
        "think_start_ids": tokenizer.encode(THINK_START, add_special_tokens=False),
        "answer_start_ids": tokenizer.encode(ANSWER_START, add_special_tokens=False),
        "all_prompt_prefixes_align": True, "all_labels_mask_prompt_and_preserve_assistant": True,
        "all_full_sequences_round_trip": True, "padding_mask_verified": True,
        "overflow_rejected_without_truncation": overflow_rejected, "selected_training_context": None,
        "sample_id": sample["reference"]["id"], "sample_total_tokens": len(first["input_ids"]),
        "sample_prompt_tokens": first["prompt_tokens"],
        "sample_supervised_tokens": len(first["input_ids"]) - first["prompt_tokens"],
        "sample_sha256": file_sha256(directory / "decoded_sample.txt"),
        "per_example_sha256": file_sha256(directory / "per_example.jsonl"),
        "batch_shape": [len(batch["input_ids"]), len(batch["input_ids"][0])],
    }
    save_json(directory / "summary.json", report)
    (root / "docs/training_format_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
