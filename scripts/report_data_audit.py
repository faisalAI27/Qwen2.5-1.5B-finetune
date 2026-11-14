"""Regenerate the reviewed Stage 2 data report from saved audit outputs."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256, read_jsonl
from s1small.lengths import summarize_lengths


def main():
    root = Path(__file__).resolve().parents[1]
    audit = root / "results/data_audit"
    schema = json.loads((audit / "schema.json").read_text())
    summary = json.loads((audit / "lengths/summary.json").read_text())
    lengths = read_jsonl(audit / "lengths/per_example.jsonl")
    eligibility = json.loads((root / "manifests/s1k_eligibility.json").read_text())
    provenance = json.loads((root / "manifests/data_provenance.json").read_text())
    split = json.loads((root / "manifests/s1k_split.json").read_text())
    limits = [row["limit"] for row in summary["context_limits"]]
    eligible_ids = {row["id"] for row in eligibility["final_training"]}
    eligible_lengths = [row for row in lengths if row["id"] in eligible_ids]
    eligible_summary = summarize_lengths(eligible_lengths, limits)
    partition_ids = {key: {row["id"] for row in eligibility[key]} for key in ("train", "development")}
    lines = [
        "# Stage 2 data and length audit", "", "Measured on 2026-11-14. No model weights were loaded, no training ran, and no benchmark answers or scores were used.", "",
        "## Acquisition and validation", "",
        "Pinned source revisions are in `configs/sources.json`; input hashes and transfer details are in `manifests/data_provenance.json`.", "",
        f"- Original s1K rows: **{schema['rows']}**. Required-field validation errors: **{len(schema['problems'])}**.",
        f"- Observed columns: {', '.join('`' + field + '`' for field in schema['columns'])}.",
        "- Required fields have nonempty string questions/attempts and nonempty lists of nonempty reasoning strings. Extra fields are retained in the original file; `cot` is null and is not the training target.",
        f"- Persisted input files total {sum(row['bytes'] for row in provenance['files']):,} bytes. This includes tokenizer assets and the projected benchmark questions, not model weights.",
        "- MATH-500 has 500 question/ID records. Its pinned source is JSONL: the transfer necessarily includes other fields transiently; only problem text and IDs are retained. Gold answers/solutions are not persisted, displayed or used.", "",
        "## Fixed split and duplicate questions", "",
        f"Seed **{split['seed']}**; deterministic SHA-256 ranking of seed and stable row ID, with the first 100 ranked IDs assigned to development. Within partitions, stable ID order is saved. Original split: **900 train / 100 development**. Identical reruns are allowed; changed artifacts fail instead of overwriting.", "",
        f"There is **{schema['duplicate_question_groups']} normalized duplicate question pair**: zero-based source rows 486 and 751. Both are in training. They remain separate original examples; the split guard rejects normalized question overlap across training and development. No automatic deduplication was applied.", "",
        "## Benchmark overlap and eligibility", "",
        "All 1,000 x 500 question pairs were screened. Exact screening uses Unicode NFKC, case folding and whitespace normalization while preserving mathematical punctuation/digits. Near screening uses token-sequence similarity >= 0.80, at least eight tokens, and SequenceMatcher with autojunk disabled. Candidates were inspected using question text only. Case folding can merge variable names; this is a lexical screen, not a proof of mathematical equivalence.", "",
        "| s1K row | Benchmark ID | Decision basis |", "|---|---|---|",
    ]
    for exclusion in eligibility["exclusions"]:
        lines.append(f"| {exclusion['s1k_id']} | {exclusion['benchmark_id']} | {exclusion['reason']} |")
    lines += ["", "Two pairs are exact; two were found by near screening. All four overlap examples are in the original training partition. They are excluded from training/tuning eligibility; raw data and the 900/100 manifest are preserved. No replacement rows or benchmark exclusions are introduced.", "",
              f"**Eligible now: {len(eligibility['train'])} train / {len(eligibility['development'])} development; {len(eligibility['final_training'])} for eventual final training**, before any measured context eligibility decisions. `configs/data_exclusions.json` records the reversible exclusion policy; `manifests/s1k_eligibility.json` is the downstream source of eligible IDs.", "",
              "## Complete-sequence token lengths", "",
              "The pinned 1.5B fast tokenizer serialized the released s1 format with its chat template. Each full string was tokenized once, without truncation. Prompt includes chat/system scaffolding up to assistant content. Reasoning/answer count tokens wholly inside those text spans; delimiters, suffix and tokens crossing a phase boundary count as structure. These are additive audit counts, not verified training loss masks (Stage 4). Percentiles use linear interpolation at (n-1)*p.", "",
              "| Component, all 1,000 rows | Min | Median | p90 | p95 | Max |", "|---|---:|---:|---:|---:|---:|"]
    for key, values in summary["components"].items():
        lines.append("| " + key + " | " + " | ".join(f"{values[name]:,.1f}" for name in ("min", "median", "p90", "p95", "max")) + " |")
    lines += ["", "## Context coverage and total-length histogram", "",
              "A row fits only when its entire formatted prompt, reasoning, answer and ending fit. No tail cutting or answer loss is allowed. These counts do not prove that training at the indicated context fits in 8 GB memory.", "",
              "| Context tokens | All rows fit | All rows exceed | Eligible train fit | Eligible dev fit | Eligible final fit |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in summary["context_limits"]:
        limit = row["limit"]
        train_count, dev_count = [sum(item["total"] <= limit and item["id"] in partition_ids[partition] for item in lengths) for partition in ("train", "development")]
        lines.append(f"| {limit:,} | {row['fits_complete_examples']} | {row['exceeds']} | {train_count} | {dev_count} | {train_count + dev_count} |")
    lines += ["", "| Total token bin | Original examples |", "|---|---:|"]
    for row in summary["histogram"]:
        label = f"{row['lower_exclusive']:,} < n <= {row['upper_inclusive']:,}" if row["upper_inclusive"] else f"n > {row['lower_exclusive']:,}"
        lines.append(f"| {label} | {row['count']} |")
    lines += ["", "## Interpretation and next gate", "",
              "No context filtering or truncation is applied. An initial hardware pilot should use complete eligible training examples, then increase length gradually. A context ceiling must be selected from measured memory and runtime; report the resulting coverage before training. Do not claim a short-example pilot covers the full original distribution. The remaining duplicated training pair is retained and must be disclosed in reporting.", "",
              "Lexical overlap screening cannot rule out paraphrases, shared solutions beyond the reviewed candidates, or pretraining contamination. No evaluation score informed exclusions or settings. Keep the full benchmark and use the same 500 IDs for all frozen test conditions.", "",
              "## Reproduction", "", "```bash", ".venv/bin/python scripts/prepare_data.py --offline", ".venv/bin/python scripts/inspect_lengths.py", ".venv/bin/python scripts/report_data_audit.py", "```", "",
              f"Raw split SHA-256: `{file_sha256(root / 'manifests/s1k_split.json')}`.", "",
              f"Eligibility manifest SHA-256: `{file_sha256(root / 'manifests/s1k_eligibility.json')}`.", "",
              f"Tokenizer chat-template SHA-256: `{summary['chat_template_sha256']}`.", ""]
    (root / "docs/data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    export = {"all_rows": summary, "eligible_rows": eligible_summary,
              "eligibility_manifest_sha256": file_sha256(root / "manifests/s1k_eligibility.json")}
    (root / "docs/data_audit_summary.json").write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    print("Regenerated docs/data_audit.md and docs/data_audit_summary.json")


if __name__ == "__main__":
    main()
