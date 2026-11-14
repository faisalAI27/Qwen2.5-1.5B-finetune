# Stage 2 data and length audit

Measured on 2026-11-14. No model weights were loaded, no training ran, and no benchmark answers or scores were used.

## Acquisition and validation

Pinned source revisions are in `configs/sources.json`; input hashes and transfer details are in `manifests/data_provenance.json`.

- Original s1K rows: **1000**. Required-field validation errors: **0**.
- Observed columns: `attempt`, `cot`, `cot_type`, `metadata`, `question`, `solution`, `source_type`, `thinking_trajectories`.
- Required fields have nonempty string questions/attempts and nonempty lists of nonempty reasoning strings. Extra fields are retained in the original file; `cot` is null and is not the training target.
- Persisted input files total 18,498,049 bytes. This includes tokenizer assets and the projected benchmark questions, not model weights.
- MATH-500 has 500 question/ID records. Its pinned source is JSONL: the transfer necessarily includes other fields transiently; only problem text and IDs are retained. Gold answers/solutions are not persisted, displayed or used.

## Fixed split and duplicate questions

Seed **42**; deterministic SHA-256 ranking of seed and stable row ID, with the first 100 ranked IDs assigned to development. Within partitions, stable ID order is saved. Original split: **900 train / 100 development**. Identical reruns are allowed; changed artifacts fail instead of overwriting.

There is **1 normalized duplicate question pair**: zero-based source rows 486 and 751. Both are in training. They remain separate original examples; the split guard rejects normalized question overlap across training and development. No automatic deduplication was applied.

## Benchmark overlap and eligibility

All 1,000 x 500 question pairs were screened. Exact screening uses Unicode NFKC, case folding and whitespace normalization while preserving mathematical punctuation/digits. Near screening uses token-sequence similarity >= 0.80, at least eight tokens, and SequenceMatcher with autojunk disabled. Candidates were inspected using question text only. Case folding can merge variable names; this is a lexical screen, not a proof of mathematical equivalence.

| s1K row | Benchmark ID | Decision basis |
|---|---|---|
| s1k-0081-10cffa339e5e | test/intermediate_algebra/1849.json | Identical question |
| s1k-0124-b45987b898e8 | test/intermediate_algebra/1411.json | Same question with whitespace and punctuation spacing changes; token sequence similarity 1.0 |
| s1k-0181-34c785ee4d2f | test/intermediate_algebra/558.json | Identical question |
| s1k-0498-dfdb7eb862b7 | test/intermediate_algebra/1063.json | Same complex recurrence and initial value; asks a+b instead of z_2002. Conservatively exclude shared-solution task overlap |

Two pairs are exact; two were found by near screening. All four overlap examples are in the original training partition. They are excluded from training/tuning eligibility; raw data and the 900/100 manifest are preserved. No replacement rows or benchmark exclusions are introduced.

**Eligible now: 896 train / 100 development; 996 for eventual final training**, before any measured context eligibility decisions. `configs/data_exclusions.json` records the reversible exclusion policy; `manifests/s1k_eligibility.json` is the downstream source of eligible IDs.

## Complete-sequence token lengths

The pinned 1.5B fast tokenizer serialized the released s1 format with its chat template. Each full string was tokenized once, without truncation. Prompt includes chat/system scaffolding up to assistant content. Reasoning/answer count tokens wholly inside those text spans; delimiters, suffix and tokens crossing a phase boundary count as structure. These are additive audit counts, not verified training loss masks (Stage 4). Percentiles use linear interpolation at (n-1)*p.

| Component, all 1,000 rows | Min | Median | p90 | p95 | Max |
|---|---:|---:|---:|---:|---:|
| total | 712.0 | 5,898.0 | 7,336.6 | 7,767.6 | 9,109.0 |
| prompt | 41.0 | 138.0 | 246.1 | 307.0 | 859.0 |
| reasoning | 355.0 | 4,968.5 | 6,135.6 | 6,673.4 | 7,993.0 |
| answer | 6.0 | 774.0 | 1,085.1 | 1,240.1 | 3,937.0 |
| structure | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 |

## Context coverage and total-length histogram

A row fits only when its entire formatted prompt, reasoning, answer and ending fit. No tail cutting or answer loss is allowed. These counts do not prove that training at the indicated context fits in 8 GB memory.

| Context tokens | All rows fit | All rows exceed | Eligible train fit | Eligible dev fit | Eligible final fit |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2 | 998 | 2 | 0 | 2 |
| 2,048 | 39 | 961 | 34 | 5 | 39 |
| 4,096 | 127 | 873 | 115 | 12 | 127 |
| 8,192 | 988 | 12 | 888 | 96 | 984 |
| 16,384 | 1000 | 0 | 896 | 100 | 996 |
| 32,768 | 1000 | 0 | 896 | 100 | 996 |

| Total token bin | Original examples |
|---|---:|
| 0 < n <= 1,024 | 2 |
| 1,024 < n <= 2,048 | 37 |
| 2,048 < n <= 4,096 | 88 |
| 4,096 < n <= 8,192 | 861 |
| 8,192 < n <= 16,384 | 12 |
| 16,384 < n <= 32,768 | 0 |
| n > 32,768 | 0 |

## Interpretation and next gate

No context filtering or truncation is applied. An initial hardware pilot should use complete eligible training examples, then increase length gradually. A context ceiling must be selected from measured memory and runtime; report the resulting coverage before training. Do not claim a short-example pilot covers the full original distribution. The remaining duplicated training pair is retained and must be disclosed in reporting.

Lexical overlap screening cannot rule out paraphrases, shared solutions beyond the reviewed candidates, or pretraining contamination. No evaluation score informed exclusions or settings. Keep the full benchmark and use the same 500 IDs for all frozen test conditions.

## Reproduction

```bash
.venv/bin/python scripts/prepare_data.py --offline
.venv/bin/python scripts/inspect_lengths.py
.venv/bin/python scripts/report_data_audit.py
```

Raw split SHA-256: `b05b266913beb7fd276861a6e193c112ed72ad37cf9a624e2a46c71f1bfcd77a`.

Eligibility manifest SHA-256: `124c450f710b98791bfec5c6853ebfd1813eb6829bba86dc8906ecfd4d7a7fee`.

Tokenizer chat-template SHA-256: `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
