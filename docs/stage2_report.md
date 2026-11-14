# Stage 2 verification record

Date: 2026-11-14 (Asia/Karachi). Result: Stage 2 acceptance gate passed. No model generation, training or benchmark scoring was run.

## Implemented files and artifacts

- Code: `src/s1small/data.py`, `src/s1small/formatting.py`, `src/s1small/lengths.py`, `scripts/prepare_data.py`, `scripts/inspect_lengths.py`, `scripts/report_data_audit.py`, `tests/test_data.py`.
- Configuration: `configs/dataset.yaml`, `configs/data_exclusions.json`, the optional data dependencies in `pyproject.toml`, and `requirements-data-lock.txt`.
- Immutable manifests: `manifests/s1k_split.json`, `manifests/s1k_eligibility.json`, `manifests/math500_questions.json`, `manifests/data_provenance.json`. These contain IDs, hashes, source pins and eligibility decisions rather than copied training text.
- Reports: `docs/data_audit.md`, `docs/data_audit_summary.json`, this report, and updates to `README.md`, `docs/decisions.md`, `docs/hardware.md`, and `plan.md`.
- Ignored local inputs: `data/raw/s1k/data/train-00000-of-00001.parquet`, four tokenizer assets, and the benchmark question/ID projection. Total input size: 18,498,049 bytes.
- Ignored local evidence: `results/data_audit/schema.json`, `overlaps.json`, `lengths/per_example.jsonl`, `lengths/summary.json`, `smoke-1/`, `isolated_lengths_run.json`, `code_manifest.json`, and the Stage 2 global/isolated environment JSONs.

## Commands actually run

The initial audit used the already installed global data libraries while the isolated dependencies downloaded. The complete preparation and length workflows were then verified in the isolated environment:

```bash
.venv/bin/python -m pip install -e '.[data]'
python3 scripts/prepare_data.py
python3 scripts/inspect_lengths.py --limit 1
python3 scripts/inspect_lengths.py
python3 scripts/prepare_data.py --offline
python3 scripts/report_data_audit.py
.venv/bin/python scripts/prepare_data.py --offline
.venv/bin/python scripts/inspect_lengths.py > results/data_audit/isolated_lengths_run.json
.venv/bin/python scripts/report_data_audit.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
python3 scripts/check_environment.py --skip-torch --output results/environment/stage2-global.json
.venv/bin/python scripts/check_environment.py --skip-torch --output results/environment/stage2-isolated.json
.venv/bin/python -m compileall -q src scripts tests
git diff --check
git check-ignore data/raw/math500/questions.jsonl results/data_audit/lengths/per_example.jsonl
df -h .
```

The editable install command was repeated after adding Jinja2 explicitly for chat-template rendering. The final data environment contains Transformers 5.10.1, Tokenizers 0.22.2, PyArrow 22.0.0, PyYAML 6.0.2 and Jinja2 3.1.6. The complete installed runtime snapshot was generated with `pip freeze --exclude-editable` plus a portable local editable requirement in `requirements-data-lock.txt`; this is not a training lock or a cross-platform hash lock.

Sandbox DNS initially blocked the Hub file-list inspection and data acquisition. Both succeeded on approved retries outside the sandbox. Dataset acquisition used only immutable revision URLs. PyArrow emitted nonfatal CPU-cache/NEON sysctl warnings inside the sandbox; all audit commands completed successfully. Transformers reported that Torch is absent in the isolated environment, as intended for tokenizer-only work. No upstream executable code or remote custom tokenizer code was loaded.

## Observed results

- All 1,000 original s1K rows passed required-field validation. The extra `cot` column is null and is not used; `attempt` remains the teacher-answer target.
- Seed 42 produces the immutable 900/100 split. One normalized duplicate question pair lies entirely in training. A guard rejects duplicate questions across training/development.
- Exhaustive lexical screening identified two exact and two near overlap pairs. Question-only inspection confirmed three duplicate questions and one shared-solution recurrence task. All four are excluded by the separate eligibility manifest, leaving 896 train / 100 development, and 996 eventual final-training examples before context exclusions. No benchmark examples were removed and the split was not refilled.
- Total formatted sequence lengths: min 712, median 5,898, p90 7,336.6, p95 7,767.65, max 9,109 tokens. All four measured phases are additive, with boundary-spanning tokens assigned to structure for this audit.
- Complete original examples fitting contexts 1,024 / 2,048 / 4,096 / 8,192 / 16,384: 2 / 39 / 127 / 988 / 1,000. No truncation or context exclusion was applied. The clean eligible set has 984 of 996 examples fitting 8,192.
- The full isolated length audit took 8.191 seconds on this invocation and matched the previously saved immutable results. This is tokenizer processing time, not model inference or training speed.
- All 30 tests passed in the final installed data environment (0.462 seconds in that invocation). `pip check` found no broken requirements. The suite covers schema failures, unique/stable IDs, deterministic splits, leakage guards, immutable writes, benchmark field projection, reviewed exclusions, overlap matching, official text construction, phase accounting, context boundaries, statistics, and the prior environment checks.
- Additional consistency assertions verified one length record for every original ID, all 1,000 records' additive counts, the 896/100/996 eligibility totals, and the benchmark projection's exact `id`/`question` field set. Raw data and local results are ignored by Git.

## Decisions and limitations

The fixed benchmark source is JSONL, so a literal question-column-only network transfer was unavailable. The raw response is parsed transiently; only question/ID records are retained, and benchmark answer/solution fields are never logged, persisted or used. This transport deviation is documented rather than claiming those bytes were never transferred.

The conservative overlap exclusions are documented in `configs/data_exclusions.json` and preserve all original data/split artifacts. Lexical screening cannot prove absence of semantic or pretraining contamination. Original duplicated training examples remain disclosed rather than silently deduplicated.

Source formatting was introduced now because length auditing requires it; Stage 4's training label masks and collator are not implemented or approved by these tests. No training context, precision, quantization or final hyperparameters have been selected. MPS/model memory feasibility is unmeasured. The latest isolated environment snapshot recorded 16,039,608,320 free disk bytes (about 14.94 GiB); this changes with system usage. No GPU cost or training duration is claimed.

Next stage: base-model loading and deterministic development-only generation smoke tests. Use `manifests/s1k_eligibility.json` for all downstream training/development selection and keep the 500-question benchmark sealed from tuning and scoring until freeze.
