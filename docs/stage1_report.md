# Stage 1 verification record

Date: 2026-11-14 (Asia/Karachi). Result: Stage 1 acceptance gate passed. No later stage is marked complete.

## Implemented files

`.gitignore`, `README.md`, `pyproject.toml`, `src/s1small/__init__.py`, `src/s1small/environment.py`, `scripts/check_environment.py`, `tests/test_environment.py`, `configs/sources.json`, `docs/research_audit.md`, `docs/decisions.md`, this report, and progress updates in `plan.md`. The existing proposal and hardware record are preserved. A local Git repository and isolated `.venv` were created; no commit or remote repository was created.

## Actual verification commands and outcomes

Commands below ran from the project root:

```bash
git init
python3 -m venv .venv
.venv/bin/python -m pip install -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_environment.py --output results/environment/stage1-isolated.json
python3 scripts/check_environment.py --output results/environment/stage1-global-sandbox.json
.venv/bin/s1small-check --skip-torch --output results/environment/stage1-entrypoint.json
.venv/bin/python -m pip check
git status --short
git check-ignore .venv/ results/environment/stage1-isolated.json tmp/pdfs/proposal-1.png
python3 scripts/check_environment.py --output results/environment/stage1-global-native.json
```

- Git initialization first failed because `.git` writes are protected, then succeeded with approval outside the sandbox.
- Clean editable installation succeeded. The package has no ML runtime dependencies; only build tooling was needed. A complete ML dependency lock is deliberately not generated until ML workflows are validated.
- Both unit-test invocations passed all 12 tests. The second used the installed package without `PYTHONPATH`; it completed in 0.260 seconds on this invocation. Tests include absent/broken Torch, CPU-only, CUDA, MPS, a sandbox-hidden GPU, missing/timed-out system commands, hardware identifier filtering, malformed hardware JSON, serialization and CLI output persistence.
- All four environment commands exited successfully and produced JSON. The clean venv correctly reports absent ML packages. The global installation reports Torch 2.9.1; MPS is hidden in the sandbox and available in the approved native run. Neither command allocates GPU tensors or loads models.
- `pip check` reported no broken requirements in the isolated environment. Git ignore checks confirmed that the venv, generated reports and temporary PDF images are excluded. There is no initial commit, so recorded Git commit values are null.
- The native report recorded 17,973,129,216 available disk bytes and an MPS recommended working set of 5,726,633,984 bytes. These are time-dependent observations, not training memory measurements.

Public source inspection used the GitHub connector to read immutable repository files, and the Hub metadata endpoints listed in `configs/sources.json`. A Python `urllib.request.urlopen` metadata check first failed on sandbox DNS and succeeded on the approved retry. The checked URLs were:

```text
https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct
https://huggingface.co/api/datasets/simplescaling/s1K
https://huggingface.co/api/datasets/HuggingFaceH4/MATH-500
```

`train/requirements.txt` returned 404 during discovery; the actual dependency file is the repository-root `requirements.txt`, which was then inspected. No upstream code was executed.

## Scope and limitations

No model weights, tokenizer files, dataset rows, training jobs, baseline generations or benchmark scores were produced. No measured GPU cost, model peak memory or generation speed is claimed. Original multi-GPU dependencies were not installed. Only Stage 1 is complete.

The target is local M1/8 GB compute, replacing the roadmap's earlier generic CUDA hardware assumption. MLX, quantization and sequence eligibility are still decisions for measured feasibility work. Stage 2 is next: pinned data/tokenizer acquisition, schema checks, immutable splits, length audit and overlap analysis. Approximately 17 GiB free disk remains; budget the downloads before proceeding. Full training and 4,500-prediction evaluation feasibility are unresolved.
