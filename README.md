# s1 on a resource-constrained 1.5B model

Study whether original s1K reasoning fine-tuning and test-time budget forcing improve Qwen2.5-1.5B-Instruct math accuracy relative to their token and latency costs. This is an adaptation using LoRA, with no assumed performance improvement.

Specification: [`NLP_Proposal.pdf`](NLP_Proposal.pdf). Roadmap and progress: [`plan.md`](plan.md).

## Current status

Stages 1–6 are complete under the user-approved **single existing adapter** decision. The completed model trained on **896 examples** and validated on **100 held-out examples**, with teacher-target NLL 0.563623 → 0.479717. **No second trial or fresh all-data retraining is scheduled.** We are proceeding to native budget-protocol validation, then frozen benchmark evaluation and reporting as gates pass. This departs from the original 996-example final-retraining plan; see [`configs/single_run_decision.json`](configs/single_run_decision.json) and [`docs/first_trial_review.md`](docs/first_trial_review.md). No benchmark score exists yet. Read progress with `python3 scripts/status.py`.

The raw split is 900 train / 100 development (seed 42). Four confirmed benchmark-overlap examples are excluded through a separate eligibility manifest, leaving 896 train / 100 development and 996 eventual final-training examples before context filtering. See [`docs/data_audit.md`](docs/data_audit.md). All downstream training/development scripts must use `manifests/s1k_eligibility.json`, not all raw rows.

The target computer is an Apple M1 MacBook Air with 8 GB unified memory and a 7-core GPU. FP16 MPS inference passed, followed by a stable MLX QLoRA pilot using 4-bit/group-64 weights and FP32 activations. All 996 eligible complete examples remain covered. See [`docs/hardware.md`](docs/hardware.md) and [`docs/research_audit.md`](docs/research_audit.md).

Stage 3 completed 28/28 generations, with exact token-ID repeats within each run and across fresh model loads. The main development test used five questions, A/B prompts and two repetitions; unfinished answers are explicitly marked. No accuracy claim is made. See [`docs/stage3_report.md`](docs/stage3_report.md).

Stage 4 formatting/masking checks passed for all 996 eligible examples. Complete examples are preserved without truncation; measured full-length feasibility supports the 9,216-token training ceiling. The latest grader-environment suite ran 111 tests successfully (five native GPU tests and one optional PEFT test skipped). The PEFT check passes in the inference environment; the five native tests passed before full training. See [`docs/training_format_audit.md`](docs/training_format_audit.md).

The remaining workflow is documented in [`docs/budget_protocol.md`](docs/budget_protocol.md), [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) and [`docs/reproducibility.md`](docs/reproducibility.md). The final paper writer includes hardware specifications and measured runtime at the end of `docs/paper.md`, after actual results exist. It labels incomplete studies explicitly.

## Minimal setup

Python 3.11 or newer is required; the inspected local interpreter is Python 3.13.3. Run from this repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_environment.py --output results/environment/stage1-isolated.json
```

The editable install may fetch the setuptools build requirement. The base package has no runtime dependencies. The optional data extra adds the tested tokenizer/Parquet/template stack without Torch. Separate validated training, MLX, grading and reporting locks are now available; do not install the original multi-GPU requirements.

You can also run checks without installing this package:

```bash
python3 scripts/check_environment.py --output results/environment/current.json
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

If Python 3.13 reports `No module named s1small` despite an editable install, this Mac may have marked the project's `.pth` link hidden. Run `.venv/bin/python scripts/fix_editable_install.py` (or use the `.venv-inference` interpreter for that environment). The repair is limited to this project's verified source-path link; Python otherwise skips hidden `.pth` files. The `PYTHONPATH=src` command also supports direct source testing.

The command-line entry point `s1small-check` is available after installation. All checker output is JSON. `--skip-torch` avoids importing PyTorch. Missing Torch, CUDA or MPS is reported without failing an otherwise valid CPU environment. A sandbox may hide Metal devices; use an approved ordinary local terminal to confirm MPS access. The checker does not allocate GPU tensors or load models.

## Reproducibility and storage

- Exact upstream implementation, model, tokenizer and dataset revisions: [`configs/sources.json`](configs/sources.json). Later download commands must use these explicit pins.
- Immutable row/split/eligibility manifests and file hashes live in `manifests/`; they contain IDs and checksums, not copied dataset text.
- Source audit and modern API alternatives: [`docs/research_audit.md`](docs/research_audit.md).
- Recorded design choices: [`docs/decisions.md`](docs/decisions.md).
- Save local data in `data/`, weights/adapters in `artifacts/`, and raw runs in `results/`; these locations are ignored by Git. Export small reviewed summaries into documentation later.
- Never commit credentials, Hub caches, weights or large raw runs. The environment checker filters Apple serial numbers and UUIDs from its saved report.
- Save run configuration hashes, package versions, actual hardware, source revisions and Git commit for each experiment. Initial reports predate the first local commit and report null explicitly; later runs record the actual commit and code hashes.

The Stage 2 isolated environment snapshot recorded approximately 15 GiB free. Recheck before downloads; 30-40 GiB free is a suggested working allowance, not a validated requirement. Free disk does not increase the 8 GB unified memory.

## Planned experiment

Use original `simplescaling/s1K`, a deterministic 900/100 development split, and final retraining from the same base on all eligible examples after settings are frozen. Compare direct and reasoning-prompted base baselines, adapted natural stopping, four reasoning caps, and up to one/two forced `Wait` continuations. Final evaluation uses the fixed MATH-500 test split, with per-question predictions, accuracy/cost analysis and paired uncertainty.

Current: Stage 6 resumable development/final training on the measured MLX QLoRA path. See `docs/training_protocol.md` for the predeclared two-learning-rate grid, complete development-loss selection, fresh final retraining and resume commands. The technical pilot completed five updates and a fresh-process adapter reload; see `docs/hardware_pilot.md`. These are not final accuracy results. No test scores may guide development choices.

## Stage 2 commands

```bash
.venv/bin/python -m pip install -e '.[data]'
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/inspect_lengths.py --limit 1
.venv/bin/python scripts/inspect_lengths.py
.venv/bin/python scripts/report_data_audit.py
.venv/bin/python -m unittest discover -s tests -v
```

The acquisition command needs network access on first use. It downloads only pinned s1K and tokenizer files; MATH-500 JSONL is streamed and only IDs/questions are saved, with other fields discarded. It does not download model weights. Subsequent verification can use `scripts/prepare_data.py --offline`. Changed existing manifests or length outputs fail rather than overwrite. `report_data_audit.py` regenerates the derived Markdown/summary export from saved audit files.

The validated Stage 2 runtime snapshot is [`requirements-data-lock.txt`](requirements-data-lock.txt). On the same platform/interpreter family, install it with `.venv/bin/python -m pip install -r requirements-data-lock.txt`. It covers data/tokenizer workflows only; use the separate MLX lock for the accepted training path. Actual commands and checks are in [`docs/stage2_report.md`](docs/stage2_report.md).

No examples have been truncated or excluded for length. All 1,000 complete examples range from 712 to 9,109 tokens; 988 fit in 8,192. Complete backward feasibility was subsequently measured in Stage 5. The corpus has one normalized duplicate question pair, both in training, retained as original rows and disclosed in the data audit.

## Stage 3 commands (model verification passed)

Reuse the existing local ML packages without modifying them:

```bash
python3 -m venv --system-site-packages .venv-inference
.venv-inference/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python scripts/download_model.py --transport ranges
.venv-inference/bin/python scripts/complete_baseline_smoke.py --prefix stage3-local-v1
```

On this Mac, MPS inference needs an ordinary native terminal or an approved run outside the execution sandbox. The workflow waits for verified checkpoint provenance, then runs a one-question short test, five-question A/B repetitions, and a fresh-load comparison. It stops on a generation/load failure. It never opens MATH-500. All outputs go to ignored `results/dev_baselines/` directories. Use a new prefix to retain an earlier failed run for review; existing completed runs are retained.

For a fully isolated environment instead of shared system packages, install `.[data,inference]` or the recorded runtime closure in `requirements-inference-lock.txt`. That inference snapshot passed actual model verification. Full `pip check` in the shared environment reports unrelated global conflicts; the project's 36-distribution runtime closure has been checked separately and is consistent. See [`docs/baseline_protocol.md`](docs/baseline_protocol.md).

## Research reference and release status

Muennighoff et al. (2025), [s1: Simple test-time scaling, original paper version](https://arxiv.org/abs/2501.19393v1), and the [official implementation](https://github.com/simplescaling/s1). This study changes model size, training method and hardware. Dataset/model/code license and redistribution checks remain required before a public release; no project release license is selected here.

## Full local training and progress

The current workflow uses the exported `local-study-v1-trial-1` adapter directly. Its 56 updates and all 896 complete examples are retained. The original two-trial scheduler is retired; do not resume it or launch fresh training. `complete_project.py` uses the adopted Stage 6 summary to start development protocol validation directly, with no training command. It then freezes evaluation, runs the nine conditions, grades locally, generates analysis and writes the local paper/demo handoff. Progress and logs are in `results/runtime/project_pipeline.json` and `project_pipeline.log`.

Read durable progress without loading a model:

```bash
python3 scripts/status.py
```

The tested source checkpoint before these full runs is local Git commit `f1b0f4e78af13337a54350d8eeeb81f2643991f2`. Older setup/smoke records predate that commit and retain their recorded code hashes/null commit.
