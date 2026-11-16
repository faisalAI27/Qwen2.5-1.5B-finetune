# Reproducing the local study

Run commands from this repository root. All model computation and grading run locally on the recorded Apple M1 MacBook Air. See `hardware.md`, `hardware_pilot.md`, `training_protocol.md`, `budget_protocol.md` and `evaluation_protocol.md` for the measured constraints and methodological changes.

## Environments and inputs

The actual interpreter is CPython 3.13.3, macOS arm64. Separate runtime closures are recorded in `requirements-data-lock.txt`, `requirements-inference-lock.txt`, `requirements-training-lock.txt`, `requirements-mlx-lock.txt`, `requirements-eval-lock.txt` and `requirements-reporting-lock.txt`. The local MLX and grader environments reuse existing system packages without changing them; unrelated global package conflicts are documented, and the relevant closures were checked separately. For a new machine, create separate environments and install the relevant locks; MLX requires Apple silicon. Do not install the original multi-GPU s1 requirements.

The final experiment requires the MLX/data stack; grading requires the evaluation lock, and analysis also requires the reporting lock. Scripts add `src` to their import path. Use `PYTHONPATH=src` for tests because this Mac sometimes marks editable-install `.pth` files hidden. Model and data preparation commands are in the README; every acquisition uses `configs/sources.json` pins, and existing manifests are verified rather than silently replaced.

The converted base must be produced with `scripts/convert_mlx_base.py` from the verified original checkpoint, not substituted with a third-party quantization. Its provenance records source weight SHA-256, all conversion file checksums, four-bit affine quantization and group size 64. Final adapters are MLX format and require `mlx_runtime.load_local_model` / `mlx_decode.load_decoder` with the supplied FP32 activation wrapper. Generic BF16 loading, PEFT loading and adapter merging are not verified substitutes.

## Running and monitoring

**Current override (2026-11-16):** The user requested a review after trial 1 and validation. Trial 1, its 100-row validation, fresh-process verification and the matched 100-row read-only base validation are now complete. The review is in `docs/first_trial_review.md`; the review controller stopped successfully. The original scheduler remains intentionally suspended and the downstream pipeline is paused. Do not run the historical full-continuation commands below or remove `results/runtime/PAUSE` before an explicit next-step decision. See `training_protocol.md` and the review request/status files.

Read current status without loading any model:

```bash
python3 scripts/status.py
```

The currently active Stage 6 command is:

```bash
caffeinate -i .venv-mlx/bin/python scripts/run_development.py --prefix local-study-v1
```

Do not start a duplicate while its parent or training worker is alive. The continuation controller has its own process lock, waits for this job, then runs held-out protocol validation, freeze, all 4,500 predictions, grading, analysis and report generation. Its durable status/logs are `results/runtime/project_pipeline.json` and `project_pipeline.log`. A detached native controller can survive closure of the IDE; `caffeinate` prevents idle sleep while the computer remains awake. Closing the lid, shutdown and power loss can still interrupt computation.

After checking that no controller is already running, a continuation waiting for an existing training parent uses:

```bash
caffeinate -i .venv-mlx/bin/python scripts/complete_project.py \
  --wait-for-stage6 --stage6-pid TRAINING_PARENT_PID --resume-interrupted-stage6
```

The PID is the actual `run_development.py` parent, not a copied stale number. If that parent ends, the controller waits for any remaining training worker, validates checkpoint files, retains incomplete files in an archive, and permits one automatic same-recipe interruption recovery. Numerical failures halt with evidence. It does not start a second training worker. An explicit `results/runtime/PAUSE` file prevents automatic progression while waiting; to stop active computation also stop its process group. Remove that marker only when resuming intentionally. The controller is not a reboot service.

When no training/controller is active, a direct full continuation is:

```bash
caffeinate -i .venv-mlx/bin/python scripts/complete_project.py
```

Completed verified runs are retained. Resume checks reject changed training code, recipe, data or conversion identity. Checkpoints contain adapter and optimizer tensors, exact shuffled cursor, per-example dropout identity and checksums. Last-two retention bounds checkpoint storage. No examples are silently skipped or shortened.

## Reanalysis and delivery

```bash
.venv-eval/bin/python scripts/grade_results.py --run-id math500-local-v1
.venv-eval/bin/python scripts/analyze_results.py --run-id math500-local-v1
.venv-eval/bin/python scripts/report_project.py --run-id math500-local-v1
```

For portable reanalysis, preserve `evaluated.jsonl` and use `analyze_results.py --input evaluated.jsonl --output results/reanalysis`. For complete provenance also preserve the evaluation freeze and hashed files, training manifests/summaries, eligibility and source manifests, raw attempts/slots, converted-base provenance, exported adapters and relevant locks. Data, weights and raw runs are intentionally outside Git; a source checkout alone does not contain trained weights or scores.

Tests can run during training without loading a second GPU model:

```bash
PYTHONPATH=src .venv-inference/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv-eval/bin/python -m unittest discover -s tests -p test_grading.py -v
```

The five native MLX tests require explicit opt-in and were passed separately before full training; do not run them concurrently with the active full model. Plotting tests use synthetic fixtures only. They are not experimental results.

Inspected upstream metadata is recorded in `license_metadata.json`: the pinned Qwen model and original s1K declare Apache-2.0; the inspected MATH-500 metadata does not declare a license. No project release license, redistribution grant, public repository publication or venue submission is assumed. This handoff is a local research draft; review upstream terms before any public release.
