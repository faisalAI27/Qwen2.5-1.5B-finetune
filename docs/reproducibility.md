# Reproducing the completed study

The final benchmark is the separately frozen RunPod FP16/vLLM run: 500 MATH-500 IDs x nine conditions, 4,500 predictions and 4,500 resolved grades. The 1,287 ungraded Mac outputs are excluded. Training finished on the Mac; no second trial, fresh final retraining or optional AIME evaluation was run. The local controller is intentionally stopped and `results/runtime/PAUSE` prevents accidental continuation. Do not run the historical `complete_project.py` workflow to reproduce the final GPU results.

## Start with saved results (no GPU)

Keep `nlp-runpod-vllm-results (1).zip` in the repository root. Its SHA-256 is `f816d62d7ab56dbbc0da0327469c4b89f0775b35647e221e22d77c64d9edcaee`; the evaluated JSONL SHA-256 is `e392495e76ea56b4505aa82e79c3045a1ee87f18d8edd02a34fb6179ffcd5413`. The ZIP is the complete raw result backup and is intentionally outside Git. The local handoff also contains the original Q4 base, trained MLX adapter, source datasets and evidence needed by the audit. Those assets are not in a source-only clone.

On this computer, the recorded `.venv-eval` environment supports:

```bash
.venv-eval/bin/python scripts/audit_runpod_archive.py 'nlp-runpod-vllm-results (1).zip' \
  --expected-evaluated-sha256 e392495e76ea56b4505aa82e79c3045a1ee87f18d8edd02a34fb6179ffcd5413
MPLCONFIGDIR=tmp/matplotlib .venv-eval/bin/python scripts/finalize_study.py --pdf
.venv-eval/bin/python scripts/demo_results.py
```

For a fresh interpreter, create a separate Python 3.13 environment and install `requirements-eval-lock.txt`, `requirements-reporting-lock.txt` and `requirements-final-report.txt` (these contain the tested local closure; some files use `-e .`, so run from the repository root). Verify package resolution for another platform rather than silently asserting parity. Report generation itself needs no Torch, MLX or CUDA. PDF authoring uses ReportLab; Poppler is used separately for rendered layout verification. Use `PYTHONPATH=src` for unit tests.

The audit verifies archive CRC/paths, pinned code via Git history, source model/adapter inputs, all slots/attempts/grade links, gold answers and complete coverage; it recomputes paired bootstrap intervals and the qualitative selection. It retains small reviewed metadata under `artifacts/remote-results/`. `finalize_study.py` requires this successful audit and known archive hash, streams the evaluated records, adds a post-hoc first-segment/Wait diagnostic, and emits the paper, tables, figures and `docs/final_results.json`. It does not change predictions or train a model.

For independent reanalysis, extract only `results/test/math500-runpod-vllm-v1/evaluated.jsonl` from the ZIP to a new ignored directory, then run:

```bash
MPLBACKEND=Agg .venv-eval/bin/python scripts/analyze_results.py \
  --input results/reanalysis/evaluated.jsonl --output results/reanalysis/analysis
```

This recalculates tables, the five original figures and 10,000 paired bootstrap samples. Fixed seed 42, full 500-question denominators, no multiple-comparison adjustment. The raw generation summary predates grading and retains `test_scored: false`; the graded JSONL and verified analysis establish completed grading. That historical field is not rewritten.

## Reproduce generation only if desired (paid GPU optional)

The final GPU backend code is pinned to `1cae805a2a29b7ba769838380f2fd71f058fa824`, source asset code to `e4096450228f194e6f39963dbe8bc4dd8be19be8`. Use [the RunPod notebook](../notebooks/runpod_inference.ipynb) and [its detailed guide](runpod_inference.md) in a **separate** Linux/CUDA workspace with the original asset bundle. The notebook handles verification, conversion, smoke, development, freeze, test, grading, analysis and durable ZIP saves. Use the pinned requirements and verify notebook path from the repository if renaming a copy. It is not necessary to repeat GPU generation for this completed project.

Do not overlay the GPU freeze on the Mac project. `manifests/evaluation_freeze.json` in this checkout is the preserved Mac parent freeze. The independent GPU freeze is inside the downloaded result ZIP and the review directory. The GPU launcher requires `VLLM_ISOLATED_WORKSPACE` and its exact bundle/launcher manifests; copying the launcher alone is insufficient. Use the existing bundle builder/notebook route. Keep every newly generated run separate from the final reported one; hardware/library changes can alter greedy tokens and must not overwrite frozen outputs.

The GPU reconstructs the frozen affine Q4 codes/scales/biases into dense FP16 weights; do not replace them with the original dense pretrained model. It transposes all 224 learned matrices into PEFT format and checks the source adapter fingerprint. Original Q4 base: `artifacts/base-mlx4/989aa7980e4cf806f80c7fef2b1adb7bc71aa306-g64/`. Accepted adapter: `artifacts/training/local-study-v1-trial-1/adapter/`. Tensor fingerprint: `70ef5f920cac164fa4a2c35c3b0c670efa812ea393ee9e2def92adac042d5144`. Reconstructed dense weights are omitted from the results ZIP because they are reproducible from these inputs.

## Training provenance and demonstration

Training used original pinned s1K, seed-42 eligibility, complete 896 training rows and 100 held-out development rows; see `configs/single_run_decision.json`, `manifests/training_freeze.json`, `docs/training_protocol.md` and `docs/first_trial_review.md`. Historical two-trial scheduling instructions are superseded by that accepted single-run decision. Do not start a new training run as part of the handoff.

The live illustrative Mac demo is bounded and separate from the benchmark:

```bash
.venv-mlx/bin/python scripts/demo.py --question 'What is 17 times 23?' \
  --condition D256 --answer-budget 128 --timeout 90 --output results/demo/live-mac.json
```

It verifies the original Mac freeze and adapter, uses the native MLX Q4/FP32 backend, and labels the output as illustrative and numerically different from final GPU inference. Metal must be available to the process. A timeout is retained as a demo failure, not silently retried or counted as a benchmark score. `demo_results.py` is CPU-only replay, explicitly not new inference. See `docs/demo_notes.md` for the short presentation and examples.

## Local archive and source history

`python3 scripts/package_handoff.py` builds `output/s1-project-handoff.zip` from an explicit allowlist, preserving the verified result archive and original conversion inputs. `docs/handoff.json` lists every included file's SHA-256; the ZIP's own checksum is outside it. Temporary files, environments, credentials, caches and the Git working directory are excluded. Its included `source-history.bundle` allows an offline clone that contains the pinned experiment commits. To use it, clone that bundle into a fresh folder, then copy the handoff's curated working-tree files/assets over the clone; this restores the final downstream report while retaining pinned history for audit.

The package contains public benchmark/model artifacts for the project team's local handoff, not a new redistribution grant. Review upstream terms before sharing it publicly. Source code and aggregate reports are on GitHub, while weights, raw datasets, full traces and the local handoff are ignored. No project-wide release license has been selected. Inspected Qwen/s1K metadata declare Apache-2.0; inspected MATH-500 metadata does not declare a license. [License metadata](license_metadata.json) records what was actually inspected. There is no venue submission or tagged model release.
