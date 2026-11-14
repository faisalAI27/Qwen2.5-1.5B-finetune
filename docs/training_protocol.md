# Development tuning and fresh final training

Stage 6 uses the measured local MLX QLoRA path in [`hardware_pilot.md`](hardware_pilot.md). Full training has not been completed yet. No MATH-500 score guides this recipe.

## Predeclared experiment

`configs/train_development.yaml` specifies **two full trials**, learning rates **5e-5 and 1e-4**, each one epoch over the same **896 eligible training examples**. Both initialize fresh from the same source-traced 4-bit/group-64 affine conversion and the same seed-42 adapter initialization. Keep every complete sequence, with a 9,216-token ceiling, FP32 activations/adapters, rank 16 / alpha 32, attention q/k/v/o in all 28 layers, and dropout 0.05.

Micro-batch size is one and accumulation is 16. The objective is the token-weighted mean over all supervised assistant targets within each window, including a possible remainder window. Use explicit **bias-corrected AdamW**, betas (0.9, 0.999), epsilon 1e-8, weight decay zero, constant learning rate, global gradient clipping at 1, and the documented generous preclip-norm sanity abort guard. This differs from the MLX default bias correction used by the technical pilot; the complete runner smoke and exact-resume test validated the explicit optimizer.

There is no epoch or seed search. Two learning rates are a bounded grid justified by the measured local cost; no claim of globally optimal settings is intended. One trial contains 5,066,380 input tokens and 4,926,536 supervised targets. Long-profile throughput was approximately 80–90 input tokens/second, so full trials take **many hours**, with variation from thermal load, other apps and swapping. These are conditional estimates, not completed-run times.

Evaluate each completed trial on **all 100 fixed development examples**, without dropout or optimizer updates. Rank by supervised-token-weighted negative log-likelihood, with an exact tie favoring the lower learning rate. This is appropriate for the mixed proof/answer teacher corpus; it is not mathematical answer accuracy and does not certify teacher correctness. An incomplete or different development denominator cannot win selection.

After both verified trials complete, write an immutable `manifests/training_freeze.json` containing actual losses, selection, source/config/eligibility identities and the initial adapter fingerprint. Initialize a **new** model/adapter, verify that initialization fingerprint, and train one epoch on **all 996 eligible examples** with the selected recipe. Final training contains 5,637,369 input tokens and 5,481,907 supervised targets. No tuned trial adapter is used as its starting point. Freeze the inference/grader/budget protocol separately on development before test evaluation.

## Durable execution

`train_full.py` writes model/data/config/code hashes and Git commit in each run manifest. Every optimizer update produces an atomic checkpoint containing adapter weights, optimizer state, cursor/counters, checksums and the last step record. Only this run's two most recent complete checkpoints are retained; raw loss/progress logs remain. Checkpoint/adapter exports are staged before rename. A failure retains its last durable cursor and a trace; no example is silently dropped.

Each example's dropout RNG seed is derived from SHA-256 of seed, epoch and immutable ID. The same shuffled order and dropout stream can therefore resume without depending on wall time or ambient RNG use. Native tests compared interrupted/resumed optimizer and model tensors exactly against uninterrupted training with dropout, and rejected a corrupted checkpoint. Code/data/config changes are rejected on resume rather than silently changing a running experiment.

The complete runner smoke trained four complete original examples (22,097 input tokens, maximum 8,165) with two bias-corrected AdamW updates, evaluated two development examples, saved checkpoints, reloaded identical adapter tensors/generation in a fresh process, and exercised the completed-checkpoint resume path. It is explicitly `technical_smoke`, and cannot be selected as a full development trial. Evidence: `results/training/full-runner-smoke-v1/`; tracked export: `full_runner_smoke.json`.

## Commands

Run from this project root in a native GPU-enabled terminal:

```bash
caffeinate -i .venv-mlx/bin/python scripts/run_development.py --prefix local-study-v1
```

The orchestrator runs/resumes each declared trial, verifies it in a fresh process, selects by complete development loss, freezes training settings, then starts and verifies fresh final training. Repeating the same command retains completed verified runs and resumes incomplete ones. It never opens MATH-500. Per-example progress is printed during computation; raw records and attempts are under `results/training/`, adapters/checkpoints under `artifacts/training/`.

Use the provided loader for all inference: the adapter is **MLX format** and depends on the identical converted base and **FP32 activation wrapper**. A generic BF16 MLX load or a PEFT loader does not reproduce this recipe. No merge/interchangeability claim is made. Source and conversion provenance are packaged beside the adapter; hardware measurements must accompany final results.

## Held-out inference-protocol development

Budget/marker/answer-allowance decisions use the **selected 896-row development adapter**, whose 100 development rows remain held out. The final 996-row adapter is used for the frozen benchmark only. Training settings freeze before fresh final retraining; inference settings freeze separately before test, using the development adapter even if final training has already finished. This preserves the information boundary without relying on a development set that the final adapter has seen. Final-adapter technical reload checks do not choose settings or calculate accuracy.
