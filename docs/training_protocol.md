# Training protocol and accepted single-run adapter

Stage 6 uses the measured local MLX QLoRA path in [`hardware_pilot.md`](hardware_pilot.md). The first development trial, its validation, fresh-process reload and matched base validation are complete. Final all-data training has not run and is deferred by the current decision. No MATH-500 score guides this recipe.

## Current decision: use the existing run only (2026-11-16)

The user accepted the first run and instructed us to proceed with that run only. **Evaluate the existing 896-example adapter directly. Do not launch trial 2 or fresh 996-example retraining.** All 100 development examples remain held out from this evaluated adapter. This is a compute-constrained single-candidate study, not a learning-rate optimization or a claim that the unrun alternative is equivalent.

`configs/single_run_decision.json` binds this decision to the completed review. `freeze_existing_run.py` verified complete training/validation, the review's checksums, unchanged training code, final checkpoint/export checksums and absence of benchmark access. It writes the training freeze and an adopted Stage 6 summary with `evaluation_model` set to the actual completed run and **`final_training: null`**. It never invokes training. The original two-rate YAML and original run provenance are preserved as history.

The prior review boundary is resolved for inference/evaluation. The old suspended scheduler is retired, not resumed; its request remains historical and still prohibits trial 2. Continue with `complete_project.py`, which sees the adopted Stage 6 summary and skips its training branch. Native held-out protocol checks must still pass before the separate evaluation freeze and benchmark answers. The paper records 896 actual training examples, a deferred 996-example final pass, and one tested learning rate. The historical review and original design below are retained for audit.

## Historical first-trial review boundary (resolved 2026-11-16)

The user wants to inspect trial 1 and its validation before deciding whether trial 2 justifies its cost. **The first trial completed all 896 training examples and all 100 development targets.** The original scheduler remains intentionally suspended; the training worker has finished. The downstream continuation controller honored `results/runtime/PAUSE`. No second trial, recipe freeze, final retraining or test evaluation may start before the review decision.

`review_first_trial.py` completed fresh-native-process verification and evaluated the identical original-derived Q4/FP32 base on the same 100 held-out target sequences, without an optimizer or parameter updates. Both losses use the same 555,371 supervised targets: base NLL 0.5636226723, adapted NLL 0.4797165291, a 14.8869% relative reduction. All 56 training updates had finite gradients, and adapter tensors/generation matched after reload. Recorded training windows total 64,238.271 seconds (17.844 hours), excluding separately recorded system sleep. Actual reviewed artifacts are `docs/first_trial_review.md` / `.json`; their source checksums and both ordered 100-row validation logs were checked. No benchmark accuracy is implied.

The original paper's 32B MATH-500 accuracy is contextual; our 1.5B held-out teacher-target NLL is a different metric on a different dataset. We will not score MATH-500 to decide whether to run another learning rate. A decision to skip trial 2 must be documented as a compute-constrained single-trial change, not evidence that the unrun learning rate is equivalent. The original two-trial recipe below is preserved as the historical predeclared plan; no active training core/config has been changed.

Review request/status/log: `results/runtime/first_trial_review_request.json`, `first_trial_review.json`, `first_trial_review.log`. Scheduler PID 7894 is intentionally stopped; training worker PID 7897 has finished. The review controller exited with status `awaiting_user_decision`. Do not mistake this boundary for failed training or resume it automatically. Keep both the request and `PAUSE` marker until an explicit decision about the next run.

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

## Historical full-training command (superseded)

Run from this project root in a native GPU-enabled terminal:

```bash
caffeinate -i .venv-mlx/bin/python scripts/run_development.py --prefix local-study-v1
```

The orchestrator runs/resumes each declared trial, verifies it in a fresh process, selects by complete development loss, freezes training settings, then starts and verifies fresh final training. Repeating the same command retains completed verified runs and resumes incomplete ones. It never opens MATH-500. Per-example progress is printed during computation; raw records and attempts are under `results/training/`, adapters/checkpoints under `artifacts/training/`.

Use the provided loader for all inference: the adapter is **MLX format** and depends on the identical converted base and **FP32 activation wrapper**. A generic BF16 MLX load or a PEFT loader does not reproduce this recipe. No merge/interchangeability claim is made. Source and conversion provenance are packaged beside the adapter; hardware measurements must accompany final results.

## Held-out inference-protocol development

Budget/marker/answer-allowance decisions use the **selected 896-row development adapter**, whose 100 development rows remain held out. Under the current single-run decision, the same accepted 896-row adapter is used for the frozen benchmark; no all-data adapter exists. Training settings freeze before fresh final retraining; inference settings freeze separately before test, using the development adapter even if final training has already finished. This preserves the information boundary without relying on a development set that the final adapter has seen. Final-adapter technical reload checks do not choose settings or calculate accuracy.
