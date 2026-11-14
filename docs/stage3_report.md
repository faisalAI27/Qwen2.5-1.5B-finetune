# Stage 3: measured base-model smoke test

Completed on 2026-11-14, using only the local Apple M1 MacBook Air (8 GB unified memory, 7 GPU cores). This is a technical development test, not an accuracy evaluation.

## Verified checkpoint and execution

The pinned `Qwen/Qwen2.5-1.5B-Instruct` checkpoint was downloaded and its complete upstream SHA-256 verified: `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`. The weight file is 3,087,467,144 bytes. Provenance, config and tokenizer files reside beside it under ignored `artifacts/base/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/`.

Command: `.venv-inference/bin/python scripts/complete_baseline_smoke.py --prefix stage3-local-v1`, run natively with MPS access. The workflow completed successfully, in separate processes for short, main and reload runs. The model has 1,543,714,304 parameters and loaded on `mps:0` as FP16, without quantization or an adapter. Model load times were 10.12, 11.51 and 13.83 seconds respectively.

| Run | Questions | Conditions × repetitions | Output limit | Successful generations |
|---|---:|---:|---:|---:|
| Short | 1 development | 2 × 2 | 64 | 4/4 |
| Main | 5 development | 2 × 2 | 512 | 20/20 |
| Fresh load | 1 development | 2 × 2 | 512 | 4/4 |

All 10 main within-process A/B repeat pairs had identical generated token IDs. All four comparable records across fresh model loads also matched exactly. A/B weight identities, question IDs/order, rendered prompt hashes and tokenizer revisions were checked. Raw outputs, generated IDs, effective generation settings, timings, termination/extraction status and environment/code hashes are saved in `results/dev_baselines/stage3-local-v1-{short,main,reload}/`. The small tracked export is [`baseline_smoke_summary.json`](baseline_smoke_summary.json).

## Observed behavior and limitations

| Main run, including both repetitions | A: direct | B: reasoning |
|---|---:|---:|
| Generations | 10/10 | 10/10 |
| Extracted final answers | 8/10 | 0/10 |
| Output-cap events | 2/10 | 10/10 |
| Timeouts | 0 | 0 |
| Mean generated tokens | 113.4 | 512 |
| Median generation seconds | 2.424 | 51.937 |

The output parser explicitly marks unfinished outputs as `no_final_answer`. Extraction success is not correctness; no gold answers were loaded and no accuracy was calculated. These provisional limits are unsuitable for judging final reasoning quality. Reasoning and answer phase allowances will be selected on development data later.

The maximum post-generation sampled MPS driver allocation in the main run was 3,468,754,944 bytes (approximately 3.23 GiB). These samples are not exact peak VRAM; MPS and process RSS overlap on unified memory. These timings mix lengths and compilation/cache effects and do not establish a controlled prompt speed comparison.

Transformers warned that its unused default `max_length=20` and explicit `max_new_tokens` were both present. It explicitly selected `max_new_tokens`; saved 512-token outputs confirm the intended allowance. The effective settings and warning are retained rather than modifying generation code midway through the runs.

The project's 36-distribution inference dependency closure passed its consistency check; unrelated inherited global package conflicts remain documented in [`baseline_protocol.md`](baseline_protocol.md). The latest full unit suite before these runs passed 55 tests. This baseline success does not establish full-sequence training feasibility.

## Progression

Stages 3 and 4 acceptance gates are satisfied: actual reproducible A/B generation, explicit extraction failures, and exhaustive exact-format/mask validation on all 996 eligible examples. The approved length policy is complete examples with no truncation; no context exclusions have been made. Stage 5 begins with one complete-example backward profile before a 20-example pilot, then longer complete-sequence profiles. MATH-500 remains sealed for scoring.
