# Local LoRA feasibility measurements

Stage 5 is active. These are technical training profiles on eligible **training** examples, not final training or accuracy results. No MATH-500 answers or scores have been used. Hardware: Apple M1 MacBook Air, 8 GB unified memory, 7-core GPU, macOS 26.6.2. Runtime: Torch 2.9.1, Transformers 5.10.1, PEFT 0.21.3. See [`hardware.md`](hardware.md) for the full specifications and [`../requirements-training-lock.txt`](../requirements-training-lock.txt) for the checked 39-distribution dependency closure.

## Method

Load the same checksum-verified original base as the baseline. Freeze base weights and attach rank-16, alpha-32, dropout-0.05 LoRA to q/k/v/o in all 28 layers: 112 projection modules, 224 trainable tensors, 4,358,144 FP32 parameters. Micro-batch size is one; decoder gradient checkpointing uses `use_reentrant=False`. All examples retain the complete original reasoning/answer target and assistant-only labels. MPS CPU fallback is disabled; the configured allocation fraction is 0.9 of the runtime recommendation. No memory-limit bypass is used.

`scripts/train_pilot.py --mode profile` performs forward/backward and validates every adapter gradient, without an optimizer update. Each run saves code/config/data/weight hashes, environment, example length, model details, memory samples and success or failure. Sampled allocator/driver peaks are not exact VRAM peaks and must not be added to RSS on unified memory.

## Observations so far

All rows below used the complete shortest eligible training example, `s1k-0548-f8828cbdd584`, 712 input tokens and 645 supervised next-token targets. Each has zero optimizer updates.

| Run under `results/training_pilot/` | Precision | Attention | Loss | Outcome |
|---|---|---|---|---|
| `torch-profile-short-v1` | BF16 | SDPA | Standard full vocabulary | OOM at cross entropy; exception reported 5.07 GiB allocated / 4.80 GiB allowance |
| `torch-profile-short-chunked-v1` | BF16 | SDPA | Checkpointed chunks of 128 tokens | Backward finished; gradient validation failed |
| `torch-profile-short-chunked-bf16-diagnostic` | BF16 | SDPA | Chunked | Finite loss 1.318693; all 224 gradients present, nonfinite gradients detected |
| `torch-profile-short-chunked-fp16-v1` | FP16 | SDPA | Chunked | Finite loss 1.322020; all 224 gradients present, nonfinite gradients detected |
| `torch-profile-short-chunked-eager-bf16-v1` | BF16 | Eager | Chunked | **Passed**: loss 1.314369; all 224 gradients finite, some nonzero; 13.234 seconds forward/backward |

The successful run sampled maximum MPS allocated memory 3,519,767,040 bytes and driver memory 4,658,544,640 bytes. These include sampling/allocator effects; they are not independent measures of total physical usage. Losses across implementations/precisions are not claimed bit-identical; numerical attention differences are expected. The controlled CPU loss/gradient equivalence tests validate the chunked objective itself.

The measured failures support eager attention for the next PyTorch profile. They do not establish a universal SDPA bug or that all other precisions/frameworks fail. Retain all failure logs. The next example, `s1k-0886-e4b2ce9dd745` (2,006 tokens / 1,771 supervised), exhausted memory during the ordinary eager attention softmax in backward. Its run is `torch-profile-2000-chunked-eager-v1`; no optimizer update occurred. A query-chunked retry, `torch-profile-2000-query-chunked-v1`, **passed**: loss 0.845244, all 224 gradients finite with some nonzero, 36.914 seconds, sampled peak MPS allocated 3,912,674,304 bytes / driver 4,600,840,192 bytes. Full 9,109-token corpus coverage is still unproven.

## Exact loss memory change

The standard Qwen causal loss constructs sequence × 151,936 vocabulary logits and FP32 cross-entropy buffers. `chunked_causal_loss` instead computes the decoder hidden states once, then applies the frozen output head and FP32 cross entropy to chunks of 128 token positions, checkpointing/recomputing those projections during backward. It uses the same one-position causal shift, `-100` exclusions and total supervised-token denominator. No tokens are removed and no targets approximated. The helper rejects other model families, a trainable output head and empty targets.

Tests cover prompt/padding masks, single-token and remainder chunks, no-grad evaluation, dense/chunked loss and gradients, plus an actual tiny Qwen2 LoRA model with nonzero A/B matrices and decoder gradient checkpointing. The full suite now passes 61 tests, including the attention checks below. Use `PYTHONPATH=src .venv-inference/bin/python -m unittest discover -s tests -v` to avoid this Mac's recurring hidden editable-link flags.

## Exact attention memory change

`src/s1small/attention.py` registers `s1_chunked` through Transformers' attention and mask interfaces. It preserves the standard eager mask construction, grouped key/value repetition, scaling, FP32 softmax and output dtype. It partitions query positions into 128-token chunks while retaining **all keys and values** for each query. Each chunk is checkpointed so backward does not retain the entire sequence-squared probability tensor. It does not restrict the attention window or truncate context. The underlying model's attention dropout is zero; nonzero attention dropout is rejected by this path. LoRA dropout remains 0.05.

Tests compare causal/padding outputs and q/k/v gradients for several chunk sizes against eager attention, plus actual Qwen model loss/parameter gradients with nested decoder checkpointing. CPU numerical equivalence and the complete 2,006-token MPS profile passed.

## Gradient norm and independent numerical checks

The first 20-example pilot, `torch-pilot20-query-chunked-v1`, stopped before its first optimizer update: every gradient element passed the finite check, but the MPS gradient-norm calculation returned nonfinite. The failure log is preserved. `clip_gradients_cpu_norm` now copies only the small adapter gradients to CPU, computes norms in float64 with stable `hypot`, and applies the usual global clipping factor to the original device gradients. A test covers values whose float32 squared sum would overflow. This is not evidence that all MPS norm calls are incorrect.

The complete shortest-example follow-up `torch-profile-short-cpu-norm-v1` passed, with independently finite norm 2,082.771, maximum absolute gradient 145.363 and clip scale 0.00048013. These large values prompted an independent CPU/MPS test rather than assuming finite implies correct. A separate FP16 query-chunked profile produced nonfinite loss and was retained as failed; BF16 remains the candidate path.

`scripts/probe_mps_training_numerics.py` compared a two-layer random Qwen with the real 12 query / 2 KV heads, 128-dimensional heads and 712-token sequence, LoRA, query/loss chunks and decoder checkpointing against a CPU FP32 reference. Dropout was disabled only for this numerical probe, and both devices used the same rounded base weights. FP32 gradient cosine was 0.999999999999 and relative L2 error 0.00000128; BF16 cosine was 0.999971 and relative L2 error 0.007630. Both passed the predeclared cosine >0.99 / relative error <0.10 criteria. Saved evidence: `results/environment/mps-qwen-gradient-parity.json`. This checks the execution path, not full pretrained-model gradient equivalence.

The restarted 20-row pilot `torch-pilot20-cpu-norm-v2` used complete examples of 712–1,762 tokens, accumulation 4 and five intended updates, with CPU norm clipping and post-update finite-parameter checks. **It was interrupted on 2026-11-15 after one update**, because the first independently measured preclip norm was 5.9396043215150886e35 (maximum absolute gradient 3.727217847725311e34). The finite/clipped update is not accepted as a viable training result. The saved interruption record explains the action; no adapter was exported and no reload was claimed. Finite elements alone were an insufficient acceptance check for this pretrained-model trial. The small random-model parity probe does not override this actual instability.

The next candidate is the separately pinned Apple MLX backend using the same verified original weights. Backend/precision/quantization and masking must be validated again; no claim is made that changing the backend will solve the problem. Any final base/adapted evaluation must use matched backend and weight precision. The observed PyTorch inference smoke remains a development-only artifact.

## Remaining acceptance gates

The remaining gates below were completed on 2026-11-15. Earlier failures remain part of the evidence and are not relabeled as successes.

## Accepted local path: MLX QLoRA with FP32 activations

The original unquantized BF16 MLX profile also had an extreme gradient norm (7.53e16), despite fitting in memory. Promoting activations/compute to FP32 while keeping original BF16 weight storage gave stable norm **0.862760**, but peak active MLX allocation **9,348,822,140 bytes** at only 712 tokens. This exceeds the Mac's physical memory and motivates quantization rather than dropping the dataset's long examples.

`scripts/convert_mlx_base.py` converted the same pinned, upstream-checksummed source locally to **4-bit affine weights, group size 64**, using MLX 0.32.4 / MLX-LM 0.32.2. The checkpoint plus config/tokenizer files occupy **880,169,503 bytes**. All output files have SHA-256 hashes and source/conversion provenance under `artifacts/base-mlx4/989aa7980e4cf806f80c7fef2b1adb7bc71aa306-g64/`. This conversion changes weight values; final base/adapted comparisons must both load this identical conversion with FP32 activations. The earlier FP16 PyTorch smoke is not the final comparison baseline.

The command `.venv-mlx/bin/python scripts/complete_mlx_pilot.py --prefix mlx-stage5-qlora-v1` completed the following sequential native runs:

| Complete input length | Supervised tokens | Forward/backward seconds | Gradient norm | Peak active MLX bytes |
|---:|---:|---:|---:|---:|
| 712 | 645 | 6.071 | 0.812559 | 2,223,727,774 |
| 2,006 | 1,771 | 17.985 | 0.404299 | 2,991,203,372 |
| 6,010 | 5,917 | 66.436 | 0.200147 | 4,965,186,436 |
| 8,862 (longest eligible train) | 8,715 | 111.276 | 0.130318 | 5,698,217,080 |
| 9,109 (longest eligible development) | 8,894 | 114.517 | 0.355702 | 5,832,500,352 |

Every profile had all 224 finite adapter gradients and **zero optimizer updates**. The longest development example was used only for this memory/numerical check, without training or retaining a changed adapter. No MATH-500 scoring occurred. The prospective training ceiling is 9,216 tokens, covering every one of the 996 eligible complete sequences. No context exclusions or truncation are necessary for this path.

The technical pilot used the shortest **20 eligible training** rows (26,971 input tokens), shuffled with seed 42, micro-batch 1 and token-weighted accumulation 4. It completed **five updates in 233.666 seconds**, measured **115.426 input tokens/second**, and peak active MLX allocation **2,993,071,108 bytes**. All update norms were 0.2298–0.5228, all gradients and updated adapter parameters finite. Window losses ranged 0.9514–1.6541; windows contain different examples, so this is not evidence of a monotonic learning curve. The technical pilot used the release's default AdamW bias-correction setting (false); the development/final recipe must record optimizer options explicitly.

The exported adapter directory is 17,460,094 bytes. In a **fresh native process**, all adapter tensor bytes matched SHA-256 `d4ef0e31f581b56ad73a892bae2922371e7d748ed8a99aa8659297fd6f3fac93`, and the 32-token development generation check matched token IDs exactly. These are save/load checks, not accuracy results. Evidence is `results/training_pilot/mlx-stage5-qlora-v1-acceptance.json` and the tracked `training_pilot_summary.json` export.

MLX's `set_memory_limit` is documented as a **guideline**, not a hard allocation cap; it may use RAM and swap beyond the guideline. The actual longest-sequence peaks above are reported, even when above 90% of the GPU recommendation. These are allocator active peaks rather than dedicated VRAM, and must not be added to overlapping process RSS. Free disk was approximately 15 GiB during the latest measurements; it varies with system/swap usage. Long-example profiles establish measured feasibility, not a guarantee for every future batch or application workload.

The ordinary suite ran 68 tests with four native-MLX tests skipped; those four were run separately on the GPU and passed. The Stage 5 acceptance gate is now satisfied. Development tuning and fresh final training remain to be executed; no final adapter or benchmark score exists yet.
