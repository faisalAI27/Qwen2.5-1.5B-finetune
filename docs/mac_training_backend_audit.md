# Mac training backend preflight audit

Date: 2026-11-14. This is source inspection while the checkpoint transfers, not a successful training measurement or a backend change. PyTorch/PEFT remains the first pilot candidate; MLX is a resource-driven fallback to validate if required.

The audited MLX-LM repository commit is `56230fa02b8ff8330daf26dc95c28480423ed5fe`. A future installed release must be checked independently; this inspection does not claim main is installed.

## Format and masking implications

The [LoRA guide](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/LORA.md) supports Qwen2, LoRA/QLoRA and prompt masking for chat/completion data. A raw-text dataset with prompt masking is not supported. Use the audited chat serialization or pretokenized data with an explicit loss mask; avoid an extra EOS appended by a generic text-dataset path.

The [dataset implementation](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/tuner/datasets.py) derives a completion offset from the tokenized prompt prefix. That boundary must match the Stage 4 prefix audit, not an assumed character offset.

The [trainer](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/tuner/trainer.py) truncates sequences above its maximum and pads batches before shifted next-token loss. Its offset/end-length mask needs a specific last-padding-position test against our reference labels before adoption: the inspected code uses an inclusive end comparison with raw sequence lengths after shifting. Do not assume that a prompt-mask flag guarantees the same loss as our `-100` label reference. Reject overlength examples before an iterator can truncate them. Verify gradient-accumulation remainder handling for the intended iteration count.

## LoRA parameter correspondence

The inspected [LoRA layer](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/tuner/lora.py) directly multiplies its low-rank update by `scale`. Therefore the proposal's PEFT rank 16 / alpha 32 corresponds to **MLX scale 2.0**, not 32.0, when matching the ordinary alpha/r convention. Dropout remains 0.05. Validate the effective scaling numerically before translating adapters or settings.

The [adapter utilities](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/tuner/utils.py) can adapt many layer types if keys are omitted. Explicit attention keys and all intended layers are necessary to match this study. Saving an MLX adapter does not establish PEFT format compatibility; clean reload and generation tests remain mandatory.

The [Qwen2 implementation](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/models/qwen2.py) defines q/k/v/o attention projections and supports tied word embeddings. Its original weights can be loaded locally; any quantized conversion must record source revision, exact conversion settings, hashes and framework versions.

## Hardware decisions still pending

The initial source inspection above preceded model computation. Actual PyTorch measurements and failed pilots are now in [`hardware_pilot.md`](hardware_pilot.md). Keep complete examples and verify stable full-model gradients before any further training. If quantization or another backend is necessary, record the measured reason and use matched inference settings for final comparisons. The Stage 3 inference smoke does not prove LoRA feasibility.

## Released MLX backend checked on 2026-11-15

Installed **MLX 0.32.4**, **mlx-metal 0.32.4**, **MLX-LM 0.32.2**, and SentencePiece 0.2.3 in the separate `.venv-mlx` with shared system dependencies; no global package was changed. The 36-distribution runtime closure has no requirement conflicts. See `requirements-mlx-lock.txt`; full training verification is still pending.

The MLX-LM wheel SHA-256 from [PyPI release metadata](https://pypi.org/pypi/mlx-lm/0.32.2/json) is `5c0991248cfff5872503d74f466d2532dd8cb7fb3670cb9be7a06d7305dadc04`. The installed trainer, dataset, LoRA, adapter utility, loader and Qwen2 source files exactly match the checksum-verified release wheel; individual hashes are exported in `mlx_release_audit.json`. This is a separate audit from the earlier main-branch commit.

The release still uses direct LoRA scale, implicit sequence truncation in its generic iterator and the inclusive shifted end-mask described above. Our runner uses `TrainingCorpus`, complete original token IDs and the exact Stage 4 `-100` labels instead. Prompt, end-of-answer and first-padding-target behavior are tested explicitly. No upstream text-dataset or automatic-truncation path is used. Attention adapters target all 28 layers with rank 16, scale 2.0, dropout 0.05 and FP32 trainable parameters, matching the intended alpha/r convention.

Three native GPU tests passed: algebraic scaling, dense/chunked actual-Qwen loss and adapter-gradient equivalence, and prompt/final-padding masking. Checkpointed integer targets require an explicit `stop_gradient`; the initial test exposed that requirement and the corrected tests passed. These tests do not establish pretrained-model feasibility.

The first unquantized original-BF16 MLX profile fit the GPU memory cap (peak active allocation 4,308,443,624 bytes), but was rejected because its independently observed adapter norm was 7.529009056447914e16. No optimizer update occurred. Thus merely changing frameworks did not satisfy the numerical gate. A subsequent diagnostic promotes activations/compute to FP32 while preserving the same stored BF16 weights and parameter names; it remains a measured feasibility experiment, not a selected final recipe. The configured preclip-norm abort threshold of 1e6 is a deliberately generous technical sanity guard, over two orders above the shortest PyTorch norm but far below the observed extreme trials. It is not a benchmark metric or a correctness guarantee for smaller norms.
