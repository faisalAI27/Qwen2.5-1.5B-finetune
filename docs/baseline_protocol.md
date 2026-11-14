# Development baseline smoke protocol

Status: native model smoke workflow completed: 28/28 generations, all within-run repeat pairs and four fresh-load comparisons match token IDs. See [`stage3_report.md`](stage3_report.md). This is not a frozen test-evaluation protocol.

## Inputs and identity

- Model and tokenizer: pinned Qwen2.5-1.5B-Instruct revisions in `configs/sources.json`.
- `scripts/download_model.py` retrieves only config, generation config and the single safetensors checkpoint, then copies the Stage 2 tokenizer files. An upstream LFS SHA-256 is mandatory for the weights. A partially transferred file is never published as `model.safetensors`.
- The downloader supports resumable bounded HTTP ranges (at most eight workers, streaming 1 MiB buffers), serial HTTP and Xet. The initial Xet attempt stalled; serial HTTP worked slowly. The range path is in use. It keeps local range checksums for safe resume and verifies the final complete file against the upstream checksum. Local range checksums alone do not prove upstream identity.
- Select the first five eligible original development IDs in manifest order. The runner verifies source file, split and exclusion-config checksums and question digests. It reads only the training dataset's question column; MATH-500 is never opened by this runner.

## Prompts and decoding

`configs/prompts/baseline_v1.json` defines a shared explicit Qwen system message and two user instructions: A requests the final answer only; B requests step-by-step reasoning. Both request a final boxed answer. The same original question appears in both prompts, serialized by the same tokenizer chat template.

`configs/baseline.yaml` contains provisional development settings: MPS, FP16, SDPA, greedy single-beam generation, seed 42, strict deterministic algorithms, 512 maximum new tokens, a 1,024-token input ceiling and a 180-second per-generation deadline. A/B use the same allowance and same loaded base model. No adapter or quantization is attached. MPS CPU fallback is disabled; unavailable MPS or unsupported kernels fail explicitly. Allocation is capped at 90% of the runtime's recommended GPU working set; this is not a guarantee that the operating system will have enough memory.

The token ceiling is a total generation limit for these unstructured baselines. It is not a reasoning-only cap and does not implement budget forcing. The deadline is checked after generation steps; a long single kernel can exceed the requested deadline. Input overflow causes an explicit failure instead of truncation.

## Run order and evidence

1. Run one development question, A/B, with a short 64-token ceiling and two repetitions.
2. If successful, run five development questions, A/B, with the configured 512-token ceiling and two repetitions.
3. Reinitialize the model in another process for the first question at the same 512-token settings, then compare overlapping token sequences across fresh loads.

Each run has a fresh output directory containing the effective configuration and hash, source pins, weight hash, prompt definitions/hashes, ordered development IDs, code hashes, environment, model load details, raw output, token IDs, input/generated token counts, extraction status, timing, termination reason, warnings and sampled memory. An error halts further generation; remaining planned records are explicitly marked not attempted. No question silently disappears.

The process-wide peak RSS and post-generation MPS allocator samples are diagnostic measurements, not per-question peak VRAM. Model load time is separate from generation time. The first generation can include backend compilation; these smoke timings cannot support a controlled A/B latency claim. For unstructured base outputs, reasoning-token counts remain unavailable until a defensible phase-counting policy is validated.

`src/s1small/extraction.py` conservatively reads balanced boxed answers or an explicit final-answer line. Partial boxes and unfinished reasoning are marked unparseable; no trailing number is guessed. Extraction success is not correctness. Gold/reference answers are not loaded by the runner, and no accuracy scores are produced. Grading remains a later development/freeze task.

## Local environment choice

The isolated `.venv` from Stage 2 remains the data environment. A slow redundant Torch wheel download was stopped before installation. `.venv-inference` uses Python's `--system-site-packages` option to reuse this Mac's existing Torch 2.9.1 and Transformers 5.10.1 without modifying global packages. Its core dependency closure is recorded in `requirements-inference-lock.txt`; the header indicates whether model verification has completed. A fully isolated installation can be built from that snapshot when download time permits.

The inherited global installation has unrelated dependency conflicts (telemetry, agent/web tooling, testing packages). Full `pip check` therefore fails in this shared environment. A separate check of the 36-distribution project runtime closure found no requirement conflicts. The native FP16 SDPA probe passed with finite output under strict deterministic algorithms; it does not establish complete model compatibility.
