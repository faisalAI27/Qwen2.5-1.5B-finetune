# Local hardware and feasibility record

Inspected on 2026-11-14 (Asia/Karachi). The user requires this computer to be used for the project. No remote GPU is assumed.

## Measured specifications

| Item | Observed value |
|---|---|
| Computer | MacBook Air, MacBookAir10,1 |
| Architecture | Apple silicon, arm64 |
| Chip | Apple M1 |
| CPU | 8 cores: 4 performance, 4 efficiency |
| GPU | Integrated Apple M1, 7 cores, Metal supported |
| Memory | 8 GB unified memory, shared by CPU, GPU and macOS |
| Operating system | macOS 26.6.2, build 25G83 |
| Data volume capacity | 228 GiB reported by `df -h .` |
| Available storage | Approximately 16 GiB at the Stage 5 recheck (initially 12 GiB; varies with system usage) |
| Python | CPython 3.13.3, arm64 |
| PyTorch | 2.9.1 |
| Transformers | 5.10.1 |
| Datasets | 4.8.5 |
| Accelerate | 1.13.0 |
| PEFT / MLX / MLX-LM | Project-local PEFT 0.21.3; separate MLX 0.32.4 / MLX-LM 0.32.2 environment (2026-11-15) |
| NVIDIA CUDA | Unavailable |
| PyTorch MPS | Built and available outside the execution sandbox |
| MPS recommended maximum working set | 5,726,633,984 bytes, approximately 5.33 GiB |

The MPS working-set value is a runtime recommendation, not dedicated VRAM or an allocation guarantee. These package versions describe the existing global environment; they are not a validated project dependency lock. Device identifiers and serial numbers are intentionally excluded.

## Checks actually performed

- Read `system_profiler SPHardwareDataType SPDisplaysDataType SPSoftwareDataType -json`, `sw_vers`, `uname -m`, and `df -h .`.
- Inspected installed Python package metadata without installing anything.
- Checked PyTorch CUDA/MPS availability. MPS was unavailable inside the sandbox, so the check was repeated outside it with approval.
- Allocated a four-element tensor on MPS, added it to itself, and copied the result to CPU. Observed `[2.0, 2.0, 2.0, 2.0]`.
- Read `torch.mps.recommended_max_memory()` outside the sandbox.

These initial checks did not load a model. Subsequent actual model and training-profile measurements are recorded below. Swap was unavailable under the initial sandbox; a later native snapshot is documented at the end.

## Implications for the study

1. Local code, unit tests, data inspection, analysis and report preparation can proceed on this computer. Model inference and training require staged feasibility checks.
2. The earlier provisional 16-24 GB GPU assumption does not describe this machine. CPU and GPU share the same 8 GB memory pool with the operating system and other apps.
3. Complete 9,109-token backward feasibility and a stable 20-row training pilot are verified on the accepted MLX path. The entire full-data experiment remains in progress. Earlier PyTorch failures remain part of the evidence; inference or pilot success alone does not establish a completed study.
4. PyTorch MPS is available. MLX 0.32.4 / MLX-LM 0.32.2 are now installed in the separate project environment and validated for the source-traced QLoRA/FP32 path. The backend and quantization change is documented in `hardware_pilot.md` and `mac_training_backend_audit.md`.
5. Keep the planned Qwen2.5-1.5B-Instruct identity. If quantization is necessary, trace the conversion to a pinned source revision and use matching precision/quantization for comparable base and adapted conditions.
6. Approximately 16 GiB of free storage was reported at the Stage 5 recheck. A suggested working target is 30-40 GiB free for the environment, model/cache files, checkpoints and operating-system headroom; this is a planning allowance, not a measured requirement or guarantee. Calculate storage needs before downloading; do not delete user files automatically. Additional storage does not increase unified memory.
7. Test one complete example before increasing workload. Measure peak memory, elapsed time and eligibility at each step. Preserve answer targets and disclose any excluded examples; a short-context pilot does not establish full-context feasibility.
8. The core benchmark requires 9 conditions x 500 questions = 4,500 predictions. Completion time is unknown until local generation is benchmarked. No runtime estimate is asserted here.

## Reporting requirement

Include these specifications in the final report's experimental setup, and refresh the runtime/package/storage observations when experiments actually run. Add measured training time, peak memory, inference speed, context limit, quantization, adapter settings and final eligible sample counts from saved run logs. Distinguish this initial hardware inspection from experimental measurements.

## Official references checked

- MLX-LM: https://github.com/ml-explore/mlx-lm
- Planned model configuration: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json

## Stage 2 observation (2026-11-14)

The new isolated data environment was validated with Transformers 5.10.1, Tokenizers 0.22.2, PyArrow 22.0.0, PyYAML 6.0.2 and Jinja2 3.1.6, without Torch. See `requirements-data-lock.txt` for the full runtime snapshot; this does not replace or validate the global installation for training. The Stage 2 isolated environment report recorded 16,039,608,320 free disk bytes (approximately 14.94 GiB). Source data/tokenizer files occupy 18,498,049 bytes. No model memory or inference/training throughput measurement was made.

## Stages 3–5 observations (2026-11-14)

The verified 1,543,714,304-parameter base model runs in FP16 on MPS. All 28 development-only smoke generations succeeded, and within-run/fresh-load token-ID comparisons matched. The main run's maximum post-generation sampled MPS driver allocation was 3,468,754,944 bytes. Its reasoning-prompted outputs all reached the provisional 512-token cap; median generation time was 51.937 seconds. See [`stage3_report.md`](stage3_report.md) for denominators and limits; this is not an accuracy result or a controlled speed benchmark.

The first BF16 LoRA standard-loss profile on the complete shortest eligible example (712 input tokens, 645 supervised targets) failed at full-vocabulary cross entropy. The exception reported 5.07 GiB allocated against the 4.80 GiB configured allowance. Subsequent memory optimizations fit longer examples, but the PyTorch pilot had extreme gradients and was stopped. All failures are retained.

On 2026-11-15, the accepted path became **MLX QLoRA (4-bit affine, group 64), original-source conversion, FP32 activations/adapters**. The 20-row technical pilot completed five stable updates in 233.666 seconds, 115.426 input tokens/second, with 2,993,071,108 peak active MLX bytes. A fresh process reloaded identical adapter tensors and generation. The complete 9,109-token longest eligible sequence passed backward in 114.517 seconds, with 5,832,500,352 peak active bytes. All 996 eligible examples remain covered, without truncation/context exclusions; training will use a 9,216-token ceiling. See [`hardware_pilot.md`](hardware_pilot.md). These values establish feasibility, not final performance. MLX's memory limit is a guideline that can be exceeded using RAM/swap; active allocator peaks are not dedicated VRAM and overlap with RSS.

## Full-training host recheck (2026-11-15 local date)

`results/environment/stage6-host.json` confirms the M1/8 GB/7-core GPU host, macOS 26.6.2 and CPython 3.13.3; serial numbers are excluded. The native swap snapshot in `results/environment/stage6-swap-snapshot.json` reports 7,168 MiB total and 5,599.56 MiB used. This is a system-wide snapshot including other applications, not training-attributable memory. Approximately 14.8 GiB was free during the continuation check; read current free space with `scripts/status.py`. No additional storage is currently required to start the queued stages, subject to their disk guards and changing system usage. Keep the Mac plugged in and awake for the long local runs.
