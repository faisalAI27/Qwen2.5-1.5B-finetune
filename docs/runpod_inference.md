# RunPod vLLM generation and persistent results

`notebooks/runpod_inference.ipynb` now uses **vLLM** for generation. The former MLX notebook is retained as `notebooks/runpod_mlx_inference.ipynb`. The new notebook downloads the existing ZIP from the user's [public Drive link](https://drive.google.com/file/d/1e0G7nG1B8c6G4CKzw6-U69my-x7GJfuD/view?usp=sharing) with pinned `gdown==5.2.0` and verifies SHA-256 `fbe767cf2ed584134ce822204ae46e70461b7000cc01b51bcccb84f2eba3a926` before extraction. Anonymous download and available Drive quota are required. The web reader could not inspect the shared file during preparation; the notebook verifies its content on RunPod. Manual upload of the same ZIP to `/workspace/nlp-colab-assets.zip` is supported if Drive download fails.

## Pod configuration

Use one **RTX 4090 24 GB**, **On-Demand**, an official RunPod PyTorch template with **Python 3.11 or 3.12**, Linux and JupyterLab. Allocate about **25 GB container disk** and **50 GB persistent storage at `/workspace`** for the CUDA environment, reconstructed model, cache, outputs and backups. The [pricing page](https://www.runpod.io/pricing), checked November 17, 2026, listed $0.89/hour for RTX 4090; actual deployment and storage quotes apply. No Pod has been rented by this handoff. Speed and full-run cost must be measured.

Deploy in the console, then use **Connect → Jupyter Lab** ([documentation](https://docs.runpod.io/pods/connect-to-a-pod)). Upload the updated notebook into `/workspace`; the ZIP downloads from Drive. Setup pins a new launcher-code commit and independently checks out the original source commit recorded in the ZIP. New vLLM files are added alongside the original sources. Both original bundle hashes and launcher hashes are verified.

## Conversion and numerical changes

vLLM cannot directly read the MLX packed checkpoint or its adapter layout. CPU conversion therefore reconstructs each frozen Q4/group-64 matrix by `scale * code + bias`, rounds to FP16, and streams it to indexed Hugging Face safetensors shards. It **does not substitute the original full-precision base checkpoint**. All 224 learned FP32 adapter tensors are transposed into PEFT layout, with rank 16 and alpha 32 preserving scale 2. Serialized transposes must round-trip exactly and match the completed training fingerprint. Input/output hashes and maximum FP16 rounding error are saved. Conversion is resumable, bounded in memory, and does not create an optimizer.

The [MLX LoRA implementation](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/tuner/lora.py) defines source orientation and scaling. Generation uses [vLLM's async engine](https://github.com/vllm-project/vllm/blob/v0.11.2/examples/offline_inference/async_llm_streaming.py). The pinned stack has vLLM 0.11.2, Torch 2.9.0, Transformers 4.57.1 and NumPy 2.2.6 in a separate environment; do not install MLX/Colab requirements there.

This is an **independent numerical replication**: MLX FP32 computation changes to FP16 base/LoRA computation, with native prefill, CUDA graphs, prefix caching and concurrent requests. Tokens and scores can differ. The logical reasoning/answer state machine, real stop-token accounting, Wait behavior, answer prefix, token allowances, held-out IDs and mathematical grader remain fixed. A reference checks all 15 development prompt variants and input token IDs before conversion. Eight independent prediction state machines feed a single vLLM async engine for continuous batching; worker threads never call the engine directly.

## Run cells in order

1. Setup downloads/verifies the ZIP, clones pinned code, installs the isolated environment, restores matching state and checks source/model/prompt identity.
2. **Convert** the Q4-derived dense model and PEFT adapter once. Later calls verify the completed export.
3. **Smoke** checks a forced known stop for exact token accounting, then generates a real bounded held-out answer and verifies unchanged exported files. Empty readout, timeout, wrong stops or identity mismatch prevents evaluation.
4. **Development** generates five held-out questions × nine conditions = 45 outputs. Inspect aggregate tokens per wall second before starting a full paid benchmark. Per-output latency includes concurrent scheduling and phase transitions and is not isolated-request latency.
5. **Freeze** requires all 45 outputs, observed readout, exact runtime and unchanged grader. Only then can **test** open the pinned benchmark and generate all 4,500 outputs. Development ID: `runpod-vllm-dev-v1`. Test ID: `math500-runpod-vllm-v1`. Do not merge these predictions or timing samples with Mac, Colab or MLX RunPod runs.
6. **Grade/analyze** use the original grader and analysis code. Partial coverage stays explicit. Download results and checkpoints before deleting the Pod.

## Saving and resuming

Workspace: **`/workspace/nlp-runpod-vllm`**. Environment: `/workspace/nlp-vllm-env`. Checkpoints: **`/workspace/nlp-vllm-state`**.

Every completed prediction is saved immediately in atomic, fsynced `slots/*.json` and durable `attempts/*.json`. An fsynced `outputs.jsonl` logs attempts. At stage completion/interruption, `predictions.jsonl` is regenerated from authoritative slots, including explicit pending/failure statuses. Completed outputs are never rerolled; infrastructure-only retries are bounded at two attempts.

Raw benchmark results are under `results/test/math500-runpod-vllm-v1/`. Grading writes `evaluated.jsonl`; analysis writes JSON and PNG/SVG plots. Logs, GPU samples, package/hardware metadata, export fingerprints and the evaluation freeze are retained. Mutable-state checkpoints run every five minutes and after each stage, keeping the last two. A separate identity binds the state folder to exact asset and launcher hashes. The notebook also refreshes **`/workspace/nlp-runpod-vllm-results.zip`** after checkpoints/stages, excluding model weights and environments.

These are persistent-volume files, not an automatic Google Drive upload. A public read-only download link does not provide Drive write access. Download the results ZIP and a checkpoint to the Mac or manually copy them to Drive. A same-volume checkpoint protects against process interruption, not volume loss or deletion. A network volume persists independently; a Pod volume is deleted with the Pod. [Storage documentation](https://docs.runpod.io/pods/storage/types).

Resume with the same workspace, launcher commit, environment and frozen GPU/driver/runtime. Runtime changes reject a mixed run. The helper prints newly appended output and samples GPU activity while inference runs. Closing Jupyter or finishing a cell does not stop billing. After backup, stop the Pod in the console; retained storage remains billed. [Pod management](https://docs.runpod.io/pods/manage-pods).

## Validation

Local tests cover Q4/group reconstruction, BF16 decoding, independently readable safetensors, exact PEFT transposes/fingerprints, resumable dense export and index integrity, tamper rejection, async stop/Wait accounting, deadline aborts and context bounds. The real 224-tensor adapter passed conversion/fingerprint checks. Four two-row samples from the real base matched pinned MLX CPU dequantization exactly before the FP16 cast. Full model loading, CUDA LoRA kernels, concurrency speed and GPU smoke still require the allocated RunPod GPU. The Mac run continues independently.
