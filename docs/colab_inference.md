# Colab T4 inference handoff

The Mac continues its own frozen benchmark. Training is finished and stays on the Mac; neither this handoff nor the notebook starts training. The CUDA route is prepared but **not tested on an allocated Colab GPU yet**. Its smoke and full held-out development checks must pass first. Speedup is not assumed.

## T4 GPU, code on GitHub, assets on Drive

Choose **T4 GPU**, not the v5e TPU. Pinned [MLX 0.32.4 supports CUDA 12 on Linux](https://github.com/ml-explore/mlx/blob/v0.32.4/docs/src/install.rst), requiring NVIDIA SM >= 7.5, driver >= 550.54.14, CUDA >= 12, glibc >= 2.35 and Python >= 3.10. Our NumPy pin additionally requires Python >= 3.11. The notebook checks actual hardware and packages. This reuses the same packed four-bit/group-64 base, MLX adapter and FP32 activation wrapper without a Torch/PEFT conversion. The TPU would need another backend.

GitHub receives the small source repository: Python, notebook, configs and hash manifests. At inspection it was about **1.55 MiB tracked**, with the largest file under 0.5 MiB. GitHub [blocks ordinary Git files over 100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Existing ignores keep environments, raw data, weights, optimizer checkpoints and prediction logs out of Git. No remote push is performed automatically.

The assets ZIP stays outside GitHub. It contains the exact converted base (about 880 MB including tokenizer files), the approximately 17.5 MB adapter, required small data/provenance, and the 45-output Mac development reference. It omits the original 3 GB checkpoint, virtual environments, optimizer states, live Mac benchmark outputs and benchmark answers. The total is roughly 0.9 GB; use the build summary for the exact size.

## Build and launch

1. Commit the prepared code locally, then run `python3 scripts/build_colab_bundle.py`. This creates `artifacts/colab/nlp-colab-assets.zip` and its SHA-256 file. The bundle records the exact code commit and every required file checksum.
2. Push the code commit to the already configured GitHub repository when ready. The asset ZIP remains ignored. A private repository needs authentication in Colab; credentials do not belong in notebook source or the asset bundle.
3. Upload the assets ZIP to your own Drive at `MyDrive/NLP-Colab/nlp-colab-assets.zip`.
4. Open `notebooks/colab_inference.ipynb` in Colab, choose T4 GPU, and run setup. The repository URL is prefilled from the local origin without URL credentials. Setup checks out the bundle's exact commit, extracts assets into `/content/nlp-colab`, creates an isolated environment, and verifies hashes.
5. Run smoke, all 45 CUDA development generations, then the CUDA freeze. It compares emitted development token segments with the Mac reference and records actual GPU/driver/runtime. Kernel differences may produce different outputs; parity is measured, not promised.
6. Run the 4,500 CUDA benchmark predictions as a **separate replication**, then grade and analyze. CUDA outputs use `math500-colab-cuda-v1`; Mac outputs remain `math500-local-v1`. Never combine the two into a single accuracy denominator or mix their latency samples.

If cloning is inconvenient, `python3 scripts/build_colab_bundle.py --include-code` creates a complete inference bundle instead. Set the notebook's bundle path accordingly. This is still a curated inference export, not the whole Mac environment.

## Session interruptions and results

[Colab's free GPU availability and runtime limits vary](https://research.google.com/colaboratory/faq.html); a free notebook can run for at most 12 hours depending on availability/usage, and can end sooner. The notebook snapshots mutable CUDA state to its dedicated Drive folder every five minutes and after each stage, retaining its last two snapshots. In-flight work or outputs since the last snapshot may need recomputation. It uses normal notebook execution, without timeout-bypass mechanisms.

After interruption, rerun setup, restore the latest snapshot and rerun the unfinished stage. Completed outputs are immutable. A different GPU type/driver/runtime rejects resume under the same frozen identity; retain the old state and use a separate hardware run if needed. Checkpoint checksums and bundle identity prevent mixing source/model versions. Restore preserves newer local outputs and rejects conflicting completed predictions.

Grading/analysis can run without a GPU using the installed pinned stack. Raw evaluated JSONL, analysis, exact denominators and PNG/SVG plots are under `results/test/math500-colab-cuda-v1/`. Save the CUDA freeze and hardware metadata with them. An incomplete session remains partial, never a full benchmark score. The final research report must distinguish M1 training/Metal inference from CUDA inference and explain any numerical differences.

## Why training here and inference there is valid

The model's learned adapter is a saved set of numbers. Inference loads those exact numbers, the identical quantized base and tokenizer, then generates token IDs that decode into answers. Training does not need to be repeated on the target machine. Those answers still need mathematical grading; token generation and extraction are not evidence of correctness. Hardware/backend changes can alter floating-point computations, greedy choices and timings, which is why the CUDA route is declared, validated and recorded independently.
