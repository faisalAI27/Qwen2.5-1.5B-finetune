"""Run the verified CUDA pipeline as a separate RunPod hardware replication."""

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256

DEV_RUN = "runpod-cuda-dev-v1"
TEST_RUN = "math500-runpod-cuda-v1"


def runpod_identity(base, launcher):
    return {**base, "provider": "runpod", "runpod_launcher_sha256": file_sha256(launcher)}


def main():
    launcher = Path(__file__).resolve()
    root = launcher.parents[1]
    if not (root / "RUNPOD_ISOLATED_WORKSPACE").is_file():
        raise ValueError("Use the separate exported RunPod workspace, never the active Mac or Colab run")
    from s1small import colab_cuda
    original_signature = colab_cuda.runtime_signature

    @lru_cache(maxsize=1)
    def signature():
        return runpod_identity(original_signature(), launcher)

    # Reuse every original CUDA/model/protocol check. The provider and launcher
    # hash become part of smoke, development and frozen benchmark identities.
    colab_cuda.runtime_signature = signature
    spec = importlib.util.spec_from_file_location("runpod_verified_cuda_runner", root / "scripts/colab_inference.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.DEV_RUN = DEV_RUN
    runner.TEST_RUN = TEST_RUN
    runner.main()


if __name__ == "__main__":
    main()
