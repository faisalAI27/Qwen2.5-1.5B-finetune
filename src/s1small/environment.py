"""Read-only environment inspection; usable without ML packages or a GPU."""

import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


PACKAGES = (
    "torch", "transformers", "tokenizers", "peft", "datasets", "accelerate",
    "huggingface-hub", "trl", "mlx", "mlx-lm", "numpy", "safetensors",
)


def run_command(arguments, notes):
    """Return output or record why an optional system command failed."""
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=20, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        notes.append(f"{arguments[0]} unavailable: {type(error).__name__}")
        return None


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def inspect_torch(notes):
    """Probe backends without allocating tensors or loading a model."""
    report = {
        "import_status": "not_installed", "version": None,
        "cuda_available": False, "cuda_runtime": None, "cuda_devices": [],
        "mps_built": False, "mps_available": False,
        "mps_recommended_working_set_bytes": None,
        "preferred_available_device": "cpu",
    }
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            report["import_status"] = "error"
            notes.append(f"torch dependency missing: {error.name}")
        return report
    except Exception as error:
        report["import_status"] = "error"
        notes.append(f"torch import failed: {type(error).__name__}: {error}")
        return report

    report.update(import_status="ok", version=torch.__version__)
    try:
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(index)
                report["cuda_devices"].append({
                    "index": index, "name": device.name,
                    "total_vram_bytes": device.total_memory,
                })
            report["preferred_available_device"] = "cuda"
    except Exception as error:
        notes.append(f"CUDA inspection failed: {type(error).__name__}: {error}")

    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            report["mps_built"] = bool(mps.is_built())
            report["mps_available"] = bool(mps.is_available())
        if report["mps_available"]:
            if not report["cuda_available"]:
                report["preferred_available_device"] = "mps"
            get_limit = getattr(torch.mps, "recommended_max_memory", None)
            if get_limit is not None:
                report["mps_recommended_working_set_bytes"] = int(get_limit())
        elif report["mps_built"]:
            notes.append("MPS is built but unavailable in this process; sandbox restrictions may hide the GPU.")
    except Exception as error:
        notes.append(f"MPS inspection failed: {type(error).__name__}: {error}")
    return report


def inspect_apple_hardware(notes):
    """Filter system_profiler output so serials and device UUIDs are never saved."""
    output = run_command([
        "system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json",
    ], notes)
    if output is None:
        return None
    try:
        raw = json.loads(output)
        hardware = raw.get("SPHardwareDataType", [{}])[0]
        return {
            "model": hardware.get("machine_model"),
            "chip": hardware.get("chip_type"),
            "cpu_description": hardware.get("number_processors"),
            "unified_memory": hardware.get("physical_memory"),
            "gpus": [{
                "name": gpu.get("sppci_model"),
                "cores": gpu.get("sppci_cores"),
                "metal_support": gpu.get("spdisplays_metal"),
            } for gpu in raw.get("SPDisplaysDataType", [])],
        }
    except (ValueError, IndexError, TypeError, AttributeError) as error:
        notes.append(f"Apple hardware parsing failed: {type(error).__name__}")
        return None


def collect_environment(project_root, include_torch=True):
    root = Path(project_root).resolve()
    notes = []
    disk = shutil.disk_usage(root)
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "system": {"os": platform.system(), "release": platform.release(),
                   "macos_version": platform.mac_ver()[0] or None,
                   "architecture": platform.machine(), "processor": platform.processor()},
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "packages": package_versions(),
        "git_commit": run_command(["git", "-C", str(root), "rev-parse", "HEAD"], notes),
        "apple_hardware": inspect_apple_hardware(notes) if platform.system() == "Darwin" else None,
        "torch": inspect_torch(notes) if include_torch else {"import_status": "skipped"},
        "notes": notes,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this file")
    parser.add_argument("--skip-torch", action="store_true", help="Inspect without importing PyTorch")
    args = parser.parse_args()
    if not args.project_root.is_dir():
        parser.error("--project-root must be an existing directory")
    report = collect_environment(args.project_root, include_torch=not args.skip_torch)
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()

