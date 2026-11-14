"""Inspect the pinned architecture and LoRA targets on meta tensors, without weights."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.training import attach_lora


def main():
    import torch
    import yaml
    from transformers import AutoConfig, AutoModelForCausalLM
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/train_pilot.yaml").read_text())
    source = json.loads((root / "configs/sources.json").read_text())["model"]
    directory = root / "artifacts/base" / source["revision"]
    architecture = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(architecture, dtype=getattr(torch, config["dtype"]), trust_remote_code=False)
        model, inspected = attach_lora(base, config)
    if any(parameter.device.type != "meta" for parameter in model.parameters()):
        raise ValueError("Architecture inspection unexpectedly allocated real parameter tensors")
    report = {"scope": "architecture_only_no_weight_loading", "source": source,
              "weights_loaded": False, "training_performed": False,
              "base_parameter_dtype": config["dtype"],
              "decoder_layers": architecture.num_hidden_layers, "vocab_size": architecture.vocab_size,
              "lora_rank": config["lora_rank"], "lora_alpha": config["lora_alpha"],
              "lora_dropout": config["lora_dropout"], **inspected}
    destination = root / "results/training_pilot/architecture.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    (root / "docs/lora_architecture.json").write_text(json.dumps({key: value for key, value in report.items() if key not in ("target_module_names", "trainable_names")}, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("target_module_names", "trainable_names")}, indent=2))


if __name__ == "__main__":
    main()
