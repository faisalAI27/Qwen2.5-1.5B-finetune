"""Illustrative local question demo using the actual frozen final adapter."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.budget_forcing import CONDITIONS, condition_budget, run_structured
from s1small.experiment import verify_freeze
from s1small.generate import build_prompt
from s1small.readout import extract_readout
from s1small.mlx_decode import load_decoder, s1_prompt, token_protocol, unstructured_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="What is 17 times 23?")
    parser.add_argument("--condition", choices=CONDITIONS, default="D512")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads((root / "manifests/evaluation_freeze.json").read_text())
    verify_freeze(root, freeze)
    training = json.loads((root / "manifests/training_freeze.json").read_text())
    baseline = args.condition in ("A", "B")
    decoder, tokenizer, _ = load_decoder(root, training["training_recipe"], None if baseline else root / freeze["final_adapter_directory"])
    config = freeze["protocol"]
    prefix = config.get("answer_readout_prefix", "")
    decoder.prefill_step_size = config["prefill_step_size"]
    templates = json.loads((root / "configs/prompts/baseline_v1.json").read_text())
    prompt = build_prompt(args.question, args.condition, templates, tokenizer) if baseline else s1_prompt(args.question, tokenizer)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if baseline:
        result = unstructured_answer(decoder, ids, config["natural_reasoning_ceiling"] + config["answer_budget"],
                                     config["absolute_safety_ceiling"], config["timeout_seconds"])
    else:
        result = run_structured(decoder, ids, condition_budget(args.condition, config["natural_reasoning_ceiling"], config["answer_budget"],
                                config["absolute_safety_ceiling"]), token_protocol(tokenizer, decoder.eos_ids, prefix), config["timeout_seconds"])
    result.update(extract_readout(result["answer_text"], "" if baseline else prefix))
    result.update(question=args.question, condition=args.condition, scope="illustrative_demo_not_benchmark")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
