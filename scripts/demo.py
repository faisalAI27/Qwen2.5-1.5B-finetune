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
    parser.add_argument("--condition", choices=CONDITIONS, default="D256")
    parser.add_argument("--answer-budget", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.answer_budget <= 1536 or not 0 < args.timeout <= 900:
        parser.error("Use an answer budget of 1-1536 and timeout of (0, 900] seconds")
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads((root / "manifests/evaluation_freeze.json").read_text())
    verify_freeze(root, freeze)
    training = json.loads((root / "manifests/training_freeze.json").read_text())
    baseline = args.condition in ("A", "B")
    decoder, tokenizer, _ = load_decoder(root, training["training_recipe"], None if baseline else root / freeze["final_adapter_directory"])
    config = dict(freeze["protocol"])
    config.update(answer_budget=args.answer_budget, timeout_seconds=args.timeout)
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
    result.update(question=args.question, condition=args.condition, scope="illustrative_demo_not_benchmark",
                  backend="local_mlx_q4_fp32_not_final_gpu_backend", answer_budget=args.answer_budget,
                  timeout_seconds=args.timeout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
