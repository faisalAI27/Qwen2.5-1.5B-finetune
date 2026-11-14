"""Greedy baseline generation with explicit termination and token accounting."""

import platform
import resource
import time
import warnings

from s1small.data import digest
from s1small.extraction import extract_answer
from s1small.model import synchronize


def build_prompt(question, condition, templates, tokenizer):
    if condition not in templates["conditions"]:
        raise ValueError("Unknown baseline condition")
    messages = [{"role": "system", "content": templates["system"]},
                {"role": "user", "content": question + "\n\n" + templates["conditions"][condition]}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def termination_reason(token_ids, eos_ids, limit, timed_out):
    if token_ids and token_ids[-1] in eos_ids:
        return "eos"
    if timed_out:
        return "timeout"
    if len(token_ids) >= limit:
        return "max_new_tokens"
    return "unexpected_early_stop"


def sampled_memory(torch, device):
    result = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)}
    if device == "mps":
        result.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    elif device == "cuda":
        result.update(cuda_current_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return result


def generate_one(model, tokenizer, question, condition, templates, config):
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

    class Deadline(StoppingCriteria):
        def __init__(self):
            self.end = time.perf_counter() + config["timeout_seconds"]
            self.triggered = False

        def __call__(self, input_ids, scores, **kwargs):
            self.triggered = time.perf_counter() >= self.end
            return self.triggered

    started = time.perf_counter()
    prompt = build_prompt(question, condition, templates, tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
    input_count = encoded["input_ids"].shape[-1]
    if input_count > config["max_input_tokens"]:
        raise ValueError("Input exceeds the smoke-test ceiling; refusing silent truncation")
    if input_count + config["max_new_tokens"] > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generation allowance exceeds model context")
    encoded = {key: value.to(config["device"]) for key, value in encoded.items()}
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=config["max_new_tokens"],
        use_cache=True, eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        temperature=1.0, top_p=1.0, top_k=50,
    )
    # Pinned Transformers 5.10.1 fills unset fields from the checkpoint/global
    # defaults. Resolve them now so the saved dictionary describes the run.
    generation, _ = model._prepare_generation_config(generation)
    synchronize(torch, config["device"])
    deadline = Deadline()
    generation_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as observed, torch.inference_mode():
        output = model.generate(**encoded, generation_config=generation,
                                stopping_criteria=StoppingCriteriaList([deadline]))
    synchronize(torch, config["device"])
    generation_seconds = time.perf_counter() - generation_started
    tokens = output[0, input_count:].cpu().tolist()
    raw = tokenizer.decode(tokens, skip_special_tokens=False)
    visible = tokenizer.decode(tokens, skip_special_tokens=True)
    extraction = extract_answer(visible)
    eos = generation.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    result = {
        "condition": condition, "prompt_version": templates["version"], "prompt": prompt,
        "generation_parameters": generation.to_dict(),
        "prompt_sha256": digest(prompt), "input_tokens": input_count,
        "raw_output": raw, "visible_output": visible, "generated_token_ids": tokens,
        "total_generated_tokens": len(tokens), "reasoning_tokens": None,
        "reasoning_tokens_status": "Unstructured base output; no verified per-phase token boundary",
        **extraction,
        "latency_seconds": generation_seconds, "end_to_end_seconds": time.perf_counter() - started,
        "termination_reason": termination_reason(tokens, eos_ids, config["max_new_tokens"], deadline.triggered),
        "warnings": [str(w.message) for w in observed],
        "memory_after_generation": sampled_memory(torch, config["device"]),
        "status": "ok",
    }
    return result


def repeat_comparison(records):
    groups = {}
    for row in records:
        groups.setdefault((row["question_id"], row["condition"]), []).append(row)
    return [{"question_id": question_id, "condition": condition, "observations": len(rows),
             "comparable": all(row["status"] == "ok" for row in rows),
             "token_ids_identical": (all(row["status"] == "ok" for row in rows)
                                     and all(row["generated_token_ids"] == rows[0]["generated_token_ids"] for row in rows))}
            for (question_id, condition), rows in groups.items()]
