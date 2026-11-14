"""Released s1 text construction; training-label masking is a later stage."""


THINK_START = "<|im_start|>think\n"
ANSWER_START = "\n<|im_start|>answer\n"


class ContextOverflowError(ValueError):
    """A complete training example exceeds the explicitly requested context."""


def assistant_parts(row):
    reasoning = "\n".join(row["thinking_trajectories"]).strip()
    attempt = row["attempt"]
    answer = ("Answer: " + attempt if "Answer:" not in attempt else attempt).strip()
    return reasoning, answer


def format_example(row, tokenizer):
    reasoning, answer = assistant_parts(row)
    return tokenizer.apply_chat_template([
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": THINK_START + reasoning + ANSWER_START + answer},
    ], tokenize=False, add_generation_prompt=False)


def measure_example(row, tokenizer):
    text = format_example(row, tokenizer)
    prompt = tokenizer.apply_chat_template([
        {"role": "user", "content": row["question"]},
    ], tokenize=False, add_generation_prompt=True)
    reasoning, answer = assistant_parts(row)
    expected = prompt + THINK_START + reasoning + ANSWER_START + answer
    if not text.startswith(expected):
        raise ValueError("Chat template does not preserve the expected s1 content/prompt boundary")
    # Tokenize the complete rendered sequence once, so phase lengths are additive
    # even where BPE tokens cross a text boundary. This is an audit, not a loss mask.
    boundaries = [len(prompt), len(prompt + THINK_START), len(prompt + THINK_START + reasoning),
                  len(prompt + THINK_START + reasoning + ANSWER_START), len(expected)]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    counts = {"prompt": 0, "reasoning": 0, "answer": 0, "structure": 0}
    crossing = 0
    for start, end in encoded["offset_mapping"]:
        if any(start < boundary < end for boundary in boundaries):
            crossing += 1
        if end <= boundaries[0]:
            counts["prompt"] += 1
        elif start >= boundaries[1] and end <= boundaries[2]:
            counts["reasoning"] += 1
        elif start >= boundaries[3] and end <= boundaries[4]:
            counts["answer"] += 1
        else:
            counts["structure"] += 1
    counts["total"] = len(encoded["input_ids"])
    counts["boundary_crossing_tokens"] = crossing
    if counts["total"] != sum(counts[key] for key in ("prompt", "reasoning", "answer", "structure")):
        raise ValueError("Token accounting failed")
    return counts


def make_training_example(row, tokenizer, max_length=None):
    """Supervise the complete assistant response; never truncate or shift labels.

    Causal language models shift these labels internally during next-token loss.
    Prefix tokenization must agree exactly with the full conversation.
    """
    text = format_example(row, tokenizer)
    prompt = tokenizer.apply_chat_template([
        {"role": "user", "content": row["question"]},
    ], tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
    if not text.startswith(prompt) or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Assistant boundary does not align with the tokenized prompt prefix")
    if not text.endswith("<|im_end|>\n"):
        raise ValueError("Expected the original Qwen assistant end-of-message suffix")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("No assistant tokens to supervise")
    if max_length is not None:
        if max_length <= 0:
            raise ValueError("max_length must be positive or None")
        if len(full_ids) > max_length:
            raise ContextOverflowError(f"Complete example has {len(full_ids)} tokens; limit is {max_length}. No truncation applied.")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
            "prompt_tokens": len(prompt_ids)}


def collate_training_examples(examples, pad_token_id, return_tensors=False):
    """Right-pad by position, retaining real assistant EOS and delimiter labels."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    if not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("A valid existing padding token ID is required")
    maximum = max(len(row["input_ids"]) for row in examples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in examples:
        length = len(row["input_ids"])
        if len(row["labels"]) != length or row["attention_mask"] != [1] * length:
            raise ValueError("Examples must contain aligned unpadded IDs, masks and labels")
        padding = maximum - length
        batch["input_ids"].append(row["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append([1] * length + [0] * padding)
        batch["labels"].append(row["labels"] + [-100] * padding)
    if return_tensors:
        import torch
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    return batch
