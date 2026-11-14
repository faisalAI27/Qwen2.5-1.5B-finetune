"""Greedy MLX segment decoding for the explicit two-phase protocol."""

import json
import time
import importlib.metadata
import platform

from s1small.budget_forcing import Protocol, Segment
from s1small.data import digest
from s1small.formatting import THINK_START
from s1small.generate import build_prompt
from s1small.mlx_runtime import load_local_model


def runtime_signature():
    packages = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "mlx-lm", "transformers", "numpy")}
    expected = {"mlx": "0.32.4", "mlx-metal": "0.32.4", "mlx-lm": "0.32.2", "transformers": "5.10.1", "numpy": "2.4.3"}
    if packages != expected:
        raise ValueError("Generation runtime differs from validated pins")
    return {"packages": packages, "python": platform.python_version(), "architecture": platform.machine(),
            "macos": platform.mac_ver()[0], "kernel": platform.release()}


def s1_prompt(question, tokenizer):
    return tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                         tokenize=False, add_generation_prompt=True) + THINK_START


def token_protocol(tokenizer, eos_ids):
    encode = lambda text: tuple(tokenizer.encode(text, add_special_tokens=False))
    marker = encode("<|im_start|>")
    if len(marker) != 1 or encode(THINK_START) != (151644, 26865, 198):
        raise ValueError("Tokenizer differs from the audited s1 markers")
    return Protocol(marker[0], tuple(eos_ids), encode("<|im_start|>answer\n"), encode("\n"), encode("\nWait"))


class MLXDecoder:
    def __init__(self, model, tokenizer, eos_ids, prefill_step_size=512):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_ids = tuple(eos_ids)
        self.prefill_step_size = prefill_step_size
        self.unmapped = set()
        model.eval()

    def decode(self, tokens):
        # Qwen's embedding table has unused entries above tokenizer length.
        # Keep valid model IDs rather than round-tripping them through text.
        limit = len(self.tokenizer)
        if all(token < limit for token in tokens):
            return self.tokenizer.decode(tokens, skip_special_tokens=False)
        text, pending = [], []
        for token in tokens:
            if token < limit:
                pending.append(token)
            else:
                text.append(self.tokenizer.decode(pending, skip_special_tokens=False))
                pending = []
                text.append(f"<|unmapped_token_{token}|>")
                self.unmapped.add(token)
        text.append(self.tokenizer.decode(pending, skip_special_tokens=False))
        return "".join(text)

    def generate(self, context, maximum, stop_ids, deadline):
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        if maximum <= 0 or not context:
            raise ValueError("A nonempty context and positive generation allowance are required")
        if len(context) + maximum > self.model.args.max_position_embeddings:
            raise ValueError("Prompt plus allowance exceeds context; no truncation is performed")
        if any(token < 0 or token >= self.model.args.vocab_size for token in context):
            raise ValueError("Invalid embedding-table token ID")
        if time.perf_counter() >= deadline:
            return Segment((), timed_out=True)
        stream = generate_step(mx.array(context), self.model, max_tokens=maximum,
                               sampler=lambda logits: mx.argmax(logits, axis=-1),
                               prefill_step_size=self.prefill_step_size,
                               max_kv_size=None, kv_bits=None)
        output, timed_out = [], False
        try:
            for token, _ in stream:
                output.append(int(token))
                timed_out = time.perf_counter() >= deadline
                if token in stop_ids or timed_out:
                    break
        finally:
            stream.close()
        return Segment(tuple(output), timed_out=timed_out)


def load_decoder(root, recipe, adapter_path=None):
    runtime_signature()
    model, tokenizer, _, provenance = load_local_model(root, recipe, adapter_path)
    if adapter_path is None:
        # The training loader creates zero-initialized adapters. Remove them so
        # A/B use exactly the converted pretrained model with no adapter module.
        from mlx_lm.tuner.utils import remove_lora_layers
        model = remove_lora_layers(model)
    revision = provenance["source"]["revision"]
    config_path = root / "artifacts/base-mlx4" / (revision + "-g64") / "generation_config.json"
    eos = json.loads(config_path.read_text())["eos_token_id"]
    eos_ids = eos if isinstance(eos, list) else [eos]
    return MLXDecoder(model, tokenizer, eos_ids), tokenizer, provenance


def unstructured_answer(decoder, prompt_tokens, maximum, absolute, timeout):
    started = time.perf_counter()
    maximum = min(maximum, absolute)
    segment = decoder.generate(prompt_tokens, maximum, set(decoder.eos_ids), started + timeout)
    tokens = list(segment.tokens)
    if not tokens and not segment.timed_out:
        raise RuntimeError("Empty baseline generation")
    termination = ("eos" if tokens and tokens[-1] in decoder.eos_ids else
                   "timeout" if segment.timed_out else "generation_budget")
    visible_ids = tokens[:-1] if tokens and tokens[-1] in decoder.eos_ids else tokens
    return {"status": "ok", "segments": [{"origin": "model", "phase": "unstructured", "token_ids": tokens,
                                            "text": decoder.decode(tokens), "timed_out": segment.timed_out}],
            "answer_text": decoder.decode(visible_ids), "total_generated_tokens": len(tokens),
            "reasoning_tokens": None, "reasoning_phase_generated_tokens": None,
            "answer_generated_tokens": None, "injected_tokens": 0,
            "total_new_tokens_consumed": len(tokens), "forced_wait_count": 0,
            "termination_reason": termination, "timed_out": segment.timed_out,
            "safety_cap_hit": termination == "generation_budget", "answer_cap_hit": None,
            "reasoning_cap_hit": None, "latency_seconds": time.perf_counter() - started,
            "visible_transcript": decoder.decode(tokens)}
