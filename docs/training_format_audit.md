# Assistant-only training format audit

Date: 2026-11-14. This independent Stage 4 validation was performed while the Stage 3 checkpoint transfer was pending. It uses the already pinned tokenizer and eligible original data; it does not train or load model weights.

## Observed checks

`scripts/audit_training_format.py` checked all **996 eligible examples** (896 train / 100 development). Each full formatted sequence round-tripped through the tokenizer without a text change, and each tokenized user/system/assistant-prefix region matched the prefix of its full conversation exactly. The script rejected any ambiguous prefix instead of guessing a loss boundary.

Labels are `-100` over the complete prompt, then equal the original input IDs over the complete assistant target. They are not manually shifted: the causal model handles its one-token shift during loss calculation. The teacher reasoning, answer transition, teacher `attempt` and assistant ending remain supervised. The released answer-prefix rule is preserved rather than replacing `attempt` with the separate `solution` column.

The existing token vocabulary supplies:

| Marker | Token IDs |
|---|---|
| `<|im_start|>think` plus newline | `151644, 26865, 198` |
| newline plus `<|im_start|>answer` plus newline | `198, 151644, 9217, 198` |
| `<|im_end|>` | `151645` |
| Training padding `<|fim_pad|>` | `151662` |

The think/answer strings are composite sequences, not new single special tokens. No tokenizer tokens were added. Padding uses the existing distinct fim-pad ID, right-pads batches, and masks only added positions. Real assistant ending labels remain intact. The original inference tokenizer's padding setting is not mutated by this collator.

The two-example real batch padded to **2 x 8,165** positions and passed positional mask checks. Unit tests also verify integer Torch tensor conversion, exact-limit acceptance, overflow rejection, label alignment, and retention of real content even if its ID equals the padding ID.

## Decoded sample review

The complete sample `s1k-0000-aa8c2d557d63` was decoded to `results/training_format/decoded_sample.txt` and inspected in full, including the separate ignored and supervised sections. It contains:

- **156 prompt tokens**: Qwen system message, original user question, and assistant prefix, all ignored in loss.
- **8,009 assistant tokens**: official think marker, original reasoning trace, answer transition, teacher response, end-of-message token and trailing newline, all supervised.
- **8,165 total tokens**, with no missing answer or truncated ending.

The sample is a proof problem and includes unsuccessful derivations in the released teacher text. This audit establishes formatting/masking, not mathematical correctness. Original teacher text is retained; it is not silently repaired, replaced or excluded. Teacher-answer noise and the mixed nature of development tasks must be considered when choosing development metrics later.

## Overflow policy and training gate

`configs/formatting.json` declares rejection of overlength complete examples, with **no silent truncation**. The audit explicitly passed a limit one token shorter than the real sample and observed `ContextOverflowError`. The full target remained unchanged.

The global training context ceiling remains unset. No sample is currently excluded for context. The later memory pilot must measure complete examples at increasing lengths before choosing a ceiling and reporting coverage; a format audit is not proof that an 8,165-token forward/backward pass fits this Mac. No training has run. Before scaling training, measure forward/backward memory in a one-example pilot.

## Reproduction and evidence

```bash
.venv/bin/python scripts/audit_training_format.py
.venv-inference/bin/python -m unittest discover -s tests -v
```

The observed test run passed **52 tests**. The full-data audit summary is exported to `docs/training_format_summary.json`; ignored local evidence includes `results/training_format/per_example.jsonl`, the full decoded sample, and `summary.json`. The exported summary records source/config hashes, token IDs and exact counts. These tests do not complete the pending Stage 3 model-generation acceptance gate.
