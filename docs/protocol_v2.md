# Revised common readout: development protocol v2

Training remains complete, with the same accepted 896-example adapter. There are no further optimizer updates or changes to the checkpoint. MATH-500 remains sealed until this revised protocol passes its full native development gate.

## Observed failure and bounded probes

The original `protocol-dev-v1` completed all 45 generations. Five outputs had extractable answers, all from direct-answer condition A; none of the 35 adapted structured outputs did. No reasoning-phase im_start stop was observed; seven outputs encountered reasoning EOS and two outputs used actual Wait continuations. Empty answer phases and repeated calculations were retained. The native readout gate stopped the pipeline before benchmark freeze/access.

The readout diagnostic then completed 15 bounded outputs: three common headers on the same five held-out questions, with 512 reasoning tokens, 256 answer tokens, an absolute ceiling of 1,024 and a 120-second timeout. The original header produced zero extracted answers. `Answer: ` produced five prefix-assisted extractions, but three outputs still hit the answer cap. `Final Answer: ` produced five short nonempty answer bodies, all terminating at EOS (3–21 generated answer tokens). These counts concern readout behavior, not correctness. Units and numeric values in these mixed math/physics development questions were not scored to select the label.

The exact original-run audit and diagnostic summary are `protocol_validation_protocol-dev-v1.json` and `readout_diagnostic_v1.json`. Both hashes are recorded as development evidence in `configs/budget_protocol_v2.json`. Raw visible segments remain under `results/protocol_dev/protocol-dev-v1/` and `results/diagnostics/readout-v1/`.

## Defined revision

Every C/D/E condition now injects the same training-format answer header followed by `Final Answer: `. A/B keep their original prompts. Reasoning caps, answer allowance, safety ceiling, EOS/role stops, Wait behavior, greedy decoding, backend, precision and model weights are unchanged. The common label is an explicit readout adaptation; it is not evidence that the model learned the original s1 boundaries.

The label's tokens are charged as injected controls against the absolute ceiling. `answer_text` and generated-token counts still contain only the model's answer body. Extraction uses the declared label plus that body; records retain the exact extraction input and origin. The label supplies no answer value. An empty body remains unextracted; later unfinished boxes still prevent fallback to an earlier claim. Prefix-assisted extraction does not establish mathematical correctness.

`readout.py` implements this context explicitly. Synthetic tests cover empty bodies, model-only versus contextual extraction, provenance and malformed boxes. The pipeline tests verify that a selected protocol file is forwarded consistently to development, freeze and test. The latest ordinary suite ran 117 tests with no failures (six optional/native skips); native v2 validation is still pending. A CPU tokenizer check confirms the injected token sequence matches the diagnostic prefix.

## Required next gate

Run a fresh `protocol-dev-v2` on all five questions and all nine conditions. Retain v1 outputs unchanged. Freeze cannot proceed until all 45 outputs execute and structured readout is observed. Learned-marker limitations, cap hits, extraction failures and actual Wait counts remain reportable. Only then can the benchmark open under the frozen v2 config and extraction context.

```bash
caffeinate -i .venv-mlx/bin/python scripts/complete_project.py \
  --protocol-run-id protocol-dev-v2 --protocol-config budget_protocol_v2.json
```

This command uses the already selected Stage 6 adapter and invokes no training. The report writer discloses the failed calibration, bounded probes, common injected label, single-run model choice and actual training coverage.
