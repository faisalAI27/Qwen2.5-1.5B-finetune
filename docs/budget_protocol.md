# Budget forcing and held-out validation

Implementation is complete and unit-tested; native validation with the selected full development adapter is pending Stage 6. `configs/budget_protocol_v1.json` is provisional until `manifests/evaluation_freeze.json` exists. No benchmark scores have been computed.

The controller follows the original s1 inference harness at implementation revision `77272c6e925d610257a50b520bad15330b513389`, `vllm_causallms.py` blob `caeb333eef5b2558fd9ccddf5c677637723aa6a9`. That harness stops reasoning on im_start or model EOS, removes the encountered stop before continuation, appends Wait, and uses an answer header for readout. Our implementation uses greedy MLX segments and explicit cumulative budgets. It does not reproduce the upstream overshoot-and-post-truncate path.

## Boundaries

The prompt is the matching tokenizer's user chat template followed by `<|im_start|>think\n`, exactly as in training. The token audit in `budget_token_audit.json` records:

| Control | Token IDs |
|---|---|
| Think prefix | 151644, 26865, 198 |
| Reasoning role-marker stop | 151644 |
| Answer header | 151644, 9217, 198 |
| Newline | 198 |
| Continuation `\nWait` | 198, 14190 |

EOS IDs come from the pinned model's generation configuration. These are tokenizer identities, not evidence that the trained small model has learned them. Native development outputs will separately record role-marker stops, EOS, caps and actual Wait use. Zero learned role-marker stops must be disclosed in the final report.

## Nine conditions

A and B use the identical converted pretrained base, with no adapter modules, and the versioned direct/reasoning prompts. Their allowance is 9,728 total generated tokens. Their unstructured reasoning-token count is unavailable.

C, D256/D512/D1024/D2048 and E1/E2 use the adapted model and a common two-phase readout. C stops reasoning naturally, subject to the provisional 8,192-token safety allowance. D limits reasoning to the named cap, then transitions to answer generation. E intercepts up to one or two actual role-marker/EOS stops, removes the emitted stop from the next context, injects `\nWait` and continues. An exhausted cap without a stop never triggers Wait. Zero-content actual stops are bounded by the continuation count; an unexplained empty backend response is an infrastructure error.

Every structured condition receives the same training-format answer header and 1,536-token answer allowance. A newline is prepended only when absent. There is no condition-specific instruction that could confound a reasoning-cap comparison. The absolute ceiling is 12,288 new tokens including injected controls; the full answer allowance is reserved before reasoning or a continuation. The 900-second deadline includes prefill and all segments, checked between generated tokens; a single in-flight GPU operation may finish after the deadline. A timeout is retained as a model outcome, including an empty/unextracted answer, and is never rerolled.

Stop tokens remain in raw model-segment logs and count toward emitted-token budgets even when removed from the next input. Visible reasoning-content counts exclude removed stop tokens. Control injections are recorded separately, never described as model-generated tokens. Latency includes segment re-prefill and control overhead. No internal model states are logged.

## Validation and freeze

Twelve state-machine tests cover natural stops, caps, one/two Waits, no stop, EOS, answer overflow, remaining allowance, timeout, zero-content stops and invalid empty segments. Prompt/token tests validate the actual tokenizer without a GPU. These tests do not replace native validation.

After both full trials and fresh final training finish, the first five fixed development questions run through all nine conditions using the **selected 896-row adapter**, keeping development targets held out. All 45 generations must execute successfully before freeze. Failed extraction or learned-marker incompatibility remains visible; neither is silently repaired. Any protocol change requires retained development evidence and a new protocol version before test access. The final 996-row adapter never selects inference settings.

Both evaluation variants use the source-traced four-bit/group-64 base with FP32 activations, no KV quantization, greedy decoding and prefill chunks of 512. Two models remain resident during evaluation. The development gate measures whether this fits on the actual Mac before benchmark answers are opened.
