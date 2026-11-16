# Frozen evaluation and analysis

The completed first-run adapter is accepted for direct evaluation without further training. It trained on 896 examples; all 100 development examples remain held out. Native development-protocol validation is the current gate before MATH-500. There are no final scores yet.

`freeze_evaluation.py` requires verified Stage 6 completion and all 45 held-out development generations. It records exact model/conversion/adapter/data/prompt/code hashes, runtime versions, token budgets, decoder, grader and analysis choices. It audits only already-sealed benchmark questions and context lengths. If benchmark answer artifacts already exist without a freeze, it refuses to create a retrospective freeze.

Native execution success is insufficient by itself: the structured conditions must demonstrate an extractable answer readout on development. A completed run with zero such outputs fails before the benchmark freeze. This diagnoses controller/model compatibility without using correctness or benchmark scores. Failed native evidence and bounded readout-prefix diagnostics are documented in `protocol_development_notes.md`.

Only `benchmark.sealed_test_rows` opens the benchmark answers for evaluation. It first verifies the physical freeze, all frozen file hashes and the generation runtime, then acquires the pinned 500-row source, validates upstream checksums when supplied, and confirms the exact sealed question IDs, order and text hashes. The earlier Stage 2 transport projection discarded non-question fields; that documented transport deviation is not a scoring run.

## Predictions and failures

The run creates exactly 4,500 fixed slots: 500 questions times A/B/C/D256/D512/D1024/D2048/E1/E2. Conditions rotate cyclically within fixed question order. Model load and a 16-token warmup on development are outside prediction timings. Peak active MLX memory includes both resident models and is not dedicated VRAM.

Each attempted output retains prompt, token IDs, visible segments, injected controls, stop/cap/deadline events, adapter/conversion identities, extraction result, timing and memory. Atomic immutable attempts precede slot updates; interrupted slot pointers recover from durable attempts. Infrastructure failures can be retried at most twice. Completed wrong, unextracted, capped or timed-out model outputs cannot be rerolled. Unattempted and exhausted infrastructure slots remain in the denominator.

The local grader runs separately after generation. It accepts the last balanced boxed expression or the latest explicit Answer/Final answer claim, including the next nonempty line. An unfinished latest box prevents fallback to an earlier claim. It never guesses a trailing number from working. Exact literals are checked first; otherwise Math-Verify 0.9.0 parses LaTeX with no fallback and compares gold first, prediction second. Parser and verifier timeouts are each five seconds; strict comparison, six-digit float rounding and numeric precision 15 are frozen with the runtime and normalization settings. Synthetic tests cover fractions, radicals, powers, polynomial forms, tuples, sets, intervals, symbols, signs, decimals and malformed claims.

Extraction/invalid-prediction failures count as incorrect. Gold-parse or verifier errors are explicit unresolved grades. Infrastructure failures are distinct from model errors. Full accuracy is correct / 500 only when all 500 planned grades resolve. Partial results show resolved/planned counts, confirmed-correct fraction of planned and conditional resolved accuracy separately; no full-coverage accuracy is invented.

## Statistics and artifacts

Analysis retains every planned slot. It reports mean/median measured reasoning-content tokens, total emitted tokens and generation latency, with actual-output cost denominators; extraction, timeout, safety, answer-cap and Wait statistics remain available in JSON. A/B reasoning counts are unavailable. Nominal caps never substitute for measured token use.

Every comparison against C pairs identical question IDs. Bootstrap resamples paired IDs with replacement, seed 42, 10,000 draws, 95% percentile intervals. Unresolved IDs are explicitly listed. Intervals are descriptive, unadjusted for multiple comparisons, and do not capture training-seed variation. Transition tables retain wrong-to-right, right-to-wrong, wrong-to-wrong and right-to-right outcomes. Qualitative cases use the first lexicographic ID in every category, including empty categories, avoiding positive-only selection.

`analyze_results.py --input PATH --output DIR` regenerates analysis and PNG/SVG figures from all 4,500 evaluated JSONL slots alone. The report writer uses actual saved analysis and training artifacts to create `docs/paper.md`, `results.md`, `error_cases.md`, `demo_notes.md` and `handoff.json`. Hardware and measured runtime appear at the end of the paper. A partial run produces a plainly labeled partial draft.
