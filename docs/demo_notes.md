# Five-minute presentation and demonstration

This is a completed single-run research project, with a negative/mixed result. The paper is `output/pdf/s1-small-model-report.pdf`; Markdown and plots are in `docs/`.

## Suggested presentation

1. **Question (30 seconds):** Does s1-style training plus longer inference help a 1.5B model enough to justify the cost? Original s1 used a 32B model. We test a constrained adaptation, not its headline scores.
2. **Implementation (60 seconds):** Train on 896 complete original s1K examples, keep 100 held out, use affine Q4 with FP32 LoRA on the 8 GB M1. One accepted epoch, 56 updates; teacher NLL falls 14.9%. Explain that NLL is not math accuracy. A shared answer-label repair was selected on development behavior before testing.
3. **Evaluation (60 seconds):** Show `figures/final_accuracy_cost.png`. A separately validated FP16 reconstruction runs all 500 questions under nine conditions on one RTX 4090. B scores 43.0%; adapted C 28.6%; D1024 28.6% using 82.1% fewer output tokens. The B/C comparison changes prompt/readout and adapter.
4. **Continuation (45 seconds):** Show `figures/final_paired_transitions.png`. E2 scores 28.8%, one net answer better, 30.2% more tokens; paired interval [-2.2, 2.6] pp. No clear improvement. Both rescued and spoiled answers count.
5. **Demo (60 seconds):** Replay a saved rescued case and a spoiled case below. Say explicitly that replay loads saved outputs. If demonstrating fresh inference, run the bounded toy question on the Mac and label it illustrative, on a different backend from final evaluation.
6. **Limitations (45 seconds):** One training seed; finite natural safety ceiling; learned reasoning marker behavior was not established; E trajectories can diverge before Wait. Show the hardware appendix and explain 2.29 GPU generation hours versus 17.84 recorded local training hours; neither is total billing or total project compute.

## CPU-only saved-output replay

```bash
.venv-eval/bin/python scripts/demo_results.py --question-id test/algebra/170.json --condition D256
.venv-eval/bin/python scripts/demo_results.py --question-id test/algebra/1031.json --condition D256
```

The first case changes C's 21 to the correct 11; the second changes C's correct 5.5 to an incorrect 10. Both are the frozen lexicographic case selections. These commands validate the source ZIP hash, read saved GPU records, and load no model. All 24 selected categories are in `docs/error_cases.md`.

## Bounded fresh Mac inference

```bash
.venv-mlx/bin/python scripts/demo.py --question 'What is 17 times 23?' \
  --condition D256 --answer-budget 128 --timeout 90 --output results/demo/live-mac.json
```

This uses the original frozen adapter and a separate illustrative question. The expected mathematical answer is 391, but a model failure remains a failure. It does not resume training or the benchmark. The output records the actual generated body, extracted answer, injected label, tokens, elapsed time and backend. The shorter demo answer allowance is explicitly different from the full benchmark. See `docs/demo_verification.json` for the observed demonstration outcome.

## Questions to be ready for

- **Why not claim fine-tuning hurt by exactly 14.4 pp?** B and C differ in prompt/readout as well as adapter. We measured policies, not a matched prompt adapter ablation.
- **Does 28.6% vs 28.6% prove equivalence?** No. It is equal observed accuracy with a paired interval, not a prespecified equivalence test.
- **Did Wait fix some answers?** Some E outputs become correct and others become wrong; first-segment divergence means changes are not all solely attributable to Wait.
- **Can the project be reproduced without renting another GPU?** Saved-output auditing, grading/analysis, report generation and replay are CPU work. Exact new generation requires the documented assets and an appropriately configured GPU or a separately reported local backend.
