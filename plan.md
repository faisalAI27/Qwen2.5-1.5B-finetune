# Implementation Plan — s1 Test-Time Scaling on a 1.5B Model

**Project:** Resource-Constrained Adaptation of *s1: Simple test-time scaling*  
**Team:** Faisal Imran · Muhammad Ibrahim · Hamza Saeed  
**Planning status:** Stages 1–6 complete under the approved single-run decision; proceed to Stage 7 with the existing 896-example adapter and no further training
**Primary specification:** `NLP_Proposal.pdf` (the group's approved project proposal)  
**Last reviewed:** 2026-11-16

**Hardware constraint (2026-11-14):** The user requires this local computer for the project: Apple M1 MacBook Air, 8 GB unified memory, 7-core GPU. PyTorch MPS passed a small tensor test outside the sandbox. See `docs/hardware.md` for measured specifications and limitations. Remote GPU access is not assumed; full-length backward feasibility and stable local pilot training are verified, while full training/evaluation remain in progress. Include the actual hardware and runtime measurements in the final report.

**Execution instruction update (2026-11-14):** The user repeatedly asked to "go ahead and complete it" during execution. Continue through the remaining project work without requiring another go-ahead after each stage. Keep the scientific and measured hardware acceptance gates, record actual outcomes, and report any genuine blocker. The stage-by-stage stop-for-review wording below is original guidance, superseded by this instruction; it does not require pausing already authorized work. Independent tokenizer/format checks may proceed during a checkpoint-download wait, without claiming model or training success.

**Execution update (2026-11-16):** The user now wants to review the first run and validation before deciding whether a second trial is worth the compute. Finish the current trial and all 100 development targets, verify its adapter, and compare held-out teacher-target NLL with the same original 1.5B base. Hold all later training and benchmark work for this decision. The original 32B paper is contextual, not a validation-loss threshold. The original scheduler is intentionally suspended; do not resume it automatically. See `docs/training_protocol.md` and `results/runtime/first_trial_review_request.json`. This update supersedes automatic progression beyond the first-trial review.

**Execution update (2026-11-16, accepted existing run):** The user has decided to proceed with this run only. Use the completed 896-example adapter directly for protocol validation and frozen benchmark evaluation. Skip trial 2 and defer the original fresh 996-example final retraining. Preserve the 100 held-out development examples and document this single-candidate, compute-constrained deviation in the report. `configs/single_run_decision.json` is the current choice. The earlier review hold is resolved for inference/evaluation; it does not authorize more training.

> **Purpose of this file:** This is the executable project roadmap for a human developer working with Codex in VS Code. Complete **one stage at a time**, test the stage, update its checkbox, and stop for review. Do not ask Codex to build the entire project in one prompt.

---

## 0. Mission and boundaries

### Research question

**Can supervised fine-tuning on 1,000 curated reasoning examples, combined with budget forcing at inference time, improve mathematical accuracy in Qwen2.5-1.5B-Instruct? How does any improvement compare with additional reasoning tokens and latency?**

This is a **small-model adaptation**, **not** an exact reproduction of the original 32B-parameter paper. A null, mixed, or negative result is a legitimate result if the experiments are rigorous.

### Canonical inputs and references

| Item | Locked choice or reference |
|---|---|
| Paper | [s1: Simple test-time scaling, v1](https://arxiv.org/abs/2501.19393v1) |
| Original implementation | https://github.com/simplescaling/s1 |
| Pretrained model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Training data | **Original** `simplescaling/s1K`, 1,000 examples |
| Development split | Fixed, seeded **900 train / 100 development** |
| Final training | Reinitialize from the same original base model; train on all eligible original 1,000 examples with frozen settings (document any exclusions) |
| Primary held-out benchmark | `HuggingFaceH4/MATH-500`, fixed 500-question `test` split |
| Optional additional benchmark | AIME24, **only after** the primary experiment |
| Training | LoRA; QLoRA only if hardware requires it, with the change recorded |
| Development environments | Local Apple M1 MacBook Air with 8 GB unified memory for all work; measure training and inference feasibility before scaling |

**Important:** The original repository now documents s1.1 in several default examples. We must use the original **s1 / s1K** for the proposed study; do not silently change to `s1K-1.1`, the 32B checkpoint, or a newer release. Record exact revision IDs for everything downloaded.

### In scope

- Reproduce the original s1 reasoning/answer training **format** as closely as compatible with the 1.5B tokenizer.
- Use assistant-only next-token loss while masking the user prompt and padding.
- Train LoRA adapters, verify correct loading, and generate outputs.
- Compare direct and reasoning-prompted **base-model** baselines with the **fine-tuned** natural-stopping and budget-forcing conditions.
- Measure accuracy, reasoning tokens, total tokens, wall-clock latency, answer-extraction failures, and forced-stop/safety-limit events.
- Export reproducible code, configs, checkpoints/adapters, per-question outputs, figures, and a report.

### Out of scope (unless core work is finished)

- Training a 32B/64B model; training from scratch; full dataset recreation with teacher APIs.
- Changing to s1K-1.1 without explicit, documented research-design change.
- GPQA, GSM8K, REBASE, reinforcement learning, rejection sampling, a web app, or a polished chatbot UI.
- Claims that results replicate original s1 performance or that additional reasoning **must** improve accuracy.

---

## 1. Non-negotiable rules for Codex

1. **Read this `plan.md` at the beginning of every new Codex session.** Use it as the source of truth, alongside the proposal.
2. **Implement only the next unfinished stage.** Do not jump ahead or modify completed experiment settings silently.
3. Write **clear, beginner-readable Python**. Prefer straightforward scripts/functions over complex abstractions or unneeded dependencies.
4. Before significant coding, briefly state: current stage, files to create/change, verification to run, and any unresolved assumptions.
5. End each stage with: implemented files, exact commands run, test results, risks/blockers, and next stage; **stop** for review.
6. Never claim that a command, training job, benchmark, or test succeeded unless it actually ran and its output was observed.
7. Never invent memory usage, duration, GPU cost, benchmark scores, checkpoint revisions, or published definitions.
8. Use fixed random seeds where possible. Record package versions, Git commit, dataset revisions, model revision, tokenizer revision, hardware, and runtime settings.
9. Don't fetch or install the original multi-GPU/FSDP stack blindly. Adapt to single-GPU LoRA and verify modern library compatibility before pinning versions.
10. Write **unit tests for logic**, **smoke tests for workflows**, and **small dry runs** before expensive full runs.
11. Do not call the MATH-500 *test* split to tune prompts, training parameters, token budgets, stop markers, or answer extraction. Only evaluate it after freeze.
12. Save **one raw prediction per question per condition**, including failures, so aggregate scores can be reproduced.
13. If a stage is impossible due to hardware or model-format incompatibility, report evidence and propose the smallest defensible adjustment; **do not hide deviations**.
14. Never commit credentials, cache directories, datasets copied from the hub, model weight files, or large result artifacts to Git.
15. After review, mark stages complete and append a concise dated entry to the progress log at the end of this file.

---

## 2. Experiment matrix (locked design)

| ID | Model | Condition | Main comparison |
|---|---|---|---|
| A | Unmodified 1.5B | Direct-answer prompt | Ordinary pretrained baseline |
| B | Unmodified 1.5B | Explicit reasoning prompt | Effect of prompting without fine-tuning |
| C | LoRA fine-tuned 1.5B | Natural stopping | Effect of s1-style training |
| D256 | LoRA fine-tuned | Maximum **reasoning** budget: 256 tokens | Short reasoning cap |
| D512 | LoRA fine-tuned | Maximum **reasoning** budget: 512 tokens | Medium reasoning cap |
| D1024 | LoRA fine-tuned | Maximum **reasoning** budget: 1,024 tokens | Longer reasoning cap |
| D2048 | LoRA fine-tuned | Maximum **reasoning** budget: 2,048 tokens | Longest core cap |
| E1 | LoRA fine-tuned | Up to one forced `Wait` continuation | Extra reasoning opportunity |
| E2 | LoRA fine-tuned | Up to two forced `Wait` continuations | Further reasoning opportunity |

**Fair-comparison rules**

- Use the **same question order, fixed dataset IDs, answer allowance, answer-grading policy, decoder settings (greedy / deterministic), and model revision** wherever comparable.
- Preserve the intended distinction between *base prompting* (A/B) and *fine-tuned structural reasoning output* (C/D/E).
- Record exact prompting and delimiter choices as immutable config files; make no assumptions that a bare `Wait` always induces correct continuation.
- **Reasoning cap != total answer cap.** A reasoning-cap event should transition the model toward its answer phase; it must not simply truncate all generation before any answer.
- Keep a **separate answer-token allowance** and **shared absolute safety ceiling** for natural-stopping and forced-continuation conditions. Finalize values on development data, not on MATH-500.
- `Wait` is added only when the protocol identifies an attempted reasoning stop, subject to remaining budget and an explicit continuation count.
- Log whether forced continuation was actually triggered. E1/E2 may make zero continuations if the model never emits the targeted stop marker before the safety ceiling.
- An optional combined cap-plus-`Wait` condition is allowed **only after** the core matrix is complete; report it as exploratory.

### Hypotheses (not assumed outcomes)

- **H1:** Reasoning fine-tuning may change MATH-500 accuracy relative to both unmodified baselines.
- **H2:** Reasoning-token caps may alter accuracy and cost; the trend may improve, plateau, or decline.
- **H3:** Forced continuation may rescue some wrong answers and spoil some correct answers; its net benefit is uncertain.

---

## 3. Suggested repository layout

```text
s1-small-model/
├── plan.md                    # This implementation contract and progress tracker
├── README.md                  # Setup, reproducible commands, run table
├── pyproject.toml             # Project and dependencies; versions locked after audit
├── requirements-lock.txt      # Generated only after environment validation
├── .gitignore
├── configs/
│   ├── dataset.yaml
│   ├── baseline.yaml
│   ├── train_pilot.yaml
│   ├── train_final.yaml
│   └── eval.yaml
├── docs/
│   ├── research_audit.md
│   ├── decisions.md
│   ├── hardware_pilot.md
│   └── protocol.md
├── src/s1small/
│   ├── environment.py
│   ├── data.py
│   ├── formatting.py
│   ├── model.py
│   ├── training.py
│   ├── generate.py
│   ├── budget_forcing.py
│   ├── extraction.py
│   ├── grading.py
│   └── metrics.py
├── scripts/
│   ├── check_environment.py
│   ├── prepare_data.py
│   ├── inspect_lengths.py
│   ├── run_baselines.py
│   ├── train_pilot.py
│   ├── train_final.py
│   ├── evaluate.py
│   └── analyze_results.py
├── tests/
│   ├── test_data.py
│   ├── test_formatting.py
│   ├── test_budget_forcing.py
│   ├── test_extraction.py
│   └── test_metrics.py
├── artifacts/                 # Ignored by Git: adapters/checkpoints
├── data/                      # Ignored by Git: downloaded/cached data
└── results/                   # Local result runs; export small summaries separately
```

The layout is a **target**, not a requirement to create every file in Stage 1. Create modules only when their stage is active. Avoid empty placeholder implementation files.

---

## 4. Stage-by-stage roadmap

### Stage 1 — Project setup and original-repository audit

**Status:** [x] Complete (2026-11-14; evidence: `docs/stage1_report.md`)

**Goal:** Create a working Python repository and document how the original s1 implementation must change for a single-GPU 1.5B LoRA study.

**Tasks**
- [x] Initialize repository and create minimal folders, `.gitignore`, `README.md`, `pyproject.toml` and this plan.
- [x] Inspect original `train/sft.py`, the `sft.sh` launcher, original dataset formatter, README inference section and `eval/commands.sh`.
- [x] Note that the reference uses full-model training / FSDP and may have stale TRL APIs; identify modern supported replacements instead of copying blindly.
- [x] Document exact model/dataset/implementation IDs and their resolved revisions when available.
- [x] Add an environment check that reports Python, OS, Torch, CUDA, GPU name/VRAM, and selected dependency versions without crashing on a CPU-only laptop.
- [x] Add basic tests of the environment check and run them.

**Artifacts:** `README.md`, `docs/research_audit.md`, `scripts/check_environment.py`, dependency file, basic tests.

**Acceptance gate:** Fresh setup instructions work; checks run; audit distinguishes original s1 vs our LoRA variant; no model/data download or training was started.

### Stage 2 — Dataset acquisition, validation, and split

**Status:** [x] Complete (2026-11-14; evidence: `docs/stage2_report.md`)

**Goal:** Produce immutable data manifests and inspect whether full reasoning sequences fit the training environment.

**Tasks**
- [x] Download precisely `simplescaling/s1K`; verify the expected row count and source revision.
- [x] Validate the fields used by the proposal: `question`, `thinking_trajectories`, `attempt`. Inspect missing/empty entries and list any schema deviations.
- [x] Create deterministic **900 train / 100 development** IDs using a recorded seed. Keep split manifests and hashes; never regenerate silently. Seed 42; a separate overlap eligibility manifest leaves 896/100.
- [x] Inspect original formatting logic and measure tokenized total, prompt, reasoning and answer lengths with the **1.5B tokenizer**. Report min/median/p90/p95/max and counts exceeding candidate context limits.
- [x] Retain only the MATH-500 **metadata/question text** needed for overlap checking; keep its test split sealed from tuning/evaluation until Stage 8. Transport deviation: pinned source is JSONL, so non-question fields are transferred transiently and immediately discarded without logging/persisting/using them; see `docs/data_audit.md`.
- [x] Check normalized exact matches and near-duplicates between s1K and MATH-500. Produce an overlap report; exclusions are documented in `configs/data_exclusions.json` and reflected in eligible sample counts (996 final before context filtering).
- [x] Do not truncate a sample silently or lose the answer portion. No context exclusions/truncation applied; length policy remains for measured hardware tests.
- [x] Test split stability, required fields and ID uniqueness.

**Artifacts:** `configs/dataset.yaml`, split manifests, `docs/data_audit.md`, dataset tests.

**Acceptance gate:** Exact source and revisions recorded, deterministic split verified, sequence-length histogram/table produced, overlap handling documented.

### Stage 3 — Baseline model and small generation smoke test

**Status:** [x] Complete (28/28 native MPS generations, identical repeats/fresh-load outputs; evidence: `docs/stage3_report.md`)

**Goal:** Verify that the unmodified 1.5B model loads and can generate reproducible math answers without LoRA.

**Tasks**
- [x] Load `Qwen/Qwen2.5-1.5B-Instruct` with its matching tokenizer on available inference hardware.
- [x] Implement **A direct answer** and **B prompted reasoning** as separate, versioned prompt templates.
- [x] Choose deterministic decoding; validate generation termination and output parsing on a small **development-only** sample.
- [x] Record input, prompt, raw output, extracted answer, decoding params, token counts and latency per question.
- [x] Verify that A/B use the exact same pretrained weights and question IDs.
- [x] Do not score MATH-500 yet.

**Artifacts:** baseline config, generation code, `results/dev_baselines/`.

**Acceptance gate:** Both prompts generate parseable or explicitly marked unparseable results; repeat runs with the same settings are reproducible as far as the hardware/backend allows.

### Stage 4 — s1 training format and loss-masking tests

**Status:** [x] Complete (996 eligible examples; evidence: `docs/training_format_audit.md`). Complete examples only, no truncation; no context exclusions. Full-sequence training feasibility is measured in Stage 5.

**Goal:** Convert original s1K examples into token sequences that teach *reasoning then answer* without training on the user prompt.

**Tasks**
- [x] Inspect the **released s1 formatter** rather than inventing a new format.
- [x] Construct assistant targets from `thinking_trajectories` and teacher answer in `attempt`, preserving official delimiters and transitions where compatible.
- [x] Verify exact tokenization and special token IDs using Qwen2.5-1.5B; note that text strings and model special tokens are **not automatically interchangeable**.
- [x] Construct `input_ids`, `attention_mask`, `labels` with the user/system portions and pads set to `-100`; reasoning and answer labels must remain supervised.
- [x] Print/test a single *decoded* sample with visible prompt vs supervised regions; ensure end-of-answer is present.
- [x] Test context overflow policy and ensure no sample is truncated into a missing final-answer target.

**Artifacts:** formatter/collator, unit tests, one documented sample audit.

**Acceptance gate:** Masking tests pass; one full example is visually audited; no training starts until the example and length policy are approved.

### Stage 5 — LoRA feasibility pilot (20–50 examples)

**Status:** [x] Complete (2026-11-15): stable MLX QLoRA pilot, fresh-process reload, full 9,109-token coverage; evidence: `docs/hardware_pilot.md`. Original PyTorch/low-precision failures retained.

**Goal:** Prove that fine-tuning is technically possible and measure hardware cost before using the full dataset.

**Proposed initial settings** (project-specific pilots; **not** claims from the paper):

| Parameter | Starting point |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Method | LoRA (`r=16`, `alpha=32`, dropout `0.05`) |
| Adapter targets | Attention projections; inspect actual module names before selecting |
| Micro-batch size | 1 |
| Gradient accumulation | 16 (may change for the pilot) |
| Gradient checkpointing | Enabled |
| Precision | BF16 when supported, otherwise verified FP16 / other supported path |
| Learning rate, epochs | Experimentally selected on development data; no invented final values |
| Quantization | Optional QLoRA **only after** measured hardware constraint |

**Tasks**
- [x] Confirm local MPS/MLX backend compatibility, unified-memory headroom, free disk space and actual lengths from Stage 2; CUDA is unavailable on this Mac.
- [x] Run a **single batch forward/backward**; verify gradients and supervised-token loss.
- [x] Train on 20–50 training examples, not the development set.
- [x] Measure max allocated/reserved VRAM, sequence length, tokens/sec, step time, training loss, and checkpoint size.
- [x] Verify adapter saving, clean reload on base model, and generation before/after reload.
- [x] If OOM, reduce sequence/batch configuration **only with documented impact**; prefer preserving entire examples where feasible. Explain if any examples become ineligible.

**Artifacts:** pilot config, pilot log, adapter smoke checkpoint, `docs/hardware_pilot.md`.

**Acceptance gate:** Successful forward/backward, actual training steps, adapter reload, finite loss, and measured resource report. A pilot is **not** a final accuracy result.

### Stage 6 — Development tuning and final LoRA training

**Status:** [x] Complete under the user-approved existing-adapter route: 896 training examples, 100 held-out validation examples, verified reload, and accepted single candidate. Trial 2 skipped; fresh 996-example retraining deferred. See `configs/single_run_decision.json`.

**Goal:** Choose a stable setup on the **900/100** split, freeze it, and train the final model on all eligible original examples.

**Tasks**
- [x] Specify a **small** explicit hyperparameter grid (learning rate, number of epochs/checkpoint) justified by Stage 5 cost.
- [ ] Tune using training + held-out 100-development examples only; log every trial, seed, and development metric.
- [ ] Select one configuration using a declared rule (e.g., development loss and/or development answer accuracy); avoid cherry-picking.
- [ ] Freeze configs, data processing, prompt protocol, stopping settings and evaluation grader before the MATH-500 run.
- [ ] Deferred by user: fresh all-eligible final retraining. Current evaluation uses the existing 896-example adapter, with this deviation documented.
- [ ] Save LoRA adapter + tokenizer/config references; verify adapter merge/loading compatibility without silently replacing the base model.
- [x] Report final eligible count if any examples were removed due to overlap or context limits.

**Artifacts:** trial table, frozen YAMLs, full adapter, training logs and reproducibility metadata.

**Acceptance gate:** An independently loadable final adapter and a documented, frozen training recipe. The MATH-500 test has not influenced model selection.

### Stage 7 — Budget-forcing engine and protocol tests

**Status:** [ ] V1 native run and 15 diagnostics complete. The common Final Answer label improved concise readout in five development probes; correctness is unscored. V2 config/code/tests are ready for a fresh 45-output native validation before any benchmark freeze. See `docs/protocol_v2.md`.

**Goal:** Implement **genuine structural budget forcing**, not a generic “think longer” prompt.

**Tasks**
- [x] Inspect official s1 inference logic and any evaluation-harness modifications for boundary behavior.
- [ ] Identify think-start, think-end/answer-start, and final-end markers as actually produced by the trained model; document them exactly.
- [x] Implement **C natural stop**, **D cap and answer transition**, **E1/E2 interrupted stop + append `Wait` + resume**, with reproducible token accounting.
- [x] Keep separate `reasoning_budget`, `answer_budget`, and `absolute_safety_ceiling` controls. Prevent infinite loops.
- [x] For E1/E2, define exactly when an attempted stop occurs and how many continuations are consumed; do not blindly add `Wait` if a natural response never attempted to finish.
- [ ] Record *visible* reasoning segments, injected continuation text, stop reason, cap hits, continuation count, final answer, and total time; **do not log hidden internal model state**.
- [x] Unit-test: natural stop, forced early finish, one Wait, two Waits, no encountered stop marker, early EOS, answer overflow, remaining-budget exhaustion, and 0-length segments.
- [ ] Validate the engine on the development set; set all safety and answer allowances **before** touching the test split.

**Artifacts:** `budget_forcing.py`, protocol docs, deterministic fake-generator tests, dev smoke outputs.

**Acceptance gate:** Tests demonstrate the requested 9 conditions differ in the intended way; results are not merely cut-off text. Any incompatibility with the small model's learned markers is documented.

### Stage 8 — Frozen MATH-500 test evaluation

**Status:** [ ] Freeze, fixed-slot generation and local grader implemented and tested; benchmark remains sealed. See `docs/evaluation_protocol.md`.

**Goal:** Evaluate every predeclared condition on exactly the same benchmark, without test-set tuning.

**Tasks**
- [ ] Review the frozen-config checklist and confirm model/dataset revisions, dataset IDs, prompt templates, answer grader, budgets, decoder, timeout, and safety limits.
- [ ] Use a deterministic answer-extraction and mathematical-equivalence grading approach compatible with MATH-500; unit-test it on **synthetic examples and development data**, not by optimizing for test outputs.
- [ ] Evaluate A, B, C, D256, D512, D1024, D2048, E1, E2 on the **same 500 test IDs**; if resource failures prevent full completion, mark the study **partial** and report per-condition denominators.
- [ ] Save a JSONL record for **every question and condition** with: question ID, source revision, model/adapter revision, condition, prompts/config IDs, raw generated text or segments, extracted answer, gold answer, correctness, extraction status, reasoning/generated token counts, latency, and termination reason.
- [ ] Distinguish model errors, extraction errors, and infrastructure failures; do not quietly omit difficult cases.
- [ ] Keep all evaluation code and config checksummed/versioned.

**Artifacts:** `results/test/<run_id>/*.jsonl`, run manifest, per-condition score summaries.

**Acceptance gate:** All planned conditions complete or are clearly identified as incomplete; every score traces back to saved per-question outputs.

### Stage 9 — Analysis, significance and error study

**Status:** [ ] Paired statistics, transition/error cases and PNG/SVG export implemented and tested on synthetic fixtures; real results pending Stage 8.

**Goal:** Answer the research question rigorously, including cases where extra reasoning hurts.

**Tasks**
- [ ] Compute accuracy = correct / fixed evaluated denominator; report exact denominator for each condition.
- [ ] Report mean and median reasoning tokens, total generated tokens, latency and answer-extraction failure rate; report safety-cap hit rates.
- [ ] Plot **accuracy vs measured reasoning tokens**, **accuracy vs latency**, and performance differences relative to **C natural stopping**.
- [ ] Use **paired bootstrap** confidence intervals for per-question accuracy differences when comparing conditions on the same IDs; state resampling method, seed, iterations, and confidence level.
- [ ] Make a transition table for each intervention: `wrong→right`, `right→wrong`, `wrong→wrong`, `right→right`, and extraction failures.
- [ ] Analyze representative examples where continuation **fixes**, **spoils**, and **does not change** the answer.
- [ ] If reporting the original paper's **Control, Scaling, Performance** metrics, match its definitions exactly; otherwise use ordinary names, not borrowed labels.
- [ ] Discuss the smaller model, context truncation/exclusions, benchmark leakage checks, the number of seeds, and runtime/hardware trade-offs.

**Artifacts:** reproducible analysis script, tables, figures, error-case appendix, stats summary.

**Acceptance gate:** Tables and charts can be regenerated from raw JSONL alone; claims match the observed data and include uncertainty.

### Stage 10 — Conference-style paper, demo and handoff

**Status:** [ ] Evidence-driven paper/demo/handoff writer implemented; actual draft waits for saved results. Reproducibility and license metadata documented.

**Goal:** Produce a defensible research deliverable, even if the result is a performance plateau or decline.

**Tasks**
- [ ] Write abstract, introduction, related work, method, experiment setup, results, analysis, limitations and conclusion.
- [ ] Clearly distinguish **original s1 contributions** from **our small-model reproduction/adaptation**.
- [ ] Report training hardware, GPU hours, all departures, dataset and model revisions, any context filtering, baseline prompts and grader details.
- [ ] Include experiment table, accuracy/cost plots, confidence intervals, and qualitative examples; do not selectively present positive cases.
- [ ] Prepare reproducibility README, frozen configs, scripts, final LoRA weights or release instructions, and repository license/citation checks.
- [ ] Select a target venue only after results; check submission requirements, ethics, page limits and deadlines at that time.
- [ ] Prepare a concise presentation/demo that can re-run a few example questions without claiming it represents the full benchmark.

**Artifacts:** conference-style paper draft, final README, presentation/demo notes, clean public repository/package.

**Acceptance gate:** A reader can reproduce main results given documented resources; the paper makes calibrated claims supported by saved outputs.

---

## 5. Resource decisions and stop/go gates

- **Local compute:** User requires the Apple M1 MacBook Air with 8 GB unified memory and a 7-core GPU. Base inference and stable complete-example pilot training passed; full training/evaluation are active or queued. See `docs/hardware.md`.
- **Memory feasibility:** The proposal's **16–24 GB** target was provisional and does not describe the available Mac. Measure actual peak memory with s1K sequence lengths at Stage 5; document any required precision/backend/context changes.
- **Original full-context upper bound:** Official training used 32,768-token blocks. This does *not* mean our hardware can handle that context; it is why length auditing is mandatory.
- **Cost controls:** First run 1 sample → 5 samples → 20–50-example pilot → development run → final training → test matrix.
- **OOM policy:** Record the failed configuration and a trace, try supported memory optimizations, and explicitly report any change to training data coverage or context length.
- **Checkpoint policy:** Save the final adapter and all settings. Checkpoint formats are not interchangeable without verification.
- **Compute comparison:** Record throughput and latency on the actual evaluation hardware; do not equate generated tokens with measured wall time or claim universal speed-ups.

---

## 6. Data/result schemas (guidance)

### One evaluation record (illustrative; keys are suggested, not an implemented format)

```json
{
  "question_id": "example_id",
  "condition": "D512",
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "adapter_id": "path_or_revision_or_null",
  "dataset_revision": "commit_or_revision",
  "run_config_id": "config_hash",
  "reasoning_budget": 512,
  "answer_budget": null,
  "forced_wait_count": 0,
  "reasoning_tokens": null,
  "total_generated_tokens": null,
  "latency_seconds": null,
  "termination_reason": null,
  "raw_output": null,
  "extracted_answer": null,
  "gold_answer": null,
  "is_correct": null,
  "extraction_status": null
}
```

All `null` fields must become actual measurements/outputs on successful runs or explicit unavailable values with an error status. Do not fabricate them. Store prompt IDs/hashes and full output segments as needed to reproduce interpretation.

### Must-have final figures

1. Accuracy by condition, with denominators and paired uncertainty where valid.
2. Accuracy vs mean reasoning tokens (measured, not nominal cap).
3. Accuracy vs mean latency on the measured hardware.
4. Rescued vs spoiled answers relative to natural stopping.

---

## 7. Freeze checklist (mandatory before Stage 8)

- [ ] Base model ID/revision and final adapter revision recorded.
- [ ] Source s1K revision and eligibility/exclusion counts fixed.
- [ ] Development-only decisions completed and documented.
- [ ] Dataset/test benchmark revision and ordered question IDs fixed.
- [ ] Prompt templates and reasoning/answer delimiter protocol frozen.
- [ ] Decoding setup, budgets, safety ceiling, final answer allowance frozen.
- [ ] Answer extraction and equivalence grader frozen with tests.
- [ ] Raw-result schema, failure handling and output locations fixed.
- [ ] No MATH-500 score has been used for selecting these settings.

---

## 8. Minimal stage-by-stage Codex workflow

### At the start of a session

Paste this instruction into Codex (along with the project folder containing this file):

> Read `plan.md` and `NLP_Proposal.pdf` if available. Identify the **first unchecked stage** in `plan.md`. Implement **only that stage**, following its tasks and acceptance gate. Before coding, list the files you will touch. Keep the Python code simple, modular and readable. Run its tests and show actual commands/results. Do **not** train, benchmark, download multi-GB models, or advance to the next stage unless the active stage explicitly requires it. At the end, summarize changes, test results and blockers, update the stage checklist and append a dated progress-log entry, then **stop**.

### At the end of a session

Ask Codex to produce this short report (filled with facts, not guesses):

```text
Stage:
Files changed:
Commands executed:
Tests passing/failing:
Artifacts produced:
Hardware/time/storage measurements (if any):
Deviations from plan:
Open blockers:
Suggested next stage:
```

### Recommended immediate action

**Start Stage 1 only.** Do not attempt full LoRA training until Stage 2 length audit, Stage 4 loss-mask validation and Stage 5 memory pilot have passed.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A model that is small enough to fit still OOMs on long s1K examples | Audit sequence lengths, profile a pilot, preserve complete examples where possible and report exclusions |
| LoRA training executes but teaches the wrong tokens | Test `labels == -100` masks with manually decoded examples and exact start/end boundaries |
| Budget forcing is only prompt engineering or truncation | Test genuine stop interception, continuation, answer transition and per-phase budgets |
| Small base model does not use s1 delimiters naturally | Validate the learned protocol after fine-tuning; record failure rather than pretending it matches original behavior |
| MATH-500 leaks into development | Freeze data/grader/config before scoring test set; use only s1K development for tuning |
| Approximate answer extraction inflates/deflates score | Unit-test math equivalence and report extraction failures separately |
| Small score differences are overinterpreted | Paired question-level comparisons, confidence intervals, transparent negative examples |
| Training and evaluation become too expensive | Pilot first; log measured cost; prioritize one clean full run over many untracked runs |
| Research is just a reproduction with little novelty | Center the *small-model limits and compute/accuracy trade-off*, with controlled baselines and failure analysis |

---

## 10. Open decisions to resolve **during** implementation

These are deliberately marked **TBD**, not filled with guesses:

- **Compute environment:** local Apple M1 MacBook Air with 8 GB unified memory, per user instruction. Determine the viable backend, precision and context coverage through measured pilots; do not assume a remote GPU. See `docs/hardware.md`.
- **Seed for the fixed 900/100 split:** resolved as 42 in Stage 2; immutable original and clean eligibility manifests are under `manifests/`. Four benchmark-overlap examples are excluded separately (896 eligible train / 100 development; 996 final before context filtering).
- **Sequence-length policy:** resolved: 9,216 ceiling, complete eligible sequences, no truncation/context exclusions; longest measured 9,109.
- **Training learning rate, epoch count and checkpoint-selection rule:** predeclared two one-epoch trials at 5e-5 / 1e-4; select complete 100-row token-weighted NLL, tie to lower rate. Actual selection pending.
- **Whether QLoRA is necessary:** resolved by measured failures/feasibility: local four-bit affine/group-64 conversion, FP32 activations.
- **Exact reasoning/answer delimiter protocol for our fine-tuned 1.5B model:** prove through formatter/unit tests and generated samples.
- **Answer budget and absolute generation safety ceiling:** decide on development runs in Stage 7.
- **Math equivalence grading implementation and known limitations:** validate and freeze before test.

No unresolved choice is permission for Codex to select arbitrary values without documenting them.

---

## 11. Progress log

| Date | Stage | Result / evidence | Decision / blocker |
|---|---|---|---|
| 2026-11-14 | Planning | `plan.md` created from group proposal and public original s1 resources | Implementation not yet started |
| 2026-11-14 | Hardware inspection | Recorded local M1/8 GB/7-core GPU specifications and successful MPS tensor check in `docs/hardware.md` | Local-only compute required; approximately 12 GiB free disk; model inference/training feasibility unmeasured; no stage marked complete |
| 2026-11-14 | Stage 1 complete | Clean editable install; 12 tests pass; environment JSON reports; immutable upstream source pins and original-code audit; see `docs/stage1_report.md` | Approximately 17 GiB free disk; MPS detected natively; no weights/tokenizers/data rows downloaded; Stage 2 next; training feasibility remains unmeasured |
| 2026-11-14 | Stage 2 complete | Pinned data/tokenizer acquisition; 1,000 valid rows; seed-42 split; four reviewed benchmark-overlap exclusions; complete-sequence audit; isolated replay matches; 30 tests pass; see `docs/stage2_report.md` | 896/100 eligible split; 996 final before context exclusions; median 5,898/max 9,109 tokens; no truncation, weights, training or scores; Stage 3 next; GPU training feasibility unmeasured |
| 2026-11-14 | Stage 3 active / independent Stage 4 checks | Baseline implementation, queued model smoke workflow, native FP16 SDPA probe; 996 complete examples pass loss-mask audit; full decoded sample reviewed; 52 tests pass | Checkpoint transfer still active; actual model generation pending; no training or scores; context ceiling remains unset; stage-order deviation justified by independent tokenizer work during network wait |

| 2026-11-14 | Stages 3–4 complete / Stage 5 active | Verified full checkpoint; 28/28 native FP16 MPS generations; every repeat and four fresh-load comparisons identical; exact format/mask audit all 996 eligible; 55 tests pass | `docs/stage3_report.md`; no accuracy scores; first complete-example BF16 LoRA backward profile started; no truncation/context exclusions |

| 2026-11-15 | Stage 5 complete | Native MLX QLoRA/FP32 activation path; five stable pilot updates; exact fresh-process tensor/generation reload; complete 712–9,109-token backward profiles; 68 ordinary tests (4 skipped), 4 native tests passed | Source-traced 4-bit/group-64 conversion; 996 eligible examples retained without truncation; longest peak active allocation 5.833 GB; no final training/test accuracy yet; Stage 6 active |

| 2026-11-15 | Stage 6 running / Stages 7–10 implementation | Resumable full trial active; 106 tests successful in the grader environment (6 optional/native skips); separate 7/7 grader tests pass; PEFT check covered in the inference environment; budget engine, sealed evaluation, paired analysis and report automation implemented | Native protocol/model results still pending; no benchmark scores or completion claim; current status via `scripts/status.py` |

| 2026-11-16 | First-trial review boundary | Current training continues; only scheduler suspended; downstream controller honored PAUSE; dedicated verification/base-validation/review controller tested | User will decide on trial 2 after complete held-out validation; no paper-score threshold, benchmark access, training-config change or permanent trial cancellation |

| 2026-11-16 | First-trial review ready | 896/896 training examples, 56 finite-gradient updates, 100/100 held-out targets, exact fresh-process reload; base/adapted NLL 0.563623/0.479717 over identical 555,371 targets | 14.9% relative NLL reduction is not benchmark accuracy; review artifacts verified; no second trial or final training started |

| 2026-11-16 | Existing adapter accepted | User chose this run only; reviewed 896-example adapter adopted without training; 100 development examples stay held out | Trial 2 skipped and fresh 996-example retraining deferred; Stage 7 protocol validation next; benchmark remains sealed |

| 2026-11-16 | Stage 7 readout investigation | At 43/45 native outputs, five extractable direct answers and zero extractable structured answers; immediate role stops/repetition observed | Added native readout gate before benchmark freeze; bounded same-adapter development-only prefix probes queued; no retraining or benchmark access |

| 2026-11-16 | Stage 7 readout revision ready | V1 complete: 45 outputs, zero extracted structured answers; 15 same-adapter probes complete; Final Answer label gave five short EOS-terminated bodies | V1 evidence retained; v2 common readout and explicit extraction provenance prepared; full native 45-output revalidation required; no training or benchmark access |

---

## Sources

- Group's `NLP_Proposal.pdf` (primary requirements)
- Original s1 paper, v1: https://arxiv.org/abs/2501.19393v1
- Official s1 implementation: https://github.com/simplescaling/s1
- Original training script: https://raw.githubusercontent.com/simplescaling/s1/main/train/sft.py
- Original training launcher: https://raw.githubusercontent.com/simplescaling/s1/main/train/sft.sh
- Official evaluation examples: https://raw.githubusercontent.com/simplescaling/s1/main/eval/commands.sh
- Original s1K: https://huggingface.co/datasets/simplescaling/s1K
- MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- Base model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

