# s1-style test-time scaling on Qwen2.5-1.5B

Completed single-run research project by **Faisal Imran, Muhammad Ibrahim and Hamza Saeed**. Training ran on an Apple M1 MacBook Air with 8 GB unified memory. The separately frozen full evaluation ran on one RTX 4090 using FP16/vLLM. This is a resource-constrained adaptation of original s1, not a reproduction of the 32B model's scores.

## Measured result

All **500 MATH-500 questions x nine conditions = 4,500 outputs and resolved grades** are verified. No infrastructure failures or timeouts occurred. All failures remain in the denominator.

| Policy | Accuracy | Mean generated tokens |
|---|---:|---:|
| Base direct A | 18.0% | 344.9 |
| Base reasoning B | 43.0% | 1,413.9 |
| Adapted bounded natural C | 28.6% | 4,178.5 |
| Adapted cap 256 | 20.6% | 283.0 |
| Adapted cap 512 | 27.0% | 448.2 |
| Adapted cap 1024 | 28.6% | 749.1 |
| Adapted cap 2048 | 28.2% | 1,236.5 |
| Adapted up to one Wait | 27.2% | 5,059.5 |
| Adapted up to two Waits | 28.8% | 5,441.0 |

D1024 matches C's observed accuracy with **82.1% fewer output tokens**; this is not proof of statistical equivalence. E2 adds one net correct answer with 30.2% more tokens; its paired difference is +0.2 pp, 95% interval [-2.2, 2.6]. There is no clear continuation benefit in this run. B/C change prompting/readout and adapter together, so their difference is not a pure tuning effect. About one fifth of E1/E2 first reasoning segments differ from C before intervention, further limiting causal interpretation.

## Deliverables

- [Paper, editable Markdown](docs/paper.md); rendered local PDF: `output/pdf/s1-small-model-report.pdf`.
- [Complete results, uncertainty and figures](docs/results.md), [machine-readable table](docs/final_results.json), and [all 24 qualitative cases](docs/error_cases.md).
- [Reproduction and asset instructions](docs/reproducibility.md), [demo/presentation notes](docs/demo_notes.md), and [publication assessment](docs/publication_readiness.md).
- [Ten-stage plan and execution history](plan.md), [original proposal](NLP_Proposal.pdf), [training-format audit](docs/training_format_audit.md), [first-run review](docs/first_trial_review.md), and [local hardware measurements](docs/hardware.md).
- [RunPod vLLM notebook](notebooks/runpod_inference.ipynb) and [guide](docs/runpod_inference.md). It uses a separate workspace and saves durable results; rerunning is optional.
- Local complete handoff: `output/s1-project-handoff.zip`; checksums/file inventory: [docs/handoff.json](docs/handoff.json). Models, datasets, raw traces and rendered packages stay outside Git.

The accepted adapter learned from **896 complete s1K training examples**, with 100 development examples retained; four training/benchmark overlaps were conservatively excluded. No sequence was truncated or dropped for length. One epoch / 56 updates completed with rank-16 attention LoRA and affine Q4/group-64 base weights, FP32 activations/adapters. Teacher-target development NLL fell 14.9%, which is not answer accuracy. The user chose this run only; the second trial and fresh all-data retraining were skipped.

## Inspect or reproduce without a GPU

From this repository root, use the saved local assets and `.venv-eval`:

```bash
.venv-eval/bin/python scripts/audit_runpod_archive.py 'nlp-runpod-vllm-results (1).zip'
MPLCONFIGDIR=tmp/matplotlib .venv-eval/bin/python scripts/finalize_study.py --pdf
.venv-eval/bin/python scripts/demo_results.py
PYTHONPATH=src .venv-eval/bin/python -m unittest discover -s tests -v
```

A source clone alone contains no raw predictions or weights. See the reproduction guide for fresh-environment locks, exact assets, optional GPU conversion/generation, and a bounded live Mac demo. The downloaded result ZIP is verified by SHA-256 and the complete analysis was independently recomputed. Its 4,500 resolved grades are final; the archived pre-grading generation summary retains its historical `test_scored: false` field.

**Local generation is intentionally stopped.** Its 1,287 partial outputs are preserved, ungraded and excluded from these scores. The PAUSE marker remains in place; do not resume the old training/benchmark scheduler. No further heavy training or generation is required for the completed project deliverables.

## Provenance and release status

GPU source commit: `1cae805a2a29b7ba769838380f2fd71f058fa824`. Model and dataset revisions, distinct Mac/GPU freezes, adapter fingerprint, package versions and hardware/runtime accounting are documented in the paper and handoff. Final evaluation reconstructs the frozen Q4 base into FP16 and converts the learned adapter; it does not substitute the original dense base or retrain. The GPU generation window is 2.29 hours, not total rental time; per-output concurrent latency must not be summed as GPU-hours.

Original reference: Muennighoff et al. (2025), [s1: Simple test-time scaling, original version](https://arxiv.org/abs/2501.19393v1), [official implementation](https://github.com/simplescaling/s1). Original s1K is used, not s1K-1.1. Source revisions are pinned in `configs/sources.json`.

Inspected Qwen/s1K metadata declare Apache-2.0; the inspected MATH-500 metadata does not declare a license. No project-wide license grant, model release, or venue submission is implied. Keep the local package private until redistribution terms are reviewed. `CITATION.cff` identifies this project report without inventing a DOI or publication venue.
