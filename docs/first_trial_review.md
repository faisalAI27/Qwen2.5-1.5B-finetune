# First development trial review

Status: **review completed; the existing adapter was accepted for evaluation**.

The complete 896-example trial and 100-example held-out validation finished, and fresh-process tensor/generation reload matched.

| Matched local condition | Held-out teacher-target NLL (lower is better) |
|---|---:|
| Original-derived Q4 1.5B base, FP32 activations | 0.563623 |
| First-trial adapted 1.5B model | 0.479717 |

Lower teacher-target NLL indicates better prediction of held-out supplied solutions; it is not mathematical answer accuracy or proof correctness.

The [original s1 paper, Table 1](https://arxiv.org/html/2501.19393v1) reports 92.6% MATH-500 accuracy for its 32B model without budget forcing and 93.0% with it. Those figures are contextual references, not directly comparable to this NLL or a stopping threshold for our learning-rate search. No MATH-500 answers were used for this decision.

The second trial, recipe freeze, fresh final training and benchmark remain on hold. A single-trial choice can save compute, but must be reported as a resource constraint; equality with an unrun learning rate cannot be established.

## User decision (2026-11-16)

Proceed with this existing run only. Trial 2 is skipped and fresh all-data retraining is deferred for this study. The evaluated adapter therefore trained on 896 examples, with 100 held out. The immutable decision is in `configs/single_run_decision.json`; the original review JSON is retained as its historical evidence. The earlier hold described above is resolved for inference and evaluation, with no permission for additional training.
