# Stage 1: original s1 implementation audit

Reviewed 2026-11-14. Specification: `NLP_Proposal.pdf` and `plan.md`. This project studies a 1.5B LoRA adaptation; it does not reproduce the original 32B training or claim its scores.

## Source identity

The GitHub API resolved `simplescaling/s1` main to `77272c6e925d610257a50b520bad15330b513389`. All repository files below were read at that immutable commit. This is the available repository snapshot, not a claim that it is the paper-v1 release commit. It contains later s1.1 additions; the original s1 formatter and s1 evaluation sections are the references for this study.

Hub API metadata resolved the following revisions. These are planned source pins, not records of completed model/data downloads. The training dataset's actual schema and row count remain to be validated in Stage 2.

| Source | Revision |
|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` model and tokenizer | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Original `simplescaling/s1K` | `278d72baaa2b887a7e76a70a0ae254a5a45536e4` |
| `HuggingFaceH4/MATH-500` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |

Machine-readable pins and API provenance: [`configs/sources.json`](../configs/sources.json). Downloads in later stages must specify the revision explicitly. No benchmark questions or answers were read for this audit.

## Training script

[`train/sft.py`](https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/train/sft.py) loads a causal model and preformatted `text` dataset. It uses TRL's older `DataCollatorForCompletionOnlyLM`, Qwen user/assistant templates, and `<|fim_pad|>` padding to mask non-assistant labels. There is no LoRA attachment in this script. Its fallback evaluation dataset is the training data when no test split exists; we will instead use the fixed development partition.

[`train/sft.sh`](https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/train/sft.sh) launches `torchrun`, counts GPUs with `nvidia-smi`, enables full-shard FSDP and BF16, and sets a 32,768-token block size for Qwen2.5-32B-Instruct. This launcher is unsuitable for our M1. Its hyperparameters are reference settings, not selected settings for our study.

## Released formatting

[`data/tokenization.py`](https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/data/tokenization.py) uses the question as the user content. The assistant content is assembled as:

```text
<|im_start|>think\n
{thinking_trajectories joined with newlines, stripped}
\n<|im_start|>answer\n
{attempt, stripped; prepend "Answer: " if "Answer:" is absent}
```

The complete conversation is serialized with `apply_chat_template(..., tokenize=False)`. The formatter uses the 32B tokenizer and publishes a tokenized dataset; we must reformat with the pinned 1.5B tokenizer locally, without uploading. `preprocess()` is defined but not called by this formatting path. Neither `<think>...</think>` nor an invented marker pair is an equivalent replacement. Verify composite-marker tokenization, assistant boundaries, padding and end-of-answer supervision in Stage 4. Displayed `\n` means a newline, not a literal backslash in the training text.

## Inference and evaluation

The original-s1 budget-forcing example in the [`README`](https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/README.md) prefills a thinking marker, intercepts structural stop tokens, appends `Wait`, and resumes generation with a remaining thinking allowance. The final generation uses an end-of-message stop. Its illustrative allowances are not our frozen protocol. The README also contains s1.1 defaults; do not copy those model/data IDs. It identifies a modified evaluation harness based on upstream commit `4cec66e4e468d15789473d6d63c3a61a751fa524`.

[`eval/commands.sh`](https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/eval/commands.sh) includes both releases and uses vLLM, tensor parallelism and optional external grading machinery. Original-s1 arguments distinguish thinking caps and ignored stops with continuation strings. We will implement a small local evaluator with explicit phase budgets and deterministic grading. No API credentials or external grading calls are needed for the core study. Exact harness boundary behavior still requires inspection in Stage 7; this audit is not a completed engine validation.

## Supported replacement paths

The pinned repository's [`requirements.txt`](https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/requirements.txt) pins TRL 0.12.0, Transformers 4.46.1 and Torch 2.5.1 alongside a broad inference/API stack. We did not install it.

- [PEFT's documented LoRA workflow](https://huggingface.co/docs/peft/en/quicktour) uses `LoraConfig`, `get_peft_model`, and adapter `save_pretrained`, with Transformers Trainer or a custom PyTorch loop. Explicit token labels and a tested collator avoid depending on the original TRL masking API.
- [Current TRL SFT documentation](https://huggingface.co/docs/trl/en/sft_trainer) provides completion-only/assistant-only loss controls. These require appropriate data/template support; setting a flag does not substitute for our Stage 4 label tests. No TRL version is selected or installed here.
- [PyTorch 2.9 MPS documentation](https://docs.pytorch.org/docs/2.9/notes/mps.html) documents moving models and tensors to `mps`. Basic access passed our earlier tensor test; model-level operation and precision compatibility remain unverified.
- [MLX-LM](https://github.com/ml-explore/mlx-lm) documents Apple-silicon inference and low-rank fine-tuning, including quantized models. It is a fallback candidate if measured PyTorch memory is inadequate. Changing framework/quantization requires source provenance, loss-mask verification and matched comparison conditions. MLX is not installed yet.

The Stage 1 package therefore has no runtime ML dependencies. Heavy dependency selection and a complete ML lock are deferred until the relevant workflow is tested. The existing global packages in `docs/hardware.md` are observations, not a proven training environment.

## Adaptation decisions and remaining evidence

- Local M1/8 GB memory is the compute constraint. Replace CUDA/FSDP assumptions with a tested local backend. Do not assume that ordinary CUDA bitsandbytes instructions apply to this Mac.
- Preserve original s1K, original teacher `attempt`, official structural format, assistant-only loss and the planned model identity.
- Build a fixed 900/100 split before tuning. Determine overlap and complete-example length eligibility before training; disclose exclusions.
- Keep reasoning and answer budgets separate, and log model-generated tokens separately from injected control tokens. `Wait` is conditional on an attempted reasoning stop.
- Stage 2 must resolve lengths and storage needs; Stage 4 must resolve token boundaries and masks; Stage 5 must measure forward/backward feasibility and adapter reload; Stage 7 must resolve stop behavior.
- The full 4,500-prediction test matrix remains unrun. Final reporting must distinguish planned work, completed work and measured hardware limitations.

