"""Write a conference-style research draft and handoff from actual saved results."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import file_sha256


def number(value, places=2):
    return f"{value:.{places}f}" if value is not None else "N/A"


def publish_status(root, report, run_id):
    """Refresh human-facing progress only after actual report artifacts exist."""
    coverage = report["status"]
    message = (f"Stages 1–7 completed. The nine-condition MATH-500 study is **{coverage}**. "
               "Saved analysis and the local research draft are available in [`docs/results.md`](docs/results.md), "
               "[`docs/paper.md`](docs/paper.md) and [`docs/handoff.json`](docs/handoff.json). "
               "Read exact per-condition coverage in the results before interpreting any score. "
               "Hardware and measured runtime appear at the end of the paper. No venue submission or public release has been performed.")
    readme = root / "README.md"
    if readme.exists():
        before, section = readme.read_text().split("## Current status\n\n", 1)
        _, remaining = section.split("\n\n", 1)
        readme.write_text(before + "## Current status\n\n" + message + "\n\n" + remaining)
    plan = root / "plan.md"
    if plan.exists():
        lines = plan.read_text().splitlines()
        stage = None
        statuses = {6: "[x] Complete: two full development trials, declared selection and fresh final adapter verified.",
                    7: "[x] Complete: native held-out protocol validation and pre-test freeze recorded; see frozen diagnostics for learned-marker limitations.",
                    8: ("[x] Complete: all 4,500 planned model outputs and grades resolved." if coverage == "complete" else
                        "[ ] Partial: planned slots retained; unresolved outcomes and exact coverage reported."),
                    9: "[x] Analysis generated from saved evaluated records, including coverage, paired uncertainty and transitions.",
                    10: "[x] Local research draft, hardware appendix, error cases and demo notes generated. Venue/public release remains separate."}
        for index, line in enumerate(lines):
            if line.startswith("### Stage "):
                stage = int(line.split()[2])
            if line.startswith("**Status:**") and stage in statuses:
                lines[index] = "**Status:** " + statuses[stage]
            if line.startswith("**Planning status:**"):
                lines[index] = f"**Planning status:** Local research handoff generated; benchmark coverage {coverage}; see `docs/handoff.json`"
        plan.write_text("\n".join(lines) + "\n")


def make_paper(report, stage6, freeze, run_id, hardware):
    complete = report["status"] == "complete"
    scores = report["conditions"]
    c, b = scores["C"]["accuracy"], scores["B"]["accuracy"]
    abstract_result = (f"The natural reasoning condition achieved {100*c:.2f}% accuracy, compared with {100*b:.2f}% for the reasoning-prompted base. "
                       if c is not None and b is not None else
                       "Coverage is incomplete, so a full benchmark-accuracy comparison is unavailable. ")
    abstract_result += "All interventions are reported alongside actual token use, latency, extraction failures and paired uncertainty."
    lines = ["# s1 on an 8 GB Mac: A Resource-Constrained 1.5B QLoRA Study", "",
             "Faisal Imran · Muhammad Ibrahim · Hamza Saeed", "", "**Research draft.** Venue-specific submission formatting has not been selected.", "",
             "## Abstract", "",
             "We study an adaptation of s1-style supervised reasoning and test-time budget forcing to Qwen2.5-1.5B-Instruct on a local Apple M1 MacBook Air with 8 GB unified memory. "
             "The study uses original s1K, a fixed development split, conservative benchmark-overlap exclusions, and fresh final retraining after development selection. "
             "Measured low-precision numerical failures and memory costs motivated a source-traced four-bit MLX base with FP32 activations. " + abstract_result,
             "", "## 1. Introduction", "",
             "Test-time reasoning can trade additional computation for answer quality, but the trade-off need not transfer from a large full-model adaptation to a small low-rank adaptation. "
             "Our question is whether original s1K supervision and bounded continuation improve mathematical accuracy enough to justify their measured local cost. A null or negative result is valid. "
             "This study changes model size, training method, precision and hardware, and does not claim an exact reproduction of original s1 performance.",
             "", "## 2. Related work", "",
             "Muennighoff et al. introduced the s1 data-and-budget-forcing recipe [1,2]. We retain original s1K rather than the later s1K-1.1 release, and adapt the recipe to the smaller Qwen checkpoint [3]. "
             "The held-out evaluation uses the fixed MATH-500 test split [4]. Mathematical equivalence is evaluated locally with a pinned Math-Verify stack [5].",
             "", "## 3. Method", "",
             "The immutable original split is 900 train / 100 development, seed 42. Four reviewed benchmark-overlap examples are excluded separately, leaving 896 training, 100 development and 996 final-training examples. "
             "Lexical screening cannot rule out paraphrase overlap or pretraining contamination. One normalized duplicate question pair remains entirely in training and is disclosed. "
             "All complete original teacher reasoning and `attempt` targets are preserved; their mathematical correctness is not assumed. Assistant-only causal loss masks prompts and added padding.", "",
             "LoRA targets q/k/v/o in all 28 attention layers, rank 16, alpha 32 (MLX scale 2.0), dropout 0.05 and 4,358,144 FP32 trainable parameters. "
             "The same pinned base is locally converted to four-bit affine weights with group size 64. FP32 activations, decoder checkpointing and chunked vocabulary loss preserve full sequences. "
             "The 9,216-token ceiling covers all 996 eligible examples; no sequence is truncated or excluded for length. "
             "Both pretrained and adapted final evaluation use the identical conversion and activation precision.", "",
             "Two predeclared one-epoch development trials use learning rates 5e-5 and 1e-4, micro-batch 1, accumulation 16, bias-corrected AdamW and token-weighted loss. "
             "The lowest complete 100-example development negative log-likelihood selects the recipe; an exact tie favors the lower learning rate. "
             f"The selected learning rate was {stage6['selected_learning_rate']}. Final training starts from fresh base/adapter initialization on all 996 examples. "
             "Budget decisions use the selected 896-row adapter, keeping those 100 development examples held out for protocol development. "
             "The final all-data adapter does not choose inference settings. All choices freeze before benchmark answers are opened for evaluation.", "",
             "A/B are direct and reasoning-prompted pretrained baselines. C uses natural reasoning termination in a two-phase s1 protocol. "
             "D256/D512/D1024/D2048 bound reasoning; E1/E2 intercept up to one/two actual reasoning stops and append a visible newline plus `Wait`. "
             "All structured conditions use a common training-format answer readout. EOS and im_start are recorded separately. "
             f"Natural reasoning has a {freeze['protocol']['natural_reasoning_ceiling']}-token safety allowance, answers {freeze['protocol']['answer_budget']} tokens, and the absolute safety ceiling is {freeze['protocol']['absolute_safety_ceiling']}. "
             "Removed stop markers remain charged to emitted-token counts. Injected controls are logged separately. A/B are unstructured, so their reasoning-token counts are unavailable. "
             "Conditions rotate cyclically within each fixed question order, and two models remain resident to avoid per-condition reload latency.",
             "", "## 4. Experimental setup", "",
             "The exact source, tokenizer, dataset, conversion, adapter, decoder and grader hashes are in `manifests/evaluation_freeze.json`. "
             "All conditions use greedy generation, no KV quantization and the same local backend. The first five fixed development questions validate the protocol, without benchmark scoring. "
             "The final matrix retains 500 planned IDs for each of nine conditions, including explicit failures. Successful model outputs are never rerolled based on their answer; infrastructure retries are bounded and retained.", "",
             "The answer extractor accepts the last balanced boxed expression or an explicit answer line and never guesses an intermediate trailing number. "
             "Math-Verify 0.9.0, latex2sympy2_extended 1.11.0, ANTLR 4.13.2 and SymPy 1.14.0 use frozen parser and comparison settings. "
             "Extraction failures count as incorrect. Infrastructure and unresolved grader outcomes have explicit denominators; incomplete conditions do not receive a full-coverage accuracy. "
             "Differences against C use paired question-ID bootstrap resampling, seed 42, 10,000 draws and 95% percentile intervals. Intervals are descriptive and unadjusted for multiple comparisons.",
             "", "## 5. Results", "", f"Coverage status: **{report['status']}**.", "",
             "| Condition | Correct | Resolved / planned | Accuracy | Mean reasoning tokens | Mean total generated tokens | Mean generation seconds | Extraction failures |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in scores:
        row = scores[key]
        accuracy = number(100 * row["accuracy"]) + "%" if row["accuracy"] is not None else "Incomplete"
        lines.append(f"| {key} | {row['correct']} | {row['resolved_grades']}/500 | {accuracy} | {number(row['mean_reasoning_tokens'],1)} | {number(row['mean_total_generated_tokens'],1)} | {number(row['mean_latency_seconds'],1)} | {row['extraction_failures']} |")
    lines += ["", "| Condition vs C | Difference (percentage points) | Paired 95% interval | Paired resolved IDs |", "|---|---:|---|---:|"]
    for key, row in report["paired_vs_C"].items():
        delta = 100 * row["accuracy_difference"] if row["accuracy_difference"] is not None else None
        interval = "[" + ", ".join(number(100 * value) for value in row["confidence_interval"]) + "]" if row["confidence_interval"] else "N/A"
        lines.append(f"| {key} | {number(delta)} | {interval} | {row['paired_resolved_denominator']} |")
    for name, caption in (("accuracy_by_condition", "Accuracy with fixed coverage"), ("accuracy_vs_reasoning", "Accuracy versus measured reasoning-content tokens"),
                          ("accuracy_vs_latency", "Accuracy versus measured generation latency"), ("differences_vs_natural", "Paired differences relative to C"),
                          ("rescued_vs_spoiled", "Rescued and spoiled answers")):
        if name + ".png" in report["figures"]:
            lines += ["", f"![{caption}](../results/test/{run_id}/figures/{name}.png)", "", caption + "."]
    lines += ["", "## 6. Error and cost analysis", "",
              "Continuation may rescue an answer, spoil an initially correct answer, or leave it unchanged. The following counts retain all common resolved IDs; unresolved pairs are listed in the machine-readable analysis.", "",
              "| Intervention | Wrong to right | Right to wrong | Wrong to wrong | Right to right |", "|---|---:|---:|---:|---:|"]
    for key in ("D256", "D512", "D1024", "D2048", "E1", "E2"):
        t = report["paired_vs_C"][key]["transitions"]
        lines.append(f"| {key} | {t['wrong_to_right']} | {t['right_to_wrong']} | {t['wrong_to_wrong']} | {t['right_to_right']} |")
    lines += ["", "Representative cases are selected by the first lexicographic ID in each transition category, with unavailable categories explicitly retained. See `error_cases.md` and the immutable raw segments for evidence. "
              "Token costs are measured emitted tokens, including discarded stop markers. Nominal caps are not substituted for actual costs. Generation latency includes segment re-prefill and control overhead, and excludes initial model loading and symbolic grading.",
              "", "## 7. Limitations", "",
              "This is one training seed on a small, quantized, low-rank model. Paired question uncertainty does not quantify variation across training seeds, quantization methods or hardware. "
              "A/B versus C also changes prompting/structural protocol, so it does not isolate a pure causal effect of parameter tuning. "
              "The original teacher targets include mixed proof and numeric tasks and were not repaired or independently certified. "
              "Conservative extraction, finite generation/deadline limits and mathematical-parser limitations may reject otherwise meaningful responses. "
              "The benchmark can overlap pretrained knowledge even after the disclosed training-overlap screening. "
              "Local latency depends on the passive-cooled M1, other applications and swap usage. No universal speedup or exact original-s1 replication is claimed.",
              "", "## 8. Conclusion", "",
              (abstract_result + " The full nine-condition table and paired transition counts determine the observed accuracy/cost trade-off; additional reasoning is not assumed to help."
               if complete else "The saved study is partial. It documents actual coverage, failures and resource use without presenting unresolved predictions as a complete benchmark result."),
              "", "## Reproducibility", "",
              "The repository includes pinned sources, immutable eligibility, frozen settings, source-traced converted weights, independently reloadable MLX adapters, raw prediction slots/attempts, local grading and regenerable plots. "
              "Use the supplied FP32-activation loader with the identical converted base. Generic BF16 MLX loading or PEFT interchange is not equivalent. "
              "Base weights, raw datasets and large outputs remain outside Git. No external model/judge API or remote GPU is used.",
              "", "## References", "",
              "1. Muennighoff et al. (2025). *s1: Simple test-time scaling*, original version. https://arxiv.org/abs/2501.19393v1",
              "2. Official s1 implementation, pinned revision 77272c6e925d610257a50b520bad15330b513389. https://github.com/simplescaling/s1",
              "3. Qwen/Qwen2.5-1.5B-Instruct, pinned revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
              "4. HuggingFaceH4/MATH-500, pinned revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be. https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
              "5. Hugging Face Math-Verify, version 0.9.0. https://github.com/huggingface/Math-Verify",
              "6. Apple MLX-LM, version 0.32.2; MLX/Metal 0.32.4. https://github.com/ml-explore/mlx-lm",
              "7. Original simplescaling/s1K, revision 278d72baaa2b887a7e76a70a0ae254a5a45536e4. https://huggingface.co/datasets/simplescaling/s1K",
              "", "## Appendix A. Hardware and measured training runtime", "", hardware, "",
              "| Run | Training examples | Committed training seconds | Committed compute hours | Input tokens/second | Peak active MLX GiB |", "|---|---:|---:|---:|---:|---:|"]
    for row in [*stage6["trials"], stage6["final_training"]]:
        cursor = row["training_cursor"]
        seconds = cursor["training_seconds_committed"]
        lines.append(f"| {row['run_id']} | {row['training_examples']} | {seconds:.1f} | {seconds/3600:.3f} | {cursor['input_tokens_committed']/seconds:.2f} | {row['peak_mlx_active_bytes']/1024**3:.3f} |")
    lines += ["", "Training seconds sum committed training windows and exclude checkpoint serialization and validation. Failed/interrupted attempt time is retained separately. "
              "Compute hours are wall time during committed training windows on the local CPU/GPU system, not dedicated or billable GPU hours. "
              "Reported allocator peaks cover the recorded attempt, including validation; after an interruption they need not bound earlier attempts. "
              "MLX active allocation is not dedicated VRAM and must not be added to overlapping RSS. Its memory limit is a RAM/swap guideline. "
              "Evaluation memory samples include both resident model variants. The original 20-row technical pilot used 2.993 GB peak active allocation; the complete 9,109-token backward profile used 5.833 GB. "
              "These are measured local feasibility observations, not guarantees for other machines.", ""]
    diagnostics = freeze.get("development_protocol_diagnostics", {})
    if diagnostics:
        insertion = ("Held-out protocol validation recorded "
                     f"{diagnostics.get('marker_stops', 0)} im_start reasoning-stop events, "
                     f"{diagnostics.get('eos_reasoning_stops', 0)} outputs with reasoning EOS, and "
                     f"{diagnostics.get('wait_triggered_outputs', 0)} outputs with a triggered Wait. "
                     "These counts diagnose learned marker compatibility; they are not accuracy scores.")
        lines.insert(lines.index("## 6. Error and cost analysis") + 2, insertion + "\n")
        if diagnostics.get("marker_stops", 0) == 0:
            lines.insert(lines.index("## 7. Limitations") + 2,
                         "No internal reasoning role-marker stop was observed on the protocol development questions. "
                         "The controller relied on EOS/caps and common answer readout; the results do not establish faithful learned s1 boundaries.\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--study-prefix", default="local-study-v1")
    args = parser.parse_args()
    if any(Path(value).name != value or value in (".", "..") for value in (args.run_id, args.study_prefix)):
        parser.error("Use simple run IDs")
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/test" / args.run_id
    report = json.loads((directory / "analysis.json").read_text())
    if report["evaluated_sha256"] != file_sha256(directory / "evaluated.jsonl"):
        raise ValueError("Analysis does not match saved evaluated predictions")
    stage6 = json.loads((root / "results/study" / args.study_prefix / "stage6_summary.json").read_text())
    freeze = json.loads((root / "manifests/evaluation_freeze.json").read_text())
    hardware_text = (root / "docs/hardware.md").read_text()
    hardware = hardware_text.split("## Measured specifications\n", 1)[1].split("The MPS working-set", 1)[0].strip()
    paper = make_paper(report, stage6, freeze, args.run_id, hardware)
    (root / "docs/paper.md").write_text(paper)
    cases = ["# Deterministically selected error cases", "", "Raw answers below are extracted model outputs, not repaired solutions.", ""]
    for row in report["error_cases"]:
        cases += [f"## {row['condition']}: {row['category']}", ""]
        if not row["available"]:
            cases += ["No available example in this category.", ""]; continue
        cases += [f"Question ID: `{row['question_id']}`", "", f"Gold: `{row['gold_answer']}`", "",
                  f"C answer: `{row['natural_answer']}`", "", f"Intervention answer: `{row['intervention_answer']}`", "",
                  f"Reasoning tokens C/intervention: {row['natural_reasoning_tokens']}/{row['intervention_reasoning_tokens']}; forced waits: {row['forced_wait_count']}.", ""]
        for name, field in (("C visible excerpts", "natural_visible_segments"), ("Intervention visible excerpts", "intervention_visible_segments")):
            cases += ["### " + name, ""]
            for segment in row.get(field, []):
                cases += [f"{segment['origin']} / {segment['phase']} (last 1,000 characters of this segment):", "",
                          "````text", segment["excerpt"], "````", ""]
    (root / "docs/error_cases.md").write_text("\n".join(cases))
    (root / "docs/demo_notes.md").write_text("# Presentation and demonstration notes\n\n"
        "1. State the small-model accuracy/cost question and distinguish this adaptation from original s1.\n"
        "2. Explain the immutable split, four overlap exclusions, assistant-only labels and fresh final training.\n"
        "3. Show the complete nine-condition accuracy table and paired intervals from `results.md`.\n"
        "4. Show measured tokens/latency and both rescued and spoiled examples from `error_cases.md`.\n"
        "5. Present M1/8 GB hardware, actual runtime, numerical failures and the justified QLoRA/FP32 path.\n"
        "6. Discuss single-seed, teacher-quality, grading and contamination limitations.\n\n"
        "A single demo is illustrative and does not represent the benchmark:\n\n```bash\n"
        ".venv-mlx/bin/python scripts/demo.py --condition D512 --question 'What is 17 times 23?'\n```\n\n"
        "The demo uses the frozen final model/decoder and logs visible model segments and injected controls separately.\n")
    handoff = {"status": report["status"], "paper": "docs/paper.md", "results": "docs/results.md",
               "figures": str((directory / "figures").relative_to(root)), "error_cases": "docs/error_cases.md",
               "demo_notes": "docs/demo_notes.md", "final_adapter": stage6["final_training"]["adapter_directory"],
               "evaluation_freeze_sha256": file_sha256(root / "manifests/evaluation_freeze.json"),
               "free_disk_gib_at_handoff": shutil.disk_usage(root).free / 1024**3,
               "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "release_status": "Local research draft; no venue submission, public publishing or upstream-weight redistribution performed."}
    (root / "docs/handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
    publish_status(root, report, args.run_id)
    print(json.dumps(handoff, indent=2), flush=True)


if __name__ == "__main__":
    main()
