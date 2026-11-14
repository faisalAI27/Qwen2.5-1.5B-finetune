"""Regenerate all score/cost/statistical artifacts from saved evaluated JSONL."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.budget_forcing import CONDITIONS
from s1small.data import file_sha256, read_jsonl
from s1small.metrics import paired_comparison, summarize_condition


def analyze(rows, planned, settings):
    grouped = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    summaries = {condition: summarize_condition(records, planned) for condition, records in grouped.items()}
    comparisons = {condition: paired_comparison(grouped["C"], grouped[condition],
                  seed=settings["bootstrap_seed"], iterations=settings["bootstrap_iterations"], confidence=settings["confidence_level"])
                   for condition in CONDITIONS if condition != "C"}
    cases = []
    natural = {row["question_id"]: row for row in grouped["C"]}
    for condition in ("D256", "D512", "D1024", "D2048", "E1", "E2"):
        candidates = sorted(grouped[condition], key=lambda row: row["question_id"])
        for kind, a, b in (("rescued", False, True), ("spoiled", True, False), ("unchanged_wrong", False, False), ("unchanged_right", True, True)):
            match = next((row for row in candidates if row["status"] == "ok" and natural[row["question_id"]]["status"] == "ok"
                          and natural[row["question_id"]].get("is_correct") is a and row.get("is_correct") is b), None)
            if match is None:
                cases.append({"condition": condition, "category": kind, "available": False})
            else:
                reference = natural[match["question_id"]]
                cases.append({"condition": condition, "category": kind, "available": True,
                              "question_id": match["question_id"], "gold_answer": match.get("gold_answer"),
                              "natural_answer": reference.get("extracted_answer"), "intervention_answer": match.get("extracted_answer"),
                              "natural_extraction_status": reference.get("extraction_status"),
                              "intervention_extraction_status": match.get("extraction_status"),
                              "natural_reasoning_tokens": reference.get("reasoning_tokens"),
                              "intervention_reasoning_tokens": match.get("reasoning_tokens"),
                              "forced_wait_count": match.get("forced_wait_count", 0),
                              "natural_visible_segments": [{"phase": e["phase"], "origin": e["origin"], "excerpt": e["text"][-1000:]}
                                                           for e in reference.get("segments", [])],
                              "intervention_visible_segments": [{"phase": e["phase"], "origin": e["origin"], "excerpt": e["text"][-1000:]}
                                                               for e in match.get("segments", [])],
                              "selection": "lexicographically first question ID in this transition category; no positive-case preference"})
    return {"status": "complete" if all(value["complete"] for value in summaries.values()) else "partial",
            "planned_per_condition": planned, "conditions": summaries, "paired_vs_C": comparisons,
            "error_cases": cases, "settings": settings}


def plot_results(report, output):
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / "font-cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    data = report["conditions"]
    valid = [key for key in CONDITIONS if data[key]["accuracy"] is not None]
    if not valid:
        return []
    saved = []
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = output / (name + "." + extension)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            saved.append(str(path.name))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(valid, [100 * data[key]["accuracy"] for key in valid], color=["#64748b" if key in ("A", "B") else "#2563eb" for key in valid])
    ax.set(ylabel="Accuracy (%)", ylim=(0, 100), title=f"Fixed benchmark denominator: {report['planned_per_condition']} per complete condition")
    save(fig, "accuracy_by_condition")
    for x_field, name, xlabel in (("mean_reasoning_tokens", "accuracy_vs_reasoning", "Mean emitted reasoning-content tokens"),
                                  ("mean_latency_seconds", "accuracy_vs_latency", "Mean measured generation seconds")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for key in valid:
            if data[key][x_field] is None:
                continue
            x, y = data[key][x_field], 100 * data[key]["accuracy"]
            ax.scatter([x], [y], s=45)
            ax.annotate(key, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set(xlabel=xlabel, ylabel="Accuracy (%)", title="Measured cost, not nominal reasoning cap")
        if x_field == "mean_reasoning_tokens":
            ax.text(0.01, 0.01, "A/B omitted: unstructured reasoning tokens unavailable", transform=ax.transAxes, fontsize=8)
        save(fig, name)
    keys = [key for key in CONDITIONS if key != "C" and report["paired_vs_C"][key]["confidence_interval"] is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, key in enumerate(keys):
        row = report["paired_vs_C"][key]
        delta = 100 * row["accuracy_difference"]
        low, high = [100 * value for value in row["confidence_interval"]]
        ax.errorbar(index, delta, yerr=[[delta - low], [high - delta]], fmt="o", capsize=4)
    ax.axhline(0, color="#64748b", linewidth=1)
    ax.set(xticks=range(len(keys)), xticklabels=keys, ylabel="Accuracy difference vs C (percentage points)",
           title="Paired bootstrap 95% intervals; descriptive, unadjusted")
    save(fig, "differences_vs_natural")
    keys = [key for key in ("D256", "D512", "D1024", "D2048", "E1", "E2") if report["paired_vs_C"][key]["paired_resolved_denominator"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(range(len(keys)))
    ax.bar([x - 0.2 for x in xs], [report["paired_vs_C"][key]["transitions"]["wrong_to_right"] for key in keys], width=0.4, label="Rescued")
    ax.bar([x + 0.2 for x in xs], [report["paired_vs_C"][key]["transitions"]["right_to_wrong"] for key in keys], width=0.4, label="Spoiled")
    ax.set(xticks=xs, xticklabels=keys, ylabel="Questions", title="Answer transitions relative to natural stopping")
    ax.legend()
    save(fig, "rescued_vs_spoiled")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id")
    source.add_argument("--input", help="Reanalyze a saved complete planned JSONL without project manifests")
    parser.add_argument("--output", help="Output directory for standalone raw-record reanalysis")
    args = parser.parse_args()
    if args.run_id and (Path(args.run_id).name != args.run_id or args.run_id in (".", "..")):
        parser.error("Use a simple run ID")
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/test" / args.run_id if args.run_id else Path(args.output or "results/reanalysis")
    path = directory / "evaluated.jsonl" if args.run_id else Path(args.input)
    directory.mkdir(parents=True, exist_ok=True)
    freeze = (json.loads((root / "manifests/evaluation_freeze.json").read_text()) if args.run_id else
              {"analysis": {"bootstrap_seed": 42, "bootstrap_iterations": 10000, "confidence_level": 0.95}})
    rows = read_jsonl(path)
    if len(rows) != 4500:
        raise ValueError("Analysis requires all 4,500 planned slots, including explicit failures")
    report = analyze(rows, 500, freeze["analysis"])
    report.update(evaluated_sha256=file_sha256(path), evaluation_freeze_sha256=file_sha256(root / "manifests/evaluation_freeze.json") if args.run_id else None,
                  analysis_code_sha256=file_sha256(Path(__file__).resolve()), metrics_code_sha256=file_sha256(root / "src/s1small/metrics.py"))
    report["figures"] = plot_results(report, directory / "figures")
    (directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.run_id:
        (root / "docs/results_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Benchmark results", "", f"Study status: **{report['status']}**. All 500 planned IDs are retained per condition.", "",
             "| Condition | Correct | Resolved / planned | Accuracy | Mean reasoning | Mean total tokens | Mean seconds | Extraction failures |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in report["conditions"].items():
        accuracy = f"{100 * row['accuracy']:.2f}%" if row["accuracy"] is not None else "Incomplete"
        def number(value):
            return f"{value:.1f}" if value is not None else "N/A"
        lines.append(f"| {key} | {row['correct']} | {row['resolved_grades']} / 500 | {accuracy} | {number(row['mean_reasoning_tokens'])} | {number(row['mean_total_generated_tokens'])} | {number(row['mean_latency_seconds'])} | {row['extraction_failures']} |")
    lines += ["", "Paired differences use the exact common resolved question IDs; exclusions and transition counts are in `results_summary.json`. Intervals resample paired IDs with replacement, seed 42, 10,000 iterations, 95% percentile intervals. They are descriptive and unadjusted for multiple comparisons. A single training seed does not measure seed-to-seed uncertainty.", ""]
    (root / "docs/results.md" if args.run_id else directory / "results.md").write_text("\n".join(lines))
    print(json.dumps({"status": report["status"], "figures": len(report["figures"]), "conditions": len(report["conditions"])}))


if __name__ == "__main__":
    main()
