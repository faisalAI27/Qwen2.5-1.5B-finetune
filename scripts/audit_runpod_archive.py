"""Verify a downloaded vLLM result archive without touching the active Mac run."""

import argparse
import collections
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1small.data import digest, file_sha256
from s1small.experiment import PredictionStore
from s1small.grading import grader_manifest


def check(value, message):
    if not value:
        raise ValueError(message)


def same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, (float, int)):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def audit(root, archive_path, expected_evaluated=None):
    root, archive_path = Path(root).resolve(), Path(archive_path).resolve()
    archive_sha = file_sha256(archive_path)
    output = root / "artifacts/remote-results" / ("runpod-vllm-" + archive_sha[:12])
    output.mkdir(parents=True, exist_ok=True)
    test = "results/test/math500-runpod-vllm-v1"
    dev = "results/protocol_dev/runpod-vllm-dev-v1"
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist(); available = set(names)
        check(len(names) == len(available), "Duplicate archive entries")
        for member in z.infolist():
            path = PurePosixPath(member.filename)
            check(not path.is_absolute() and ".." not in path.parts, "Unsafe archive path")
            check(stat.S_IFMT(member.external_attr >> 16) != stat.S_IFLNK, "Unexpected archive symlink")
        check(z.testzip() is None, "Archive CRC check failed")
        print("Archive CRC and path checks passed", flush=True)
        def read(name):
            return json.loads(z.read(name))
        freeze_bytes = z.read("manifests/evaluation_freeze.json")
        freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()
        freeze = json.loads(freeze_bytes)
        manifest = read(test + "/manifest.json")
        check(freeze["scope"] == "frozen_test_evaluation" and not freeze["test_used_for_selection"], "Invalid test freeze")
        check(manifest["evaluation_freeze_sha256"] == freeze_sha, "Test manifest freeze hash differs")
        check(manifest["generation_runtime"] == freeze["generation_runtime"], "Runtime differs between test and freeze")
        check(grader_manifest() == freeze["grader"], "Grading configuration differs from pinned local grader")
        runtime = freeze["generation_runtime"]
        check(runtime["backend"] == "vllm_fp16_reconstructed_q4_lora", "Unexpected backend")
        launcher = read("vllm_launcher_manifest.json")
        for name, expected in launcher["files_sha256"].items():
            content = z.read(name)
            check(hashlib.sha256(content).hexdigest() == expected, "Launcher hash differs: " + name)
            trusted = subprocess.check_output(["git", "show", launcher["launcher_commit"] + ":" + name], cwd=root)
            check(content == trusted, "Launcher differs from its pinned Git source: " + name)
        check(runtime["code_sha256"] == launcher["files_sha256"], "Runtime code hashes differ from launcher")
        parent_bytes = z.read("manifests/evaluation_freeze_macos.json")
        check(hashlib.sha256(parent_bytes).hexdigest() == freeze["parent_macos_freeze_sha256"], "Parent freeze hash differs")
        check(parent_bytes == (root / "manifests/evaluation_freeze.json").read_bytes(), "Archive parent differs from the preserved Mac freeze")
        exported_bytes = z.read("artifacts/vllm-export/export_manifest.json")
        check(hashlib.sha256(exported_bytes).hexdigest() == runtime["export_manifest_sha256"], "Model export manifest differs")
        exported = json.loads(exported_bytes)
        for name, expected in exported["inputs_sha256"].items():
            check((root / name).is_file() and file_sha256(root / name) == expected, "Original export input missing/changed: " + name)
        selected = json.loads((root / "results/training/local-study-v1-trial-1/summary.json").read_text())
        check(exported["adapter"]["original_adapter_tensor_sha256"] == selected["adapter_tensor_sha256"], "Trained adapter identity differs")
        check(exported["adapter"]["exact_transpose_roundtrip"] and exported["optimizer_updates"] == 0, "Invalid adapter conversion evidence")
        # Model binaries are intentionally omitted from the results ZIP; their
        # checksums and all verified original inputs allow reconstruction.
        omitted = []
        for name, expected in freeze["files_sha256"].items():
            if name in available:
                check(hashlib.sha256(z.read(name)).hexdigest() == expected, "Archived frozen file differs: " + name)
            elif (root / name).is_file():
                check(file_sha256(root / name) == expected, "Local original frozen file differs: " + name)
            else:
                check(name.startswith("artifacts/vllm-export/"), "Missing unreconstructible frozen file: " + name)
                omitted.append(name)
        source_ids = [r["id"] for r in json.loads((root / "manifests/math500_questions.json").read_text())["rows"]]
        conditions = freeze["planned_conditions"]
        check(len(source_ids) == 500 and manifest["question_ids"] == source_ids and len(conditions) == 9, "Test IDs/conditions differ")
        check(read("manifests/math500_questions.json") == json.loads((root / "manifests/math500_questions.json").read_text()), "Question manifest differs")
        planned = {(key, condition) for key in source_ids for condition in conditions}
        slot_names = [name for name in names if name.startswith(test + "/slots/") and name.endswith(".json")]
        grade_names = [name for name in names if name.startswith(test + "/grades/") and name.endswith(".json")]
        check(len(slot_names) == len(grade_names) == 4500, "Expected 4500 slots and grade files")
        gold = {str(r["unique_id"]): r["answer"] for r in (json.loads(line) for line in (root / "data/raw/math500/test_with_answers.jsonl").read_text().splitlines())}
        seen, slim, extracted, timeouts = set(), [], 0, 0
        statuses, attempts = collections.Counter(), collections.Counter()
        evaluated_sha = hashlib.sha256()
        for line in z.open(test + "/evaluated.jsonl"):
            evaluated_sha.update(line)
            row = json.loads(line)
            pair = (row["question_id"], row["condition"])
            check(pair in planned and pair not in seen, "Duplicate or unplanned evaluated prediction")
            seen.add(pair)
            key = PredictionStore.key(*pair)
            raw = read(test + "/slots/" + key + ".json")
            grade = read(test + "/grades/" + key + ".json")
            check(raw["manifest_id"] == digest(manifest), "Prediction manifest ID differs")
            check(raw["status"] == "ok" and isinstance(grade["is_correct"], bool), "Incomplete generation or unresolved grade")
            check(grade["generation_sha256"] == digest(raw), "Grade is for a different generation")
            check(grade["grader_manifest_sha256"] == digest(freeze["grader"]), "Grade configuration hash differs")
            check(row == {**raw, **grade}, "Evaluated JSONL differs from slot/grade records")
            check(raw == read(test + f"/attempts/{key}-{raw['attempt']:02d}.json"), "Durable attempt differs from slot")
            check(row["gold_answer"] == gold[row["question_id"]], "Gold answer differs from the pinned local dataset")
            generated = sum(len(e["token_ids"]) for e in raw["segments"] if e["origin"] == "model")
            injected = sum(len(e["token_ids"]) for e in raw["segments"] if e["origin"] == "injected")
            check(generated == raw["total_generated_tokens"], "Emitted token accounting differs")
            check(injected == raw["injected_tokens"] and generated + injected == raw["total_new_tokens_consumed"], "Injected/total token accounting differs")
            check(generated + injected <= freeze["protocol"]["absolute_safety_ceiling"], "Absolute allowance exceeded")
            if row.get("extracted_answer") is None:
                check(not grade["is_correct"], "Missing readout incorrectly counted correct")
            statuses[raw["status"]] += 1; attempts[raw["attempt"]] += 1
            extracted += row.get("extracted_answer") is not None
            timeouts += bool(row.get("timed_out"))
            small = {k: v for k, v in row.items() if k not in ("segments", "final_context_token_ids", "answer_token_ids", "prompt", "visible_transcript")}
            small["segments"] = [{"origin": e["origin"], "phase": e["phase"], "text": e["text"][-1000:]} for e in row["segments"]]
            slim.append(small)
        check(seen == planned, "Incomplete question-condition coverage")
        evaluated_hash = evaluated_sha.hexdigest()
        if expected_evaluated:
            check(evaluated_hash == expected_evaluated, "Evaluated hash differs from reported RunPod grade output")
        summary = read(test + "/summary.json")
        check(summary["status"] == "complete" and summary["successful_model_outputs"] == 4500, "Generation summary is incomplete")
        check(summary["extracted_answers"] == extracted, "Extraction summary differs")
        check(summary["emitted_tokens_this_invocation"] == sum(r["total_generated_tokens"] for r in slim), "Generation token summary differs")
        check(summary["new_attempts_this_invocation"] == 4500, "Expected one full recorded generation invocation")
        check(read(dev + "/manifest.json")["generation_runtime"] == runtime, "Development runtime differs")
        dev_manifest = read(dev + "/manifest.json")
        check(hashlib.sha256(z.read(dev + "/manifest.json")).hexdigest() == freeze["development_manifest_sha256"], "Development manifest hash differs")
        development = [read(dev + "/slots/" + PredictionStore.key(q, c) + ".json") for q in freeze["development_question_ids"] for c in conditions]
        check(len(development) == 45 and all(r["status"] == "ok" and r.get("gold_answer") is None for r in development), "Incomplete/scored development records")
        check(digest(development) == freeze["development_outputs_sha256"], "Development outputs changed")
        smoke_files = [name for name in names if name.startswith("results/vllm_smoke/") and name.endswith(".json")]
        check(any((r := read(name))["status"] == "complete" and r["runtime"] == runtime and r["stop_token_probe"] and r["exported_files_unchanged"] for name in smoke_files), "Matching successful smoke evidence is missing")
        report = read(test + "/analysis.json")
        check(report["status"] == "complete" and report["evaluated_sha256"] == evaluated_hash, "Analysis input/coverage differs")
        check(report["evaluation_freeze_sha256"] == freeze_sha, "Analysis used a different freeze")
        check(report["analysis_code_sha256"] == file_sha256(root / "scripts/analyze_results.py"), "Analysis code differs")
        check(report["metrics_code_sha256"] == file_sha256(root / "src/s1small/metrics.py"), "Metrics code differs")
        spec = importlib.util.spec_from_file_location("trusted_runpod_review_analysis", root / "scripts/analyze_results.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        recomputed = module.analyze(slim, 500, freeze["analysis"])
        for field, value in recomputed.items():
            check(same(value, report[field]), "Locally recomputed analysis differs: " + field)
        print("All 4500 slots, grades, attempts, gold answers and recomputed statistics verified", flush=True)
        figures = [name for name in names if name.startswith(test + "/figures/") and name.endswith((".png", ".svg"))]
        check(len(figures) == 10, "Expected five plots in both PNG and SVG formats")
        from PIL import Image
        import xml.etree.ElementTree as xml
        for name in figures:
            content = z.read(name)
            if name.endswith(".png"):
                with Image.open(io.BytesIO(content)) as im:
                    im.load(); check(im.width > 0 and im.height > 0, "Empty PNG figure")
            else:
                check(xml.fromstring(content).tag.endswith("svg"), "Invalid SVG figure")
        # Retain small, directly reviewable artifacts separately. The original
        # verified ZIP remains the complete raw-data backup; avoid another GB
        # of duplicated prediction files on the user's Mac.
        retained = figures + [test + "/analysis.json", test + "/summary.json", test + "/manifest.json",
            dev + "/summary.json", "manifests/evaluation_freeze.json", "vllm_launcher_manifest.json",
            "artifacts/vllm-export/export_manifest.json", "results/installed-packages.txt"]
        for name in retained:
            path = output / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(z.read(name))
        record = {"status": "verified", "safe_to_stop_runpod_compute": True, "archive": str(archive_path),
            "archive_sha256": archive_sha, "archive_crc_verified": True, "archive_members": len(names),
            "evaluated_sha256": evaluated_hash, "questions": 500, "conditions": conditions,
            "verified_predictions": len(seen), "verified_resolved_grades": len(seen), "prediction_statuses": dict(statuses),
            "attempt_counts": dict(attempts), "extracted_answers": extracted, "timeouts": timeouts,
            "development_outputs_verified": 45, "frozen_source_and_adapter_verified": True,
            "analysis_recomputed_and_matched": True, "figure_files_verified": len(figures),
            "gpu": runtime["gpu"], "backend": runtime["backend"], "generation_summary": summary,
            "condition_accuracy": {k: v["accuracy"] for k, v in report["conditions"].items()},
            "reconstructible_model_files_not_in_results_zip": len(omitted),
            "remaining_project_work": "Research interpretation, final report and demo/handoff; these do not require repeating GPU inference.",
            "review_directory": str(output)}
        (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
        (output / "review.md").write_text(
            "# Verified RunPod result archive\n\nAll 500 questions × 9 conditions are present, with 4,500 generated outputs and resolved grades. "
            "Every slot matches its durable attempt and grade; gold answers match the pinned local dataset. "
            "The complete analysis, including paired bootstrap statistics, was recomputed locally and matched. "
            "Five PNG and five SVG figures decode successfully. Archive CRC and source/model/adapter provenance checks passed.\n\n"
            f"Archive: `{archive_path.name}`\n\nSHA-256: `{archive_sha}`\n\n"
            "It is safe to stop RunPod compute. Preserve the downloaded ZIP. Reconstructed FP16 model binaries are intentionally omitted; "
            "all original quantized inputs, the trained adapter and the exact conversion code are available locally. "
            "The Mac partial run may be retired under the user's recorded decision. Final research interpretation and reporting remain.\n")
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-evaluated-sha256")
    args = parser.parse_args()
    print(json.dumps(audit(Path(__file__).resolve().parents[1], args.archive, args.expected_evaluated_sha256), indent=2))
