"""Audit native readout compatibility without scoring benchmark answers."""

from s1small.budget_forcing import CONDITIONS


def audit_development(records):
    by_condition = {}
    for condition in CONDITIONS:
        planned = [row for row in records if row["condition"] == condition]
        outputs = [row for row in planned if row["status"] == "ok"]
        by_condition[condition] = {
            "planned": len(planned), "model_outputs": len(outputs),
            "extracted_answers": sum(bool(row.get("extracted_answer")) for row in outputs),
            "empty_answer_texts": sum(not row.get("answer_text", "").strip() for row in outputs),
            "unexpected_answer_role_markers": sum(row.get("termination_reason") == "unexpected_role_marker" for row in outputs),
            "reasoning_marker_stops": sum(event.get("removed_stop_token_id") == 151644
                for row in outputs for event in row.get("segments", [])),
            "actual_wait_outputs": sum(row.get("forced_wait_count", 0) > 0 for row in outputs),
            "reasoning_cap_hits": sum(bool(row.get("reasoning_cap_hit")) for row in outputs),
            "answer_cap_hits": sum(bool(row.get("answer_cap_hit")) for row in outputs)}
    complete = bool(records) and all(row["status"] == "ok" for row in records)
    structured = [row for row in records if row["condition"] not in ("A", "B") and row["status"] == "ok"]
    extracted = sum(bool(row.get("extracted_answer")) for row in structured)
    status = ("in_progress" if not complete else
              "readout_not_established" if not extracted else "readout_observed")
    return {"scope": "development_protocol_compatibility_not_accuracy", "status": status,
            "planned_outputs": len(records), "completed_outputs": sum(row["status"] == "ok" for row in records),
            "structured_outputs": len(structured), "structured_extracted_answers": extracted,
            "conditions": by_condition,
            "interpretation": "Execution success is separate from learned readout compatibility. No correctness scores are computed. Zero extracted structured answers requires investigation before the test freeze."}


def require_readout(audit):
    if audit["status"] != "readout_observed":
        raise ValueError("Native development readout is not established: no extractable structured-condition answers or incomplete execution. Investigate before freezing/opening MATH-500; retain all failed protocol outputs.")
