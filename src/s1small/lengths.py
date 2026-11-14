"""Deterministic sequence statistics without plotting dependencies."""


def percentile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Nonempty values and a fraction in [0, 1] required")
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize_lengths(rows, limits):
    if not rows:
        raise ValueError("No lengths to summarize")
    result = {"rows": len(rows), "percentile_method": "linear interpolation at (n-1)*p", "components": {}}
    for component in ("total", "prompt", "reasoning", "answer", "structure"):
        values = [row[component] for row in rows]
        result["components"][component] = {
            "min": min(values), "median": percentile(values, 0.5),
            "p90": percentile(values, 0.9), "p95": percentile(values, 0.95), "max": max(values),
        }
    result["context_limits"] = [{"limit": limit,
                                  "fits_complete_examples": sum(row["total"] <= limit for row in rows),
                                  "exceeds": sum(row["total"] > limit for row in rows)} for limit in sorted(limits)]
    edges = [0] + sorted(set(limits))
    result["histogram"] = [{"lower_exclusive": lower, "upper_inclusive": upper,
                            "count": sum(lower < row["total"] <= upper for row in rows)} for lower, upper in zip(edges, edges[1:])]
    result["histogram"].append({"lower_exclusive": edges[-1], "upper_inclusive": None,
                                "count": sum(row["total"] > edges[-1] for row in rows)})
    result["boundary_crossing_tokens"] = sum(row["boundary_crossing_tokens"] for row in rows)
    return result
