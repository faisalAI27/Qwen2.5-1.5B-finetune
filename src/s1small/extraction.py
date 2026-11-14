"""Conservative development-only answer extraction; no equivalence grading."""

import re


def extract_answer(text):
    """Read the last balanced boxed expression, or an explicit final-answer line.

    Never guess a trailing number from unfinished reasoning. Nested LaTeX braces
    are retained. This parser is provisional until development validation/freeze.
    """
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    matches = list(re.finditer(r"(?im)^[ \t]*final answer[ \t]*:[ \t]*([^\n]*)", text))
    if matches and (not starts or matches[-1].start() > starts[-1].start()):
        match = matches[-1]
        answer = match.group(1).strip()
        if not answer or "\\boxed" in answer:
            return {"extracted_answer": None, "extraction_status": "incomplete_final_answer", "answer_start_char": None}
        return {"extracted_answer": answer, "extraction_status": "final_answer_line", "answer_start_char": match.start()}
    if starts:
        match = starts[-1]
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    answer = text[start:index].strip()
                    return {"extracted_answer": answer or None, "extraction_status": "boxed" if answer else "empty_answer",
                            "answer_start_char": match.start()}
        return {"extracted_answer": None, "extraction_status": "unclosed_box", "answer_start_char": None}
    return {"extracted_answer": None, "extraction_status": "no_final_answer", "answer_start_char": None}
