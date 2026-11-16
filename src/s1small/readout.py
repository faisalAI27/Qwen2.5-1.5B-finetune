"""Keep generated answer text distinct from a declared shared readout label."""

from s1small.grading import extract_final_answer


def extract_readout(generated_text, prefix=""):
    text = prefix + generated_text
    return {**extract_final_answer(text), "extraction_input": text,
            "extraction_origin": ("declared_common_label_plus_generated_answer" if prefix else "model_generated_text"),
            "injected_readout_prefix": prefix}
