"""Deterministic trial ordering and token-weighted development selection."""

import math
import random

from s1small.data import digest


def epoch_order(identifiers, seed, epoch):
    if len(set(identifiers)) != len(identifiers) or epoch < 0:
        raise ValueError("Duplicate IDs or invalid epoch")
    result = sorted(identifiers)
    random.Random(seed + epoch).shuffle(result)
    return result


def dropout_seed(seed, epoch, identifier):
    return int(digest(["dropout", seed, epoch, identifier])[:8], 16)


def select_trial(rows, expected_dev_ids):
    if not rows:
        raise ValueError("No completed development trials")
    for row in rows:
        if (row["status"] != "complete" or row["development_ids"] != expected_dev_ids
                or row["development_examples"] != len(expected_dev_ids)
                or not math.isfinite(row["development_loss"])):
            raise ValueError("Incomplete or mismatched development trial")
    # A deterministic exact tie favors the lower learning rate. No test score.
    return min(rows, key=lambda row: (row["development_loss"], row["learning_rate"], row["run_id"]))
