"""Calculate the N statistic."""
import numpy as np
from collections import defaultdict


def calc_n(words):
    """Calculate the N statistic for a bunch of words."""
    binned = defaultdict(list)

    for x in words:
        binned[len(x)].append(list(x))

    binned = {k: np.array(v) for k, v in binned.items()}

    scores = {}

    for v in binned.values():
        z = np.sum(np.sum(v[:, None] != v[None, :], axis=-1) == 1, axis=1)
        w = ["".join(x) for x in v]
        for word, score in zip(w, z):
            scores[word] = score

    return np.array([scores[x] for x in words])
