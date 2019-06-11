"""Experiment 1 in the paper."""
import numpy as np
import time
import pandas as pd

from tqdm import tqdm
from old20.old20 import old20
from conll.data import load_data
from conll.helpers import load_featurizers_ortho
from conll import calc_n


if __name__ == "__main__":

    np.random.seed(44)

    use_levenshtein = True

    for lang in ("nld", "eng-uk", "fra"):

        words = load_data(lang)
        subset_words = [x for x in words if 'rt' in x]

        ortho_forms = [x['orthography'] for x in words]
        freqs = [x['frequency'] for x in subset_words]
        lengths = [len(x['orthography']) for x in subset_words]
        rt_data = [x['rt'] for x in subset_words]
        ortho_w = [x['orthography'] for x in subset_words]

        header = ["score", "freq", "length", "rt", "ortho", "id"]
        estims = []

        featurizers, ids = zip(*load_featurizers_ortho(words))
        batch_size = 1000

        for idx, f in tqdm(enumerate(featurizers), total=len(featurizers)):

            X = f.fit_transform(words).astype(np.float32)
            X = X / np.linalg.norm(X, axis=1)[:, None]

            X_base = f.transform(subset_words).astype(np.float32)
            X_base = X_base / np.linalg.norm(X_base, axis=1)[:, None]

            s = np.zeros(X_base.shape[0])
            for x in tqdm(range(0, len(X_base), batch_size),
                          total=len(X_base) // batch_size):
                d = 1 - X_base[x:x+batch_size].dot(X.T)
                d = np.partition(d, kth=21, axis=1)[:, :21]
                d = np.sort(d, axis=1)[:, 1:21].mean(1)
                s[x:x+batch_size] = d

            estims.extend(zip(s,
                              freqs,
                              lengths,
                              rt_data,
                              ortho_w,
                              ["+".join(ids[idx])] * X.shape[0]))

        n = calc_n(ortho_w)
        estims.extend(list(zip(n,
                               freqs,
                               lengths,
                               rt_data,
                               ortho_w,
                               ["n"] * X.shape[0])))
        if use_levenshtein:
            start = time.time()
            dists = old20(ortho_w, ortho_forms)
            e = time.time() - start
            estims.extend(list(zip(dists,
                                   freqs,
                                   lengths,
                                   rt_data,
                                   ortho_w,
                                   ["old20"] * len(ortho_w))))

        sample_results = np.array(estims)
        df = pd.DataFrame(sample_results, columns=header)
        df.to_csv("data/experiment_{}_all_words.csv".format(lang))
