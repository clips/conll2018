"""Experiment 1 in the paper."""
import numpy as np
import time

from tqdm import tqdm
from old20.old20 import old20
from conll.data_new import load_data
from conll.helpers import load_featurizers_ortho, to_csv
from conll import calc_n


if __name__ == "__main__":

    np.random.seed(44)

    use_levenshtein = True

    for lang in ("nld", "eng-uk", "fra"):

        words = load_data(lang)

        freqs = [x['frequency'] for x in words]
        lengths = [len(x['orthography']) for x in words]
        rt_data = [x['rt'] for x in words]
        ortho_w = [x['orthography'] for x in words]

        estims = []

        featurizers, ids = zip(*load_featurizers_ortho(words))
        ids_f = list(ids)
        ids = []

        if use_levenshtein:
            start = time.time()
            dists = old20(ortho_w)
            e = time.time() - start
            ids.append(("old_20", "old_20"))
            estims.append(list(zip(dists, freqs, lengths, rt_data, ortho_w)))

        ids.extend(ids_f)

        batch_size = 1000

        for idx, f in tqdm(enumerate(featurizers), total=len(featurizers)):

            X = f.fit_transform(words).astype(np.float32)
            X = X / np.linalg.norm(X, axis=1)[:, None]

            s = np.zeros(X.shape[0])
            for x in tqdm(range(0, len(X), batch_size),
                          total=len(X) // batch_size):
                d = 1 - X[x:x+batch_size].dot(X.T)
                d = np.partition(d, kth=21, axis=1)[:, :21]
                d = np.sort(d, axis=1)[:, 1:21].mean(1)
                s[x:x+batch_size] = d

            estims.append(list(zip(s, freqs, lengths, rt_data, ortho_w)))

        ids.append(("coltheart_n", "coltheart_n"))
        n = calc_n(ortho_w)
        estims.append(list(zip(n, freqs, lengths, rt_data, ortho_w)))

        sample_results = np.array(estims)
        to_csv("data/experiment_new_{}_all_words.csv".format(lang),
               dict(zip(ids, sample_results)),
               ("score", "freq", "length", "rt", "ortho_form"))
