"""Experiment 2 in the paper."""
import numpy as np
import torch

from tqdm import tqdm
from conll.mlp import Perceptron, train
from conll.helpers import load_featurizers_ortho, to_csv
from conll.data import load_data
from sklearn.metrics import pairwise_distances


if __name__ == "__main__":

    np.random.seed(44)
    print("started")

    for lang in ("nld", "fra", "eng-uk"):

        words = load_data(lang)
        subset_words = [x for x in words if 'rt' in x]

        ortho_forms = [x['orthography'] for x in words]
        freqs = [x['log_frequency'] for x in subset_words]
        lengths = [len(x['orthography']) for x in subset_words]
        rt_data = [x['rt'] for x in subset_words]
        ortho_w = [x['orthography'] for x in subset_words]

        estims = []

        req = [('LinearTransformer', 'fourteen'),
               ('LinearTransformer', 'one hot'),
               ('WeightedOpenBigramTransformer', 'weighted bigrams'),
               ('WickelTransformer', 'wickelfeatures')]

        f = load_featurizers_ortho(words)
        featurizers, ids = zip(*[(x, y) for x, y in f if y in req])
        ids = list(ids)

        estims = []

        for idx, f in tqdm(enumerate(featurizers), total=len(featurizers)):

            X = f.fit_transform(words).astype(np.float32)
            y = np.arange(X.shape[0])
            p = Perceptron(X.shape[1], 500, X.shape[0])
            X = torch.from_numpy(X.astype(np.float32))
            train(p, 1000, X, batch_size=250)
            p.cpu()

            x_ = []
            for x in range(0, len(X), 250):
                data = torch.from_numpy(X[x:x+250])
                _, pred = p(data)
                x_.extend(torch.max(pred, 1)[1])

            corr = len([x for x in y == x_ if x])
            print("Accuracy: {} {} {}".format(lang,
                                              ids[idx],
                                              corr / X.shape[0]))

            X = f.transform(subset_words).astype(np.float32)

            hid = []

            for x in range(0, len(X), 250):
                data = torch.from_numpy(X[x:x+250])
                hid.extend(p.hidden(data).detach().numpy())

            # Estimate density
            dist = pairwise_distances(hid, metric="cosine")
            s = np.partition(dist, axis=1, kth=21)[:, :21]
            s = np.sort(s, 1)[:, 1:21].mean(1)

            estims.append(list(zip(s, freqs, lengths, rt_data, ortho_w)))

        sample_results = np.array(estims)
        to_csv("data/experiment_mlp_{}_all_words.csv".format(lang),
               dict(zip(ids, sample_results)),
               ("score", "freq", "length", "rt", "ortho_form"))
