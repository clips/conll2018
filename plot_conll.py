"""Making the figures in the paper."""
import pandas as pd
import numpy as np

from itertools import combinations
from matplotlib import pyplot as plt

from scipy.stats import spearmanr


plt.rc('font', family='sans-serif')
plt.rc('font', serif='Helvetica')
plt.rc('text', usetex='false')


def plot(results_sig, keys, o, x_offset=.5, y_offset=.5, n_rows=3):
    """Plot results."""
    o_labs = ["{}. {}".format(idx+1, k) for idx, k in enumerate(o)]
    o_dict = {k: idx for idx, k in enumerate(o)}

    n_cols = len(keys) // n_rows

    f, sub = plt.subplots(n_rows,
                          n_cols,
                          sharey=False,
                          sharex=False,
                          figsize=(5, 12))

    if n_rows == 1:
        f.subplots_adjust(hspace=.1, wspace=.0)
    else:
        f.subplots_adjust(hspace=.1, wspace=-0.51)

    if np.ndim(sub) == 1:
        if n_rows != 1:
            sub = sub[:, None]
        else:
            sub = sub[None, :]

    plt.setp(sub[0, 0].get_yticklabels(), size=10)
    for idx, d in enumerate(keys):
        res_sig = results_sig[d]
        curr = sub[idx // n_cols, idx % n_cols]
        if idx % n_cols == 0:
            curr.set_yticks(np.arange(len(o)))
            curr.set_yticklabels(o_labs[::-1])
            curr.set_ylim(-.5, len(o)-.5)
            curr.set_title(keys[idx])
        else:
            curr.set_yticks([])
            curr.set_yticklabels([])
            curr.set_ylim(-.5, len(o)-.5)
            curr.set_title(keys[idx])

        if idx // n_cols == (n_rows - 1):
            curr.set_xticks(np.arange(len(o)))
            curr.set_xticklabels(np.arange(len(o))+1)
            plt.setp(curr.get_xticklabels(), ha="center", size=10)
        else:
            curr.set_xticks([])
            curr.set_xticklabels([])

        print([x.get_text() for x in curr.get_yticklabels()])

        plotto = np.zeros((len(o), len(o)))
        for (a, b), v in res_sig.items():
            x, y = o_dict[a], o_dict[b]
            if y < x:
                x, y = y, x

            x = (len(o) - x) - 1

            plotto[x, y] = res_sig[(a, b)]
            txt = ("%.2f" % res_sig[(a, b)]).replace("-0", "-").lstrip("0")
            curr.text(y+y_offset,
                      x+x_offset,
                      txt,
                      fontsize=8,
                      color='black',
                      ha="center")
        curr.imshow(plotto, cmap="PRGn", vmin=-1, vmax=1)


def experiment_1():
    """Make the plot for experiment 1."""
    experiment_1_files = ("data/experiment_nld_all_words.csv",
                          "data/experiment_eng-uk_all_words.csv",
                          "data/experiment_fra_all_words.csv")
    res = {}

    trans_dict = {"nld": "Dutch", "eng-uk": "English", "fra": "French"}

    trans_feats = {"coltheart_n": "N",
                   "old_20": "old 20",
                   "weighted bigrams": "rd 20 - bigrams",
                   "wickelfeatures": "rd 20 - wickel",
                   "fourteen": "rd 20 - fourteen",
                   "one hot": "rd 20 - one hot"}

    systems = ["weighted bigrams",
               "fourteen",
               "one hot",
               "wickelfeatures",
               "old_20",
               "coltheart_n"]

    res = {}

    for x in experiment_1_files:
        corrs = {}
        d = pd.read_csv(x)

        rows = {x if x not in trans_feats else trans_feats[x]:
                d[d["id"] == x] for x in systems}

        for k, v in rows.items():
            f = np.log10(v["freq"])
            corrs[(k, "freq")] = spearmanr(v["score"], f)[0]
            corrs[(k, "length")] = spearmanr(v["score"], v["length"])[0]
            corrs[(k, "RT")] = spearmanr(v["score"], v["rt"])[0]

        for (k1, v1), (k2, v2) in combinations(rows.items(), 2):
            corrs[(k1, k2)] = spearmanr(v1["score"], v2["score"])[0]

        v1f = np.log10(v1["freq"])
        corrs[("freq", "length")] = spearmanr(v1f,
                                              v1["length"])[0]
        corrs[("RT", "freq")] = spearmanr(v1f, v1["rt"])[0]
        corrs[("RT", "length")] = spearmanr(v1["rt"], v1["length"])[0]

        res[trans_dict[x.split("_")[-3]]] = corrs

    o = ["rd 20 - fourteen",
         "rd 20 - one hot",
         "rd 20 - bigrams",
         "rd 20 - wickel",
         "old 20",
         "N",
         "freq",
         "length",
         "RT"]

    plot(res, list(res.keys()), x_offset=0, y_offset=-.1, o=o)
    plt.savefig("plots/conll_figure_exp_1.pdf", bbox_inches="tight")
    plt.close()


def experiment_2():
    """Make the plot for experiment 2."""
    experiment_2_files = ("data/experiment_mlp_nld_all_words.csv",
                          "data/experiment_mlp_eng-uk_all_words.csv",
                          "data/experiment_mlp_fra_all_words.csv")
    res = {}

    trans_dict = {"nld": "Dutch", "eng-uk": "English", "fra": "French"}

    trans_feats = {"coltheart_n": "N",
                   "old_20": "old 20",
                   "weighted bigrams": "rd 20 - bigrams",
                   "wickelfeatures": "rd 20 - wickel",
                   "fourteen": "rd 20 - fourteen",
                   "one hot": "rd 20 - one hot"}

    systems = ["weighted bigrams",
               "fourteen",
               "one hot",
               "wickelfeatures"]

    res = {}

    for x in experiment_2_files:
        corrs = {}
        d = pd.read_csv(x)

        rows = {x if x not in trans_feats else trans_feats[x]:
                d[d["o_f"] == x] for x in systems}

        for k, v in rows.items():
            f = np.log10(v["freq"]+1)
            corrs[(k, "freq")] = spearmanr(v["score"], f)[0]
            corrs[(k, "length")] = spearmanr(v["score"], v["length"])[0]
            corrs[(k, "RT")] = spearmanr(v["score"], v["rt"])[0]

        for (k1, v1), (k2, v2) in combinations(rows.items(), 2):
            corrs[(k1, k2)] = spearmanr(v1["score"], v2["score"])[0]

        v1f = np.log10(v1["freq"]+1)
        corrs[("freq", "length")] = spearmanr(v1f,
                                              v1["length"])[0]
        corrs[("RT", "freq")] = spearmanr(v1f, v1["rt"])[0]
        corrs[("RT", "length")] = spearmanr(v1["rt"], v1["length"])[0]

        res[trans_dict[x.split("_")[-3]]] = corrs

    o = ["rd 20 - fourteen",
         "rd 20 - one hot",
         "rd 20 - bigrams",
         "rd 20 - wickel",
         "freq",
         "length",
         "RT"]
    plot(res, list(res.keys()), x_offset=0, y_offset=-.1, o=o)
    plt.savefig("plots/conll_figure_exp_2.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    experiment_1()
    experiment_2()
