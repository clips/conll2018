"""Making the figures in the paper."""
import pandas as pd
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

from itertools import combinations
from matplotlib import pyplot as plt

from scipy.stats import spearmanr


plt.rc('font', family='sans-serif')
plt.rc('font', serif='Helvetica')
plt.rc('text', usetex='false')

hexcols = ['#fff7f3',
           '#fde0dd',
           '#fcc5c0',
           '#fa9fb5',
           '#f768a1',
           '#dd3497',
           '#ae017e',
           '#7a0177',
           '#49006a']

rgbcols = []
for x in hexcols:
    x = x[1:]
    rgbcol = tuple(int(x[i:i+2], 16) / 255 for i in (0, 2, 4))
    rgbcols.append(rgbcol)

cmap = LinearSegmentedColormap.from_list("", rgbcols)


def plot(results_sig, keys, o, x_offset=.5, y_offset=.5, n_rows=1):
    """Plot results."""
    o_labs = ["{}. {}".format(idx+1, k) for idx, k in enumerate(o)]
    o_dict = {k: idx for idx, k in enumerate(o)}

    n_cols = len(keys) // n_rows

    f, sub = plt.subplots(n_rows,
                          n_cols,
                          sharey=False,
                          sharex=False,
                          figsize=(12, 5))

    if n_rows == 1:
        f.subplots_adjust(hspace=-.51, wspace=.0)
    else:
        f.subplots_adjust(hspace=-.51, wspace=.1)

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
            curr.set_yticklabels(o_labs)
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
            plotto[x, y] = res_sig[(a, b)]
            plotto[y, x] = res_sig[(a, b)]
            txt = ("%.2f" % res_sig[(a, b)]).replace("-0", "-").lstrip("0")
            curr.text(y+y_offset,
                      x+x_offset,
                      txt,
                      fontsize=8,
                      color='black',
                      ha="center")
            curr.text(x+x_offset,
                      y+y_offset,
                      txt,
                      fontsize=8,
                      color='black',
                      ha="center")
        curr.imshow(plotto.T, cmap=cmap, vmin=-1, vmax=1)


def experiment_1():
    """Make the plot for experiment 1."""
    experiment_1_files = ("data/experiment1_nld.csv",
                          "data/experiment1_eng-uk.csv",
                          "data/experiment1_fra.csv")
    res = {}

    trans_dict = {"nld": "Dutch", "eng-uk": "English", "fra": "French"}

    trans_feats = {"n": "N",
                   "old20": "old 20",
                   "weighted bigrams": "rd 20 - bigrams",
                   "wickelfeatures": "rd 20 - wickel",
                   "fourteen": "rd 20 - fourteen",
                   "one hot": "rd 20 - one hot"}

    res = {}

    for x in sorted(experiment_1_files):
        corrs = {}
        d = pd.read_csv(x)

        ids = np.unique(d['id'])
        rows = {trans_feats[x.split("+")[-1]]: d[d["id"] == x] for x in ids}

        for k, v in rows.items():
            corrs[(k, "freq")] = spearmanr(v["score"], v["freq"])[0]
            corrs[(k, "length")] = spearmanr(v["score"], v["length"])[0]
            corrs[(k, "RT")] = spearmanr(v["score"], v["rt"])[0]

        for (k1, v1), (k2, v2) in combinations(rows.items(), 2):
            corrs[(k1, k2)] = spearmanr(v1["score"], v2["score"])[0]

        corrs[("freq", "length")] = spearmanr(v1["freq"],
                                              v1["length"])[0]
        corrs[("RT", "freq")] = spearmanr(v1["freq"], v1["rt"])[0]
        corrs[("RT", "length")] = spearmanr(v1["rt"], v1["length"])[0]

        res[trans_dict[x.split(".")[0].split("_")[1]]] = corrs

    o = ["rd 20 - fourteen",
         "rd 20 - one hot",
         "rd 20 - bigrams",
         "rd 20 - wickel",
         "old 20",
         "N",
         "freq",
         "length",
         "RT"]

    plot(res, sorted(res.keys()), x_offset=0, y_offset=-.1, o=o)
    plt.savefig("plots/conll_figure_exp_1.pdf", bbox_inches="tight")
    plt.close()


def experiment_2():
    """Make the plot for experiment 2."""
    experiment_2_files = ("data/experiment2_nld.csv",
                          "data/experiment2_eng-uk.csv",
                          "data/experiment2_fra.csv")
    res = {}

    trans_dict = {"nld": "Dutch", "eng-uk": "English", "fra": "French"}

    trans_feats = {"coltheart_n": "N",
                   "old_20": "old 20",
                   "weighted bigrams": "rd 20 - bigrams",
                   "wickelfeatures": "rd 20 - wickel",
                   "fourteen": "rd 20 - fourteen",
                   "one hot": "rd 20 - one hot"}

    res = {}

    for x in experiment_2_files:
        corrs = {}
        d = pd.read_csv(x)

        ids = np.unique(d['id'])
        rows = {trans_feats[x.split("+")[-1]]: d[d["id"] == x] for x in ids}

        for k, v in rows.items():
            corrs[(k, "freq")] = spearmanr(v["score"], v["freq"])[0]
            corrs[(k, "length")] = spearmanr(v["score"], v["length"])[0]
            corrs[(k, "RT")] = spearmanr(v["score"], v["rt"])[0]

        for (k1, v1), (k2, v2) in combinations(rows.items(), 2):
            corrs[(k1, k2)] = spearmanr(v1["score"], v2["score"])[0]

        corrs[("freq", "length")] = spearmanr(v1["freq"],
                                              v1["length"])[0]
        corrs[("RT", "freq")] = spearmanr(v1["freq"], v1["rt"])[0]
        corrs[("RT", "length")] = spearmanr(v1["rt"], v1["length"])[0]

        res[trans_dict[x.split(".")[0].split("_")[1]]] = corrs

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
