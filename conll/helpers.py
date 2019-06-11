"""Helpers for the experiments."""
from itertools import product
from wordkit.orthography import LinearTransformer, \
                                WickelTransformer, \
                                WeightedOpenBigramTransformer, \
                                fourteen, \
                                OneHotCharacterExtractor


def to_csv(filename, data, score_names):
    """Write data to csv."""
    with open(filename, 'w') as f:
        add = ",".join(score_names)
        header_len = 3 + len(score_names)
        f.write("o,o_f,iter,{}\n".format(add))
        for k, v in data.items():
            for idx, val in enumerate(v):
                header = ",".join(["{}"] * header_len)
                f.write("{}\n".format(header).format(k[0],
                                                     k[1],
                                                     idx,
                                                     *val))


def load_featurizers_ortho(words):
    """Load the orthographic featurizers."""
    o_c = OneHotCharacterExtractor(field='orthography').extract(words)
    orthographic_features = {'fourteen': fourteen,
                             'one hot': o_c}
    possible_ortho = list(product([LinearTransformer],
                                  orthographic_features.items()))
    possible_ortho.append([WickelTransformer, ("wickelfeatures", 0)])
    possible_ortho.append([WeightedOpenBigramTransformer,
                          ("weighted bigrams", 0)])

    for o, (f_name, o_f) in sorted(possible_ortho,
                                   key=lambda x: x[1][0],
                                   reverse=False):
        if o == LinearTransformer:
            curr_o = o(o_f, field='orthography')
        elif o == WickelTransformer:
            curr_o = o(n=3, field='orthography')
        elif o == WeightedOpenBigramTransformer:
            curr_o = o(weights=(1, .7, .2), field='orthography')
        else:
            curr_o = o(field='orthography')

        yield curr_o, (o.__name__, f_name)
