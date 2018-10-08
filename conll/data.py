"""
Loads the data.

In all our experiments, we use a lexicon project for RT data, and a Subtitle
database or other equivalent database for frequencies and other data.

These functions are convenience functions.
"""
import unicodedata
import numpy as np

from wordkit.corpora import Subtlex
from wordkit.corpoa import Lexique
from .lexicon import read_blp_format, read_dlp2_format, read_flp_format


# This path needs to be modified
CORPUS_PREFIX = "data/"

# These paths need to be modified.
CORPORA = {"nld": (Subtlex,
                   "{}SUBTLEX-NL.cd-above2.txt".format(CORPUS_PREFIX),
                   read_dlp2_format,
                   "{}dlp2_items.tsv".format(CORPUS_PREFIX)),
           "eng-uk": (Subtlex,
                      "{}SUBTLEX-UK.xlsx".format(CORPUS_PREFIX),
                      read_blp_format,
                      "{}blp-items.txt".format(CORPUS_PREFIX)),
           "fra": (Lexique,
                   "{}Lexique382.txt".format(CORPUS_PREFIX),
                   read_flp_format,
                   "{}French Lexicon Project words.xls".format(CORPUS_PREFIX))}

FIELDS = ("orthography", "frequency", "log_frequency")


def normalize(string):
    """Normalize, remove accents and other stuff."""
    s = unicodedata.normalize("NFKD", string).encode('ASCII', 'ignore')
    return s.decode('utf-8')


def filter_function_ortho(x):
    """Filter words based on punctuation and length."""
    a = not set(x['orthography']).intersection({' ',
                                                "'",
                                                '.',
                                                '/',
                                                ',',
                                                '-'
                                                '&',
                                                '0',
                                                '1',
                                                '2',
                                                '3',
                                                '4',
                                                '5',
                                                '6',
                                                '7',
                                                '8',
                                                '9',
                                                '_',
                                                '&',
                                                '-',
                                                '@'})
    return a and len(x['orthography']) >= 2


def load_data(language, max_num=np.inf):
    """
    Load words and the rt data.

    Parameters
    ----------
    language : str
        The language for which to load the data.
    max_num : int or inf
        The number of items to load. If set to inf, this is ignored.

    Returns
    -------
    new_words : list of dicts
        The information for each word in the corpus.
    rt_data : dict
        The RT data for each word.

    """
    reader, path, lex_func, lex_path = CORPORA[language]

    rt_data = lex_func(lex_path)
    rt_data = {normalize(k): v for k, v in rt_data}
    r = reader(path,
               language=language,
               fields=FIELDS)

    words = r.transform(filter_function=filter_function_ortho)
    for x in words:
        x['orthography'] = normalize(x['orthography'])

    words = [x for x in words if len(x['orthography']) > 1]

    temp = set()
    words = sorted(words,
                   key=lambda x: (x['frequency'], x['orthography']),
                   reverse=True)
    new_words = []
    for x in words:
        if x['orthography'] in temp:
            continue
        temp.add(x['orthography'])
        new_words.append(x)
        if len(new_words) == max_num:
            break

    return new_words, rt_data
