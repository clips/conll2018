"""
Loads the data.

In all our experiments, we use a lexicon project for RT data, and a Subtitle
database or other equivalent database for frequencies and other data.

These functions are convenience functions.
"""
import unicodedata
import numpy as np

from string import ascii_lowercase
from wordkit.corpora import Subtlex, Lexique
from .lexicon import read_blp_format, read_dlp2_format, read_flp_format


# This path needs to be modified
C_PREFIX = "data"

CORPORA = {"nld": (Subtlex,
                   "{}/SUBTLEX-NL.cd-above2.txt".format(C_PREFIX),
                   read_dlp2_format,
                   "{}/dlp2_items.tsv".format(C_PREFIX)),
           "eng-uk": (Subtlex,
                      "{}/SUBTLEX-UK.xlsx".format(C_PREFIX),
                      read_blp_format,
                      "{}/blp-items.txt".format(C_PREFIX)),
           "fra": (Lexique,
                   "{}/Lexique382.txt".format(C_PREFIX),
                   read_flp_format,
                   "{}/French Lexicon Project words.xls"
                   "".format(C_PREFIX))}

FIELDS = ("orthography", "frequency", "log_frequency")


def normalize(string):
    """Normalize, remove accents and other stuff."""
    s = unicodedata.normalize("NFKD", string).encode('ASCII', 'ignore')
    return s.decode('utf-8')


def filter_function_ortho(x):
    """Filter words based on punctuation and length."""
    a = not set(x['orthography']) - set(ascii_lowercase)
    return a and len(x['orthography']) >= 2 and x['frequency'] > 1


def load_data(language, max_num=np.inf):
    """Load the words and the RT data."""
    reader, path, lex_func, lex_path = CORPORA[language]

    rt_data = lex_func(lex_path)
    rt_data = {normalize(k): v for k, v in rt_data}
    r = reader(path,
               language=language,
               fields=FIELDS,
               merge_duplicates=True,
               scale_frequencies=False)

    words = r.transform()
    new_words = []
    seen = set()
    for x in words:
        x['orthography'] = normalize(x['orthography']).lower()
        if x['orthography'] in seen:
            continue
        seen.add(x['orthography'])
        new_words.append(x)
    words = list(filter(filter_function_ortho, new_words))
    ortho_forms = [x['orthography'] for x in words]
    set_ortho = set(ortho_forms)
    rt_data = {k: v for k, v in rt_data.items() if k in set_ortho}

    return words, rt_data, [x for x in words if x['orthography'] in rt_data]
