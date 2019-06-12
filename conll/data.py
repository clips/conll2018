"""Loads the data."""
import numpy as np
from wordkit.corpora import LexiconProject, Subtlex, merge, Lexique
from string import ascii_lowercase


C_PREFIX = "../../corpora"


LEXICONS = {"nld": "{}/lexicon_projects/dlp-items.txt".format(C_PREFIX),
            "eng-uk": "{}/lexicon_projects/blp-items.txt".format(C_PREFIX),
            "fra": "{}/lexicon_projects/French Lexicon Project words.xls"
                   "".format(C_PREFIX),
            "eng-us": "{}/lexicon_projects/elp-items.csv".format(C_PREFIX)}
SUBTITLES = {"nld": "{}/subtlex/SUBTLEX-NL.cd-above2.txt".format(C_PREFIX),
             "eng-uk": "{}/subtlex/SUBTLEX-UK.xlsx".format(C_PREFIX),
             "eng-us": "{}/subtlex/SUBTLEXusfrequencyabove1.xls"
                       "".format(C_PREFIX),
             "fra": "{}/lexique/Lexique382.txt"
                    "".format(C_PREFIX)}

FIELDS = ("orthography", "frequency", "log_frequency")


def filter_function_ortho(x):
    """Filter words based on punctuation and length."""
    a = not set(x['orthography']) - set(ascii_lowercase)
    b = x['orthography'] == x['orthography'].lower()
    c = x.get('lexicality', 'W') == 'W'
    return a and b and 2 <= len(x['orthography']) and c


def load_data(language):
    """Load the words and the RT data."""
    path = LEXICONS[language]

    if language in {"nld", "eng-uk"}:
        fields = ("orthography", "rt", "lexicality")
    else:
        fields = ("orthography", "rt")
    lex = LexiconProject(path,
                         language=language,
                         fields=fields)
    lex_words = lex.transform(filter_function=filter_function_ortho)
    lex_words = lex_words.filter(filter_nan=("rt",))

    freqpath = SUBTITLES[language]

    if language == "fra":
        freqs = Lexique(freqpath,
                        language=language,
                        fields=("frequency", "orthography"))

    else:
        freqs = Subtlex(freqpath,
                        language=language,
                        fields=("frequency", "orthography"))
    freq_words = freqs.transform(filter_function=filter_function_ortho)
    words = merge(freq_words,
                  lex_words,
                  merge_fields=("orthography",),
                  transfer_fields=("frequency",))
    words = words.filter(filter_nan=("frequency",))

    return words
