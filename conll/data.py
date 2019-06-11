"""Loads the data."""
import os
from wordkit.corpora import LexiconProject, Subtlex, merge, Lexique
from string import ascii_lowercase


C_PREFIX = "../../corpora"
FIELDS = ("orthography", "frequency")

LEXICON = {"eng-uk": "lexicon_projects/blp-items.txt",
           "eng-us": "lexicon_projects/elp-items.csv",
           "fra": "lexicon_projects/French Lexicon Project words.xls",
           "nld": "lexicon_projects/dlp-items.txt"}
FREQ_PATHS = {"eng-uk": "subtlex/SUBTLEX-UK.xlsx",
              "eng-us": "subtlex/SUBTLEXusfrequencyabove1.xls",
              "nld": "subtlex/SUBTLEX-NL.cd-above2.txt",
              "fra": "lexique/Lexique382.txt"}


def filter_function(x):
    """Filter words based on punctuation and length."""
    a = not set(x['orthography']) - set(ascii_lowercase)
    c = x.get('lexicality', 'W') == 'W'
    return a and (3 <= len(x['orthography']) <= 13) and c


def load_data(language):
    """Load the words and the RT data."""
    path = os.path.join(C_PREFIX, LEXICON[language])
    if language in {"eng_uk", "nld"}:
        fields = ("orthography", "rt", "lexicality")
    else:
        fields = ("orthography", "rt")
    lex = LexiconProject(path,
                         language=language,
                         fields=fields)
    lex_words = lex.transform(filter_function=filter_function)
    lex_path = os.path.join(C_PREFIX, FREQ_PATHS[language])

    if language == "fra":
        freqs = Lexique(lex_path,
                        fields=("frequency",
                                "orthography"),
                        scale_frequencies=True,
                        duplicates="sum")
        freq_words = freqs.transform(filter_function=filter_function)
        words = merge(lex_words,
                      freq_words,
                      merge_fields="orthography",
                      transfer_fields="rt",
                      discard=True)
    else:
        freqs = Subtlex(lex_path,
                        language=language,
                        fields=("frequency", "orthography"),
                        scale_frequencies=True)
        freq_words = freqs.transform(filter_function=filter_function)
        words = merge(lex_words,
                      freq_words,
                      merge_fields="orthography",
                      transfer_fields="rt",
                      discard=False)

    o = words.get('orthography')
    assert len(o) == len(set(o))
    return words
