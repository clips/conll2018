"""Functions for reading the various lexicon projects."""
import pandas as pd


def read_blp_format(filename, words=set()):
    """Read RT data from the British Lexicon Project files."""
    words = set(words)
    f = open(filename)
    _ = next(f)
    temp = set()
    for line in f:

        word, _, rt, *rest = line.strip().split("\t")

        if words and word not in words:
            continue
        if word in temp:
            continue
        try:
            temp.add(word)
            yield((word, float(rt)))
        except ValueError:
            continue


def read_dlp2_format(filename, words=set()):
    """Read RT data from the Dutch Lexicon Project files."""
    words = set(words)
    f = open(filename)
    _ = next(f)
    temp = set()
    for line in f:

        _, _, word, _, _, _, rt, *rest = line.strip().split("\t")

        if words and word not in words:
            continue
        if word in temp:
            continue
        try:
            temp.add(word)
            yield((word, float(rt)))
        except ValueError:
            continue


def read_dlp1_format(filename, words=set()):
    """Read RT data from the Dutch Lexicon Project files."""
    words = set(words)
    f = open(filename)
    _ = next(f)
    temp = set()

    for line in f:

        word, _, rt, *rest = line.strip().split("\t")

        if words and word not in words:
            continue
        if word in temp:
            continue
        try:
            temp.add(word)
            yield((word, float(rt)))
        except ValueError:
            continue


def read_flp_format(filename, words=set()):
    """Read RT data from Lexique."""
    words = set(words)
    temp = set()
    for idx, line in pd.read_excel(filename).iterrows():
        if not words or line['item'] in words:
            if line['item'] in temp:
                continue
            temp.add(line['item'])
            yield((line['item'], line['rt']))
