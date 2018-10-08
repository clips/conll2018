# conll2018

The code for the conll2018 submission: [from strings to other things: linking the neighborhood and transposition effects in word reading.]()

In the paper, we explore whether the neighborhoods of words featurized using feature sets that allow for transposition, e.g. bigram features or character trigrams, explain more variance in RT measurements than conventional feature sets.

If this is the case, then there is a chance that the neighborhood effect is _early_, in the sense that it takes place _during_ word recognition, and not _after_ the word has been recognized.
If, on the other hand, feature sets that do not allow for transposition explain more variance, then it is likely that the neighborhood effect is _late_.

Across all our experiments, we find that the non-transpositional feature sets explain more variance in RT measurements.
From this, we conclude that the neighborhood is formed without taking into account transpositions.
Notably, this flies in the face of conventional psycholinguistic research on the neighborhood effect.

# Requirements

The old20 package needs to be installed manually, see [here](https://github.com/stephantul/old20).
All other requirements are in `requirements.txt`.

# Usage

Run `experiment_1.py` or `experiment_2.py` to get the raw data files as CSV from the corpora.
You can then navigate to the `r` folder and run the `R` experiments to obtain the results from the paper.

Both experiments require that the following corpora are present in `data`:

```
SUBTLEX-NL.cd-above2.txt
dlp2_items.tsv
SUBTLEX-UK.xlsx
blp-items.txt
Lexique382.txt
French Lexicon Project words.xls
```

The Subtlex files can be found [here](http://crr.ugent.be/programs-data/subtitle-frequencies).
Lexique can be found [here](http://www.lexique.org/public/Lexique382.zip) (links to direct download)
The lexicon projects can be found [here](http://crr.ugent.be/programs-data/lexicon-projects)

# Data request

We can supply the neighborhood measurements for all the corpora and feature sets on request. Please send an E-mail to the lead author.

# Citation

If you use this code, or the results from the paper, please cite us, as follows:

```
```

# License

GPL 3.0
