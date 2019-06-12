"""Analyze the data using statsmodels."""
import pandas as pd
import statsmodels.api as sm
import numpy as np
from glob import iglob
from sklearn.preprocessing import StandardScaler


def linear_model(xs, y):
    """Create a linear model."""
    s = StandardScaler()
    xs = np.stack(xs, 1)
    xs = s.fit_transform(xs)
    xs = sm.add_constant(xs)
    model = sm.OLS(y, xs).fit()
    return model


def process_df(df):
    """Process a single df, and return the scores."""
    results = {}
    for x in np.unique(df['id']):
        sub_df = df[df['id'] == x]
        # Assume frequency is logged already
        freq = sub_df['freq']
        length = sub_df['length']
        rt = sub_df['rt']
        score = sub_df['score']
        results[x] = linear_model([freq, length, score], rt)
    results['base'] = linear_model([freq, length], rt)
    return results


if __name__ == "__main__":

    results = {}

    for x in iglob("data/experiment1_*.csv"):

        df = pd.read_csv(x)
        results[x] = process_df(df)

    results_mlp = {}

    for x in iglob("data/experiment2_*.csv"):

        df = pd.read_csv(x)
        results_mlp[x] = process_df(df)
