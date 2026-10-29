"""Custom transformers used by the Lab 4 preprocessor (kept importable for joblib.load)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clips each column to percentiles learned on the training data."""

    def __init__(self, low=1, high=99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.low, axis=0)
        self.hi_ = np.nanpercentile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["capital_net"] = df["capital-gain"] - df["capital-loss"]
    df["hours_per_age"] = df["hours-per-week"] / (df["age"] - 16)
    return df
