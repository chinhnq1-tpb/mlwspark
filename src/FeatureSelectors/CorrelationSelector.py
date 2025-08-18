from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from utils.utils import split_features_by_type  # Assumes this function is already implemented

class TopCorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Select top X% of numerical features with the highest absolute correlation to the target.
    Retain all categorical features.

    Parameters:
    ----------
    top_percent : float
        Percentage of top numerical features to keep (e.g. 0.3 means top 30%).
    """
    def __init__(self, top_percent=0.3):
        self.top_percent = top_percent
        self.selected_numerical_features_ = []
        self.categorical_features_ = []

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()

        # Split features by type
        numerical_features, categorical_features = split_features_by_type(X, X.columns.tolist())

        # Compute absolute Pearson correlation on numerical features
        corrs = X[numerical_features].apply(lambda col: np.abs(np.corrcoef(col, y)[0, 1]))
        corrs = corrs.dropna()

        # Select top X% numerical features
        n_top = max(1, int(len(corrs) * self.top_percent))
        self.selected_numerical_features_ = corrs.sort_values(ascending=False).head(n_top).index.tolist()

        # Save categorical features to add later during transform
        self.categorical_features_ = categorical_features

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.selected_numerical_features_ + self.categorical_features_]
