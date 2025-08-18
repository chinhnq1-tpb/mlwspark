from optbinning import BinningProcess
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from utils.utils import split_features_by_type

class WOEProcessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, params):
        self.binning_process = None
        self.params = params

    def fit(self, X, y):
        num_feat, cate_feat = split_features_by_type(X, X.columns.to_list())
        self.binning_process = BinningProcess(variable_names=X.columns.tolist(),
                                              categorical_variables=cate_feat,
                                              binning_fit_params = self.params, 
                                              n_jobs=-1)
        self.binning_process.fit(X, y)

        return self

    def transform(self, X):
        return pd.DataFrame(
            self.binning_process.transform(X),
            columns=X.columns,
            index=X.index
        )