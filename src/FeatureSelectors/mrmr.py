"""
This module provides the MRMRSelector class, a feature selection algorithm based on the
Maximum Relevance - Minimum Redundancy (mRMR) principle.
"""

from relevanceFunc import oneWayANOVA
from pyspark.ml import Estimator
from pyspark.sql import dataframe as DataFrame
from pyspark import keyword_only
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.param.shared import (
    HasInputCols,
    HasOutputCol,
    HasLabelCol,
    HasNumFeatures,
)
from pyspark.ml.util import (
    DefaultParamsReadable,
    DefaultParamsWritable,
    MLWritable,
    MLReadable,
)


class MRMRSelector(
    Estimator,
    HasInputCols,
    HasOutputCol,
    HasLabelCol,
    HasNumFeatures,
    DefaultParamsReadable,
    DefaultParamsWritable,
    MLWritable,
    MLReadable,
):
    """
    A PySpark Estimator for feature selection using the mRMR algorithm.
    """

    @keyword_only
    def __init__(
        self,
        inputCols: list,
        numFeatures: int,
        labelCol: str,
        outputCol: str = "features",
    ) -> None:
        super().__init__()
        self._setDefault(
            inputCols=[], numFeatures=10, outputCol="features", labelCol="target"
        )
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(
        self,
        inputCols: list,
        numFeatures: int,
        labelCol: str,
        outputCol: str = "features",
    ) -> None:
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def get_selected(self, df: DataFrame, numFeatures, candidates, label) -> list:
        """
        Selects the best features based on the mRMR criteria.

        Args:
            df (DataFrame): The input DataFrame.
            numFeatures (int): The number of features to select.
            candidates (list): A list of candidate feature columns.
            label (str): The name of the label column.

        Returns:
            list: A list of selected feature names.
        """
        selectedFeatures = []
        # Calculate relevance
        relevanceMap = {}
        cumRedundancy = {}
        collectRelevance = oneWayANOVA(df, candidates, label).collect()[0]
        maxRelevance = float("-inf")
        for feature in candidates:
            relevanceMap[feature] = collectRelevance[feature]
            cumRedundancy[feature] = 0
            if maxRelevance < collectRelevance[feature]:
                maxRelevance = collectRelevance[feature]
                maxRelevanceFeat = feature
        selectedFeatures.append(maxRelevanceFeat)
        candidates.remove(maxRelevanceFeat)
        lastSelected = maxRelevanceFeat

        while len(selectedFeatures) < numFeatures:
            tmpMax = float("-inf")
            for feature in candidates:
                relevance = relevanceMap[feature]
                redundance = df.corr(feature, lastSelected)
                cumRedundancy[feature] += redundance
                mrmrValue = relevance / (redundance / len(selectedFeatures))
                if tmpMax < mrmrValue:
                    tmpMax = mrmrValue
                    lastSelected = feature
            selectedFeatures.append(lastSelected)
            candidates.remove(lastSelected)

        return selectedFeatures

    def _fit(self, df: DataFrame) -> VectorAssembler:
        """
        Fits the model to the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame.

        Returns:
            VectorAssembler: A VectorAssembler fitted to the selected features.
        """
        inputCols = self.getInputCols()
        outputCol = self.getOutputCol()
        labelCol = self.getLabelCol()
        numFeatures = self.getNumFeatures()

        selected_features = self.get_selected(df, numFeatures, inputCols, labelCol)

        return VectorAssembler(inputCols=selected_features, outputCol=outputCol)
