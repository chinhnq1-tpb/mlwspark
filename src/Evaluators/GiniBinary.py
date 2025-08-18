from pyspark.ml.evaluation import BinaryClassificationEvaluator


class GiniEvaluator(BinaryClassificationEvaluator):
    """
    Custom evaluator to compute the Gini coefficient for binary classification models.
    Gini = 2 * AUC - 1
    """

    def __init__(self,
                 rawPredictionCol: str = "rawPrediction",
                 labelCol: str = "label",
                 metricName: str = "gini"):  # metricName is informational
        # Initialize parent with AUC metric
        super(GiniEvaluator, self).__init__(
            rawPredictionCol=rawPredictionCol,
            labelCol=labelCol,
            metricName="areaUnderROC"
        )
        # Store custom metric name
        self._metricName = metricName

    def evaluate(self, dataset, params=None) -> float:
        """
        Compute Gini coefficient = 2 * AUC - 1

        :param dataset: Spark DataFrame with columns rawPrediction and label
        :param params: Optional parameters (unused)
        :return: Gini coefficient as a float
        """
        # Compute AUC using the inherited evaluator
        auc = super(GiniEvaluator, self).evaluate(dataset, params)
        # Transform AUC to Gini
        gini = 2.0 * auc - 1.0
        return float(gini)

    def isLargerBetter(self) -> bool:
        """
        Indicate that larger Gini values are better.
        """
        return True
