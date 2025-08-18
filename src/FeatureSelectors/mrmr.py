from pyspark.ml import Estimator
from pyspark.sql import dataframe as DataFrame
import mrmr
from pyspark import keyword_only
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.param.shared import HasInputCols, HasOutputCol, HasLabelCol, HasNumFeatures
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable, MLWritable, MLReadable

class MRMRSelector(Estimator, HasInputCols, HasOutputCol, HasLabelCol, HasNumFeatures, DefaultParamsReadable, DefaultParamsWritable, MLWritable, MLReadable):
    
    @keyword_only
    def __init__(self, 
                 inputCols: list,
                 numFeatures: int,
                 labelCol: str,
                 outputCol: str="features") -> None:
        super().__init__()
        self._setDefault(inputCols=[], numFeatures=10, outputCol="features", labelCol="target")
        kwargs = self._input_kwargs
        self.set_params(**kwargs)
        
    @keyword_only
    def set_params(self, 
                 inputCols: list,
                 numFeatures: int,
                 labelCol: str,
                 outputCol: str="features") -> None:
        kwargs = self._input_kwargs
        self._set(**kwargs)

    
    def _fit(self, df: DataFrame) -> VectorAssembler:
        inputCols = self.getInputCols()
        outputCol = self.getOutputCol()
        labelCol = self.getLabelCol()
        numFeatures = self.getNumFeatures()
        
        used_features = inputCols + [labelCol]

        selected_features = mrmr.spark.mrmr_classif(df=df.select(used_features), target_column=labelCol, K=numFeatures)
            
        return VectorAssembler(inputCols=selected_features, outputCol=outputCol)