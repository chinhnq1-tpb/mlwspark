from pyspark.ml import Transformer, Estimator
from pyspark.sql import dataframe as DataFrame
from pyspark import keyword_only, SparkContext
from pyspark.ml.param.shared import HasInputCols, HasLabelCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from optbinning import BinningProcess, BinningProcessSketch
from ..utils.utils import getCategoricalVars
import base64
import pickle
from typing import Iterator, Tuple, Any
import pandas as pd
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StructType, StructField, DoubleType


class ProcessBinningModel(Transformer, HasInputCols, DefaultParamsReadable, DefaultParamsWritable):
    pickledBin = Param(Params._dummy(), "pickledBin", "The pickled BinningProcess object, base64 encoded")

    @keyword_only
    def __init__(self,
                 inputCols,
                 pickledBin) -> None:
        super().__init__()
        self._setDefault(inputCols = None, pickledBin=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)
        
    @keyword_only
    def set_params(self,
                   inputCols,
                   pickledBin) -> None:
        kwargs = self._input_kwargs
        self._set(**kwargs)
    
    def setOptbinPickled(self, new_pickledBin: dict) -> Any:
        return self._set(pickledBin=new_pickledBin)

    def getOptbinPickled(self) -> dict:
        return self.getOrDefault(self.pickledBin)

    def _transform(self, df: DataFrame) -> DataFrame:
        pickled_bins = self.getOptbinPickled()
        input_cols = self.getInputCols() # Assuming you have a getter for inputCols
        woe_cols =  [f"{col}_WOE" for col in input_cols]
        
        # 1. Define the output schema for the UDF.
        # It's a StructType that holds all the new WOE columns.
        # Decode and unpickle ONCE, on the driver
        pickled_bins = self.getOptbinPickled()
        optbin_object = pickle.loads(base64.b64decode(pickled_bins))

        # Broadcast it to executors
        sc = SparkContext.getOrCreate()
        optbin_bc = sc.broadcast(optbin_object)

        output_schema = StructType([
            StructField(col, DoubleType(), True) for col in woe_cols
        ])

        # 2. Define the Pandas UDF to apply the transformation.
        # It takes all input columns and returns a single Struct column.
        @pandas_udf(output_schema, "iterator")
        def apply_binning_udf(iterator: Iterator[Tuple[pd.Series, ...]]) -> Iterator[pd.DataFrame]:
            optbin = optbin_bc.value
            for input_series in iterator:
                # 'input_series' is a tuple of pd.Series, one for each input column
                # Reconstruct the DataFrame from the tuple
                batch_df = pd.DataFrame(dict(zip(input_cols, input_series)))
                
                # Apply the transformation to the batch
                transformed_pdf = optbin.transform(batch_df)

                # The UDF must return a DataFrame matching the output schema
                # We rename the columns to match woe_cols
                transformed_pdf.columns = woe_cols
                
                yield transformed_pdf

        # 3. Apply the UDF and flatten the result
        # The UDF call returns a single column of type StructType
        result_df = df.withColumn("woe_struct", apply_binning_udf(*input_cols))

        # Now, flatten the Struct column into individual columns
        # We use a list comprehension to select the fields from the struct
        # We also keep the original columns from the DataFrame
        final_cols = [col("*")] + [col("woe_struct")[c].alias(c) for c in woe_cols]
        
        return result_df.select(*final_cols)


class ProcessBinningEstimator(Estimator, HasInputCols, HasLabelCol, DefaultParamsReadable, DefaultParamsWritable):
    woeCols = Param(
        Params._dummy(),
        "woeCols",
        "List of WOE columns after transformed",
        typeConverter=TypeConverters.toList,
    )
    @keyword_only
    def __init__(self,
                 inputCols: list,
                 labelCol: str,
                 woeCols: list=[]) -> None:
        super().__init__()
        self._setDefault(woeCols=[], inputCols=[], labelCol="label")
        kwargs = self._input_kwargs
        self.set_params(**kwargs)
        
    @keyword_only
    def set_params(self,
                 inputCols: list,
                 labelCol: str,
                 woeCols: list=[]) -> None:
        kwargs = self._input_kwargs
        self._set(**kwargs)
        if woeCols == []:
            for col in inputCols:
                woeCols.append(f"{col}_WOE")
        self._set(woeCols=woeCols)
        
    def setWOECols(self, new_woeCols: list) -> Any:
        return self._set(woeCols=new_woeCols)

    def getWOECols(self) -> BinningProcess:
        return self.getOrDefault(self.woeCols)
    
    def _fit(self, df: DataFrame) -> ProcessBinningModel:
        inputCols = self.getInputCols()
        labelCol = self.getLabelCol()
        new_woeCols = []
        
        for col in inputCols:
            new_woeCols.append(f"{col}_WOE")
        
        categorical_variables = getCategoricalVars(df, inputCols)

        
        def add(partition):
            columns = inputCols + [labelCol]
            df_pandas = pd.DataFrame.from_records(partition, columns=columns)
            if df_pandas.empty:
                return iter([])
            x = df_pandas[inputCols]
            y = df_pandas[labelCol]
            bpsketch = BinningProcessSketch(inputCols,
                                            categorical_variables=categorical_variables,
                                            binning_fit_params = {"monotonic_trend": "auto_heuristic"})
            bpsketch.add(x, y)
            return [bpsketch]

        def merge(bpsketch, other_bpsketch):
            bpsketch.merge(other_bpsketch)

            return bpsketch
        
        optbsketch = df.select(inputCols + [labelCol]).rdd.mapPartitions(lambda partition: add(partition)
                                                 ).treeReduce(merge)
        optbsketch.solve()
        pickled_optbin = base64.b64encode(pickle.dumps(optbsketch)).decode('ascii')
    
        return ProcessBinningModel(pickledBin=pickled_optbin, inputCols=inputCols)

if __name__ == "__main__":
    print("Hello")
