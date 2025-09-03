from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, BooleanType
import pandas as pd

def getCategoricalVars(df, columns):
    """
    Returns a list of categorical columns from the given list of columns in a DataFrame.
    
    Parameters:
    df (DataFrame): The input Spark DataFrame
    columns (list): A list of column names to check
    
    Returns:
    list: List of column names that are categorical
    """
    categorical_cols = []
    
    for col in columns:
        dtype = dict(df.dtypes).get(col)
        if dtype in ['string', 'boolean']:
            categorical_cols.append(col)
    
    return categorical_cols

def getSchema(df):
    schema = StructType([StructField(col, df.schema[col].dataType, True) for col in df.columns])
    return schema

def initSparkSession():
    return SparkSession.builder \
            .appName("MLflow Spark Example") \
            .master("local[6]") \
            .config("spark.driver.memory", "10g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.shuffle.partitions", "48") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

def split_features_by_type(df: pd.DataFrame, features: list):
    """
    Split a list of features into numerical and categorical based on DataFrame dtypes.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - features (list): List of column names to evaluate.

    Returns:
    - numerical_features (list): List of numerical feature names.
    - categorical_features (list): List of categorical feature names.
    """
    numerical_features = []
    categorical_features = []

    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            numerical_features.append(feature)
        else:
            categorical_features.append(feature)

    return numerical_features, categorical_features
