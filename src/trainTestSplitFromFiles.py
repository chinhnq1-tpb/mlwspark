from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lit
import argparse
import json
import os

def stratified_split(df, stratified_cols, train_sz, random_seed):
    df = df.withColumn('strata', concat_ws('_', *stratified_cols))
    # Get the unique strata values
    unique_strata = [row.strata for row in df.select('strata').distinct().collect()]

    # Create a dictionary of fractions for an 80/20 split
    train_fractions = {stratum: train_sz for stratum in unique_strata}

    # Perform the stratified split
    train_df = df.sampleBy('strata', train_fractions, seed=random_seed)

    test_df = df.exceptAll(train_df)

    # You can drop the 'strata' column if you don't need it
    train_df = train_df.drop('strata')
    test_df = test_df.drop('strata')

    return train_df, test_df

def preprocess_func(df):
    df_cleaned = df.dropna(subset=['DEFAULT_FLAG_CIC'])
    
    return df_cleaned

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process CSV and Parquet files with PySpark.")
    parser.add_argument('--config-path', type=str, required=True, help="Path to the project configuration file.")
    args = parser.parse_args()
    with open(args.config_path, 'r') as file:
            proj_conf = json.load(file)
    
    conda_python_path = proj_conf["conda_path"]
    os.environ['PYSPARK_PYTHON'] = conda_python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = conda_python_path
    
    # Initialize a Spark session
    spark = SparkSession.builder \
        .appName("MLflow Spark Example") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.local.dir", "D:/tmp/spark") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        
    # Read all Parquet files in the folder
    sp_df = spark.read.parquet(proj_conf["file_paths"]["full_data"])
    sp_df = preprocess_func(sp_df)
    train_size = float(proj_conf["train_sz"])
    seed = int(proj_conf["random_seed"])
    
    train_df, test_df = stratified_split(sp_df, proj_conf["stratify_cols"], train_size, seed)
    
    # Export data
    train_df.write.parquet(proj_conf["file_paths"]["train_data"], mode="overwrite")
    test_df.write.parquet(proj_conf["file_paths"]["test_data"], mode="overwrite")
    
    num_rows = train_df.count()  # Number of rows
    num_columns = len(train_df.columns)  # Number of columns

    # Print success message with shape info
    print(f"Train data has been successfully written")
    print(f"The DataFrame has {num_rows} rows and {num_columns} columns.")
    
    num_rows = test_df.count()  # Number of rows
    num_columns = len(test_df.columns)  # Number of columns

    # Print success message with shape info
    print(f"Test data has been successfully written")
    print(f"The DataFrame has {num_rows} rows and {num_columns} columns.")
    
    # Stop the SparkSession
    spark.stop()
