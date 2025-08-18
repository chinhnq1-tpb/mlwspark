from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lit
import argparse
import json
import os

def read_data_from_mssql(sql_query, db_config, spark):
    # Define the JDBC URL with Windows Authentication
    name = db_config["name"]
    port = db_config["port"]
    db_name = db_config["db_name"]
    jdbc_url = f"jdbc:sqlserver://{name}:{port};databaseName={db_name};integratedSecurity=true;trustServerCertificate=true"
    
    # Define properties for the connection (no user and password needed)
    properties = {
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    }

    
    # Read data using the SQL query
    df = spark.read.jdbc(url=jdbc_url, table=f"({sql_query}) as query")
    
    return df


def write_to_parquet(df, output_path):
    # Write the DataFrame to a Parquet file
    df.write.parquet(output_path, mode="overwrite")
    print(f"Data has been written to {output_path} as Parquet.")

## Write processing functions here
def preprocess(df):
    return df

def stratified_split(df, stratified_cols, train_sz, random_seed):
    df = df.withColumn('strata', concat_ws('_', *stratified_cols))
    # Get the unique strata values
    unique_strata = [row.strata for row in df.select('strata').distinct().collect()]

    # Create a dictionary of fractions for an 80/20 split
    fractions = {stratum: train_sz for stratum in unique_strata}

    # Perform the stratified split
    train_df = df.sampleBy('strata', fractions, seed=random_seed)
    test_df = df.exceptAll(train_df)

    # You can drop the 'strata' column if you don't need it
    train_df = train_df.drop('strata')
    test_df = test_df.drop('strata')

    return train_df, test_df

if __name__ == "__main__":
    os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-9.0.4"
    os.environ["HADOOP_HOME"] = r"E:\a_ChinhNQ1\drivers\hadoop"
    # Set the SPARK_HOME environment variable
    os.environ['SPARK_HOME'] = r"E:\a_ChinhNQ1\bin\apache-spark\spark-3.5.6-bin-hadoop3"

    # Add Spark's bin directory to the PATH
    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.environ['SPARK_HOME'], 'bin')

    # Add Hadoop's bin directory to the PATH (important for native libraries)
    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.environ['HADOOP_HOME'], 'bin')

    # Initialize a Spark session
    spark = (SparkSession.builder.appName("MSSQLToParquet")
            .config("spark.local.dir", r"E:\a_ChinhNQ1\tmp")
            .config("spark.driver.extraClassPath", "E:/a_ChinhNQ1/drivers/sparks/mssql-jdbc-12.10.1.jre8.jar") \
            .config("spark.executor.extraLibraryPath", "E:/a_ChinhNQ1/drivers/sparks/") \
            .config("spark.driver.extraJavaOptions", "-Djava.library.path=E:/a_ChinhNQ1/drivers/sparks/")\
            .config("spark.hadoop.hadoop.native.io", "false")\
            .config("spark.hadoop.fs.file.impl.disable.cache", "true")
            .getOrCreate())
    
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Process CSV and Parquet files with PySpark.")
    parser.add_argument('--config-path', type=str, required=True, help="Path to the project configuration file.")
    args = parser.parse_args()
    
    # Read config files
    with open('./configs/db.json', 'r') as file:
        db_conf = json.load(file)
        
    with open(args.config_path, 'r') as file:
        proj_conf = json.load(file)
        
    # Get data from MSSQL
    sp_df = read_data_from_mssql(proj_conf["query"], db_conf)
    sp_df = preprocess(sp_df)
    train_size = float(proj_conf["train_sz"])
    seed = int(proj_conf["random_seed"])
    train_df, test_df = stratified_split(sp_df, proj_conf["stratify_cols"], train_size, seed)
    
    # Export data
    train_df.write.parquet(proj_conf["file_paths"]["train_data"], mode="overwrite")
    test_df.write.parquet(proj_conf["file_paths"]["test_data"], mode="overwrite")
    # train_df.write.format("avro").save(proj_conf["file_paths"]["train_data"])
    # test_df.write.format("avro").save(proj_conf["file_paths"]["test_data"])
    
    num_rows = train_df.count()  # Number of rows
    num_columns = len(train_df.columns)  # Number of columns

    # Print success message with shape info
    print(f"Train data has been successfully written to {args.output_path} as Parquet.")
    print(f"The DataFrame has {num_rows} rows and {num_columns} columns.")
    
    num_rows = test_df.count()  # Number of rows
    num_columns = len(test_df.columns)  # Number of columns

    # Print success message with shape info
    print(f"Test data has been successfully written to {args.output_path} as Parquet.")
    print(f"The DataFrame has {num_rows} rows and {num_columns} columns.")
    
    # Stop the SparkSession
    spark.stop()