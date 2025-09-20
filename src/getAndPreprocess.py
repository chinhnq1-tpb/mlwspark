from pyspark.sql.functions import concat_ws


def read_data_from_mssql(sql_query, db_config, spark):
    # Define the JDBC URL with Windows Authentication
    name = db_config["name"]
    port = db_config["port"]
    db_name = db_config["db_name"]
    jdbc_url = f"jdbc:sqlserver://{name}:{port};databaseName={db_name};integratedSecurity=true;trustServerCertificate=true"

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
    df = df.withColumn("strata", concat_ws("_", *stratified_cols))
    # Get the unique strata values
    unique_strata = [row.strata for row in df.select("strata").distinct().collect()]

    # Create a dictionary of fractions for an 80/20 split
    fractions = {stratum: train_sz for stratum in unique_strata}

    # Perform the stratified split
    train_df = df.sampleBy("strata", fractions, seed=random_seed)
    test_df = df.exceptAll(train_df)

    # You can drop the 'strata' column if you don't need it
    train_df = train_df.drop("strata")
    test_df = test_df.drop("strata")

    return train_df, test_df
