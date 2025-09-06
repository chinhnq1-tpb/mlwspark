from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("mrmr_test").master("local[2]").getOrCreate()
    df = spark.read.parquet(
        "/home/chinh/works/mrmr-python/mrmr-pyspark/data/flights-flag.parquet"
    )
    df.show(5)

    spark.stop()
