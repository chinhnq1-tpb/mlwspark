import argparse
import json
import os
import mlflow
from pyspark.sql import SparkSession
from src.getAndPreprocess import read_data_from_mssql, preprocess, stratified_split
from src.trainRandDev import train
from src.eval import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline.")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the project configuration file.",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        proj_conf = json.load(file)

    with open("./configs/db.json", "r") as file:
        db_conf = json.load(file)

    conda_python_path = proj_conf.get("conda_path")
    if conda_python_path:
        os.environ["PYSPARK_PYTHON"] = conda_python_path
        os.environ["PYSPARK_DRIVER_PYTHON"] = conda_python_path

    # Initialize a Spark session
    spark = (
        SparkSession.builder.appName("ML-Pipeline")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.port", "4040")
        .config("spark.ui.enabled", "true")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    try:
        # --- Get and Preprocess Data ---
        sp_df = read_data_from_mssql(proj_conf["query"], db_conf, spark)
        sp_df = preprocess(sp_df)
        train_size = float(proj_conf["train_sz"])
        seed = int(proj_conf["random_seed"])
        train_df, test_df = stratified_split(
            sp_df, proj_conf["stratify_cols"], train_size, seed
        )

        # --- Train Model ---
        mlflow.set_tracking_uri(proj_conf["mlflow_uri"])
        mlflow.set_experiment(proj_conf["experiment_name"])

        with mlflow.start_run() as run:
            train(train_df, proj_conf)
            model_uri = f"runs:/{run.info.run_id}/logistic_regression_model"

            # --- Evaluate Model ---
            evaluate_model(spark, test_df, proj_conf, model_uri)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
