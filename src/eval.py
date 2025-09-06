import argparse
import os
import json
import mlflow
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from Evaluators.GiniBinary import GiniEvaluator
import pandas as pd


def evaluate_model(spark, data, project_config, model_uri):
    """
    Evaluates a trained model against a dataset using PySpark and generates a report.

    :param spark: The SparkSession object.
    :param data: The Spark DataFrame to evaluate.
    :param project_config: The project configuration dictionary.
    :param model_uri: URI of the model in MLflow.
    """
    data_path = project_config["file_paths"]["test_data"]
    not_features = project_config["not_features"]
    target_col = project_config["target"]
    output_path = project_config["file_paths"]["report"]

    # Load model as a Spark PipelineModel
    model = mlflow.spark.load_model(model_uri)

    # Make predictions
    predictions_df = model.transform(data).withColumnRenamed(target_col, "label")

    # Evaluate metrics
    gini_evaluator = GiniEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
    gini = gini_evaluator.evaluate(predictions_df)

    f1_evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1 = f1_evaluator.evaluate(predictions_df)

    precision_evaluator = MulticlassClassificationEvaluator(
        metricName="weightedPrecision"
    )
    precision = precision_evaluator.evaluate(predictions_df)

    recall_evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
    recall = recall_evaluator.evaluate(predictions_df)

    accuracy_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions_df)

    # Generate and save report
    report = f"""
# Model Evaluation Report (Spark)

## Model
- **URI:** {model_uri}

## Dataset
- **Path:** {data_path}

## Metrics
| Metric    | Value      |
|-----------|------------|
| Gini      | {gini:.4f}   |
| F1 Score  | {f1:.4f}     |
| Precision | {precision:.4f}|
| Recall    | {recall:.4f}   |
| Accuracy  | {accuracy:.4f} |
"""
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with PySpark.")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the project configuration file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name in mlflow",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        required=True,
        help="Model version in mlflow",
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        proj_conf = json.load(file)

    conda_python_path = proj_conf.get("conda_path")
    if conda_python_path:
        os.environ["PYSPARK_PYTHON"] = conda_python_path
        os.environ["PYSPARK_DRIVER_PYTHON"] = conda_python_path

    # Initialize a Spark session
    spark = (
        SparkSession.builder.appName("Evaluating")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.port", "4040")  # custom port
        .config("spark.ui.enabled", "true")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    try:
        mlflow.set_tracking_uri(proj_conf["mlflow_uri"])
        mlflow.set_experiment(proj_conf["experiment_name"])

        # Load Training Data
        test_data = spark.read.parquet(proj_conf["file_paths"]["test_data"])

        # Get model uri
        model_uri = f"models:/{args.model_name}/{args.model_version}"

        evaluate_model(spark, test_data, proj_conf, model_uri)
    finally:
        spark.stop()

