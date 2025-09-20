import mlflow
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from Evaluators.GiniBinary import GiniEvaluator


def evaluate_model(spark, data, project_config, model_uri):
    """
    Evaluates a trained model against a dataset using PySpark and generates a report.

    :param spark: The SparkSession object.
    :param data: The Spark DataFrame to evaluate.
    :param project_config: The project configuration dictionary.
    :param model_uri: URI of the model in MLflow.
    """
    data_path = project_config["file_paths"]["test_data"]
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
