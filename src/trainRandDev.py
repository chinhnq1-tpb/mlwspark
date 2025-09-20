from pyspark.sql import SparkSession
from time import time
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from FeatureSelectors.mrmr import MRMRSelector
from FeatureTransformers.WOE import ProcessBinningEstimator
from Evaluators.GiniBinary import GiniEvaluator
import argparse
import json
import mlflow
import mlflow.spark
import os
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from mlflow.data.spark_dataset import SparkDataset


def filter_low_cor_numerical(data, numerical_feat, proj_conf):
    # Step 1: Compute absolute correlations
    correlations = {
        col: abs(data.stat.corr(col, proj_conf["target"])) for col in numerical_feat
    }

    # Step 2: Remove None (in case of invalid correlation) and sort by absolute correlation
    correlations = {col: corr for col, corr in correlations.items() if corr is not None}
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Get top 30% features
    num_top_features = len(sorted_features) // 3  # integer division
    top_features = [col for col, _ in sorted_features[:num_top_features]]
    return top_features


def train(data, proj_conf):
    # MLflow tracking setup
    with mlflow.start_run():
        # spark_dataset = SparkDataset(data, source=proj_conf["file_paths"]["train_data"], name="training_data")
        # mlflow.log_input(spark_dataset, context="training")

        features_list = data.columns
        unused_cols = proj_conf["not_features"]
        if proj_conf["target"] not in unused_cols:
            unused_cols.append(proj_conf["target"])
        for element in unused_cols:
            features_list.remove(element)
        print(features_list)

        ## Step 2: Process Data
        ### Step 2.1: Encode Target
        # indexer = StringIndexer(inputCol=proj_conf["target"], outputCol="label")

        ### Step 2.2: Binning Data
        woe = ProcessBinningEstimator(
            inputCols=features_list, labelCol=proj_conf["target"]
        )

        ## Step 3: Feature Select
        num_features = int(proj_conf["num_features"])
        mrmr = MRMRSelector(
            inputCols=woe.getWOECols(),
            labelCol=proj_conf["target"],
            numFeatures=num_features,
        )

        ## Step 4: Training
        lr = LogisticRegression(labelCol=proj_conf["target"])
        evaluator = GiniEvaluator(
            rawPredictionCol="prediction", labelCol=proj_conf["target"]
        )
        if proj_conf["tuning"] == "true":
            paramGrid = (
                ParamGridBuilder()
                .addGrid(lr.regParam, [0.01, 0.1])
                .addGrid(lr.elasticNetParam, [0.5])
                .build()
            )
        elif proj_conf["tuning"] == "false":
            paramGrid = ParamGridBuilder().build()

        else:
            # Error handling for an invalid configuration
            print("Error: 'tuning' must be 'true' or 'false'.")
            exit()

        ## Step 5: Pipeline Assembling
        ### Assemble the preprocessing stages first
        pipeline = Pipeline(stages=[woe, mrmr, lr])

        # Initialize model variables to be populated by the conditional blocks
        best_model = None
        final_score = None

        # 6. Instantiate the CrossValidator or TrainValidationSplit
        if proj_conf["cross_validation"] == "true":
            print("Performing Cross-Validation...")
            num_folds = int(proj_conf["num_cv_fold"])
            crossval = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=paramGrid,
                evaluator=evaluator,
                numFolds=num_folds,
            )

            cv_model = crossval.fit(data)
            best_model = cv_model.bestModel
            final_score = cv_model.avgMetrics[0]

        elif proj_conf["cross_validation"] == "false":
            print("Performing Train-Validation Split...")
            val_sz = float(proj_conf["val_sz"])
            tvs = TrainValidationSplit(
                estimator=pipeline,
                estimatorParamMaps=paramGrid,
                evaluator=evaluator,
                trainRatio=1 - val_sz,
            )

            # Corrected: Use the 'data' DataFrame and 'tvs' to fit
            tvs_model = tvs.fit(data)

            # Corrected: Access the bestModel from tvs_model, not cv_model
            best_model = tvs_model.bestModel
            final_score = tvs_model.validationMetrics[0]

        else:
            # Error handling for an invalid configuration
            print("Error: 'cross_validation' must be 'true' or 'false'.")
            exit()

        # Assemble the pre pipeline and the best model and log
        mlflow.log_metric("gini", final_score)
        mlflow.spark.log_model(best_model, "logistic_regression_model")
        # Log project configuration
        proj_conf["selected_features"] = best_model.stages[-2].getInputCols()
        mlflow.log_params(proj_conf)

        # Log the best hyperparameters
        best_lr = best_model.stages[-1]
        mlflow.log_param("best_regParam", best_lr.getRegParam())
        mlflow.log_param("best_elasticNetParam", best_lr.getElasticNetParam())
