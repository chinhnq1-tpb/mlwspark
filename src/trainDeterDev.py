from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
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
from getAndPreprocess import stratified_split
from utils.utils import getCategoricalVars

def filter_low_cor_numerical(data, numerical_feat, proj_conf):
     # Step 1: Compute absolute correlations
    correlations = {
        col: abs(data.stat.corr(col, proj_conf["target"])) 
        for col in numerical_feat
    }

    # Step 2: Remove None (in case of invalid correlation) and sort by absolute correlation
    correlations = {col: corr for col, corr in correlations.items() if corr is not None}
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Get top 30% features
    num_top_features = len(sorted_features) // 3  # integer division
    top_features = [col for col, _ in sorted_features[:num_top_features]]
    return top_features


def train(df, proj_conf):
    # MLflow tracking setup
    train_data, dev_data = stratified_split(df, proj_conf["stratify_cols"], 1 - float(proj_conf["val_sz"]), int(proj_conf["random_seed"]))
    features_list = train_data.columns 
    unused_cols = proj_conf["not_features"]
    if proj_conf["target"] not in unused_cols:
        unused_cols.append(proj_conf["target"])
    for element in unused_cols:
        features_list.remove(element)
    categorical_feat = getCategoricalVars(train_df, features_list)
    numerical_feat = [element for element in features_list if element not in categorical_feat]
    
    top_features = filter_low_cor_numerical(train_data, numerical_feat, proj_conf) + categorical_feat 
    
    train_data = train_data.repartition(100)
    with mlflow.start_run():
        # spark_dataset = SparkDataset(data, source=proj_conf["file_paths"]["train_data"], name="training_data")
        # mlflow.log_input(spark_dataset, context="training")
        
        ## Step 2: Process Data
        ### Step 2.1: Encode Target
        indexer = StringIndexer(inputCol=proj_conf["target"], outputCol="label")

        ### Step 2.2: Binning Data
        woe = ProcessBinningEstimator(inputCols=top_features, labelCol=indexer.getOutputCol())

        ## Step 3: Feature Select
        num_features = int(proj_conf["num_features"])
        mrmr = MRMRSelector(inputCols=woe.getWOECols(), labelCol=indexer.getOutputCol(), numFeatures=num_features)

        ## Step 4: Training
        evaluator = GiniEvaluator(rawPredictionCol="prediction", labelCol="label")

        ## Step 5: Pipeline Assembling
        ### Assemble the preprocessing stages first
        preprocessing_pipeline = Pipeline(stages=[indexer, woe, mrmr])
        

        # 6. Instantiate the CrossValidator or TrainValidationSplit
        transformation_model = preprocessing_pipeline.fit(train_data)

        # Step 5: Transform both train and dev data
        transformed_train = transformation_model.transform(train_data)
        transformed_dev = transformation_model.transform(dev_data)
        
        # Step 4: Define hyperparameter grid
        reg_params = [0.01, 0.1] if proj_conf["tuning"] == "true" else [0.0]
        elastic_net_params = [0.5] if proj_conf["tuning"] == "true" else [0.0]
        best_model = None
        best_score = float("-inf")

        # Step 5: Manual tuning loop
        for reg in reg_params:
            for enet in elastic_net_params:
                lr = LogisticRegression(
                    featuresCol=mrmr.getOutputCol(),
                    labelCol="label",
                    regParam=reg,
                    elasticNetParam=enet
                )
                model = lr.fit(transformed_train)
                predictions = model.transform(transformed_dev)
                score = evaluator.evaluate(predictions)

                if score > best_score:
                    best_score = score
                    best_model = model

        # Step 5: Assemble the full pipeline model (reusing transformation_model)
        final_stages = transformation_model.stages + [best_model]
        final_pipeline_model = PipelineModel(stages=final_stages)
        # Assemble the pre pipeline and the best model and log
        mlflow.log_metric("gini", best_score)
        mlflow.spark.log_model(final_pipeline_model, "logistic_regression_model")
        # Log project configuration
        proj_conf["selected_features"] = final_pipeline_model.stages[-2].getInputCols()
        mlflow.log_params(proj_conf)
        
        # Log the best hyperparameters
        best_lr = best_model.stages[-1]
        mlflow.log_param("best_regParam", best_lr.getRegParam())
        mlflow.log_param("best_elasticNetParam", best_lr.getElasticNetParam())
            

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
        spark = initSparkSession() 
        
        mlflow.set_tracking_uri(proj_conf["mlflow_uri"])
        mlflow_experiment = mlflow.set_experiment(proj_conf["experiment_name"])

        # Load Training Data
        train_df = spark.read.parquet(proj_conf["file_paths"]["train_data"])

        # Train the model
        train(train_df, proj_conf)

        spark.stop()
