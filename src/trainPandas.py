import os
import pandas as pd
import json
from feature_engine.selection import MRMR, DropCorrelatedFeatures, DropConstantFeatures, SmartCorrelatedSelection
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier
from CustomeModels.CustomPassiveAgressive import CustomPassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from FeatureTransformers.WOEPandas import WOEProcessTransformer
from FeatureSelectors.CorrelationSelector import TopCorrelationSelector
from sklearn.kernel_approximation import Nystroem
from Evaluators.GiniSk import gini_score
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import argparse

def run_train_pipeline(
    config_path="configs/bscore_config.json",
    train_path="data/train_small.parquet",
    val_path="data/val_small.parquet",
    target_col=None,
    n_trials=20,
    n_splits=3
):

    # Load config
    with open(config_path, 'r') as file:
        proj_conf = json.load(file)

    woe_params = {"solver": "ls",
                  "monotonic_trend": "auto_asc_desc",
                  "class_weight": "balanced",
                  "max_n_prebins": 20}
    mrmr_params = {'max_features': int(proj_conf["num_features"]),
                    'regression': False,
                    'method': 'FCQ'}
    
    smartcorrel_params = {"selection_method": "variance"}
    # Load data
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Prepare features
    features_list = train_df.columns.to_list()
    unused_cols = proj_conf["not_features"]
    if proj_conf["target"] not in unused_cols:
        unused_cols.append(proj_conf["target"])
    for element in unused_cols:
        features_list.remove(element)

    if target_col is None:
        target_col = proj_conf["target"]

    X_train = train_df[features_list]
    y_train = train_df[target_col]
    X_val = val_df[features_list]
    y_val = val_df[target_col]

    def objective(trial):
        #max_features = trial.suggest_int("max_features", 15, min(20, len(features_list)))
        c = trial.suggest_loguniform("C", 10, 20)
        penalty = trial.suggest_categorical("penalty", ["l2"])
        solver = "saga"

        pipeline = Pipeline(steps=[
            ("rm_correlated", SmartCorrelatedSelection(**smartcorrel_params)),
            ("select_by_corr", TopCorrelationSelector(top_percent=0.5)),
            ("woe", WOEProcessTransformer(woe_params)),
            ("mrmr", MRMR(**mrmr_params)),
            #("log_reg", LogisticRegression(solver=solver, C=c, penalty=penalty, max_iter=1000, class_weight="balanced")),
            ('p_a_model', CustomPassiveAggressiveClassifier(loss = "squared_hinge", C=c, max_iter=1000, class_weight="balanced"))
        ])

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        pred = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        gini = gini_score(y_train, pred)
        return gini

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_C = best_params["C"]
    best_penalty = best_params["penalty"]
    best_solver = "saga" 

    #lr = LogisticRegression(solver=best_solver, C=best_C, penalty=best_penalty, max_iter=1000, class_weight="balanced")

    lr = CustomPassiveAggressiveClassifier(loss = "squared_hinge", C=best_C, max_iter=1000, class_weight="balanced")

    # Final pipeline with best params
    pipeline = Pipeline(steps=[
        ("rm_correlated", SmartCorrelatedSelection(**smartcorrel_params)),
        ("select_by_corr", TopCorrelationSelector(top_percent=0.5)),
        ("woe", WOEProcessTransformer(woe_params)),
        ("mrmr", MRMR(**mrmr_params)),
        ('calibrated_classifier', CalibratedClassifierCV(lr, method='isotonic', ensemble=False))
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        train_pred = pipeline.predict_proba(X_train)[:, 1]
        val_pred = pipeline.predict_proba(X_val)[:, 1]
        train_gini = gini_score(y_train, train_pred)
        val_gini = gini_score(y_val, val_pred)

        # Log best hyperparameters and metrics
        final_config = proj_conf | {"woe_params": woe_params} | {"mrmr_params": mrmr_params} | {
            "C": best_C,
            "penalty": best_penalty,
            "solver": best_solver,
            "features": pipeline.named_steps["mrmr"].get_feature_names_out()
        }
        mlflow.log_params(final_config)
        mlflow.log_metric("train_gini", train_gini)
        mlflow.log_metric("val_gini", val_gini)
        mlflow.sklearn.log_model(pipeline, "model")

    return {
        "pipeline": pipeline,
        "train_gini": train_gini,
        "val_gini": val_gini,
        "best_params": best_params
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pipeline with Optuna hyperparameter tuning and MLflow logging.")
    parser.add_argument('--config-path', type=str, required=True, help="Path to the project configuration file.")
    parser.add_argument('--train-path', type=str, default="data/train_small.parquet", help="Path to the training parquet file.")
    parser.add_argument('--val-path', type=str, default="data/val_small.parquet", help="Path to the validation parquet file.")
    parser.add_argument('--n-trials', type=int, default=19, help="Number of Optuna trials.")
    parser.add_argument('--n-splits', type=int, default=3, help="Number of CV folds.")
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        proj_conf = json.load(file)

    # Set MLflow tracking
    if "mlflow_uri" in proj_conf:
        mlflow.set_tracking_uri(proj_conf["mlflow_uri"])
    if "experiment_name" in proj_conf:
        mlflow.set_experiment(proj_conf["experiment_name"])

    result = run_train_pipeline(
        config_path=args.config_path,
        train_path=proj_conf["file_paths"]["train_data"],
        val_path=proj_conf["file_paths"]["val_data"],
        n_trials=args.n_trials,
        n_splits=args.n_splits
    )

    print("Best hyperparameters:", result["best_params"])
    print("Train Gini coefficient:", result["train_gini"])
    print("Val Gini coefficient:", result["val_gini"])
