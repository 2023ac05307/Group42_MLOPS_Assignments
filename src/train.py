import os
import joblib
import logging
import time
import numpy as np
import psutil
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.tracking import MlflowClient

from data_preprocessing import load_data, preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _dense_example(X):
    """Return a 1-row input_example that works for signature logging (handles sparse)."""
    x1 = X[0:1]
    return x1.toarray() if hasattr(x1, "toarray") else x1

def train_and_log_model(name, model, X_train, y_train, X_test, y_test, preprocessor):
    """Train model, log metrics/artifacts, return (run_id, rmse)."""
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2   = float(r2_score(y_test, preds))

        # Save fitted preprocessor so inference can reproduce transforms
        preprocessor_path = "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(preprocessor_path, artifact_path="datapreprocessing")

        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        log_system_metrics()  # consider shortening duration to keep training snappy

        # ✅ Correct arg is artifact_path (not name)
        example = _dense_example(X_test)
        sig = mlflow.models.infer_signature(example, preds[0:1])
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=example,
            signature=sig
        )

        logging.info(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return run.info.run_id, rmse

def register_best_model(best_run_id, model_name="CaliforniaHousingModel"):
    """Register best run; (tag-based) mark as production. Consider real stages instead."""
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Wait until it's ready
    for _ in range(30):
        mv = client.get_model_version(name=model_name, version=result.version)
        if mv.status == "READY":
            break
        time.sleep(1)

    # Option A (tags): clear old 'stage' tags, then set production on the new version
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.version != result.version:
            try:
                client.delete_model_version_tag(name=model_name, version=mv.version, key="stage")
            except Exception:
                pass
    client.set_model_version_tag(name=model_name, version=result.version, key="stage", value="production")

    # Option B (preferred real stage):
    # client.transition_model_version_stage(
    #     name=model_name, version=result.version, stage="Production", archive_existing_versions=True
    # )

    logging.info(f"Registered model '{model_name}' v{result.version} as production.")

def log_system_metrics(interval=5, duration=15):
    """Log basic host metrics to the current MLflow run (shortened to 15s)."""
    start = time.time()
    while time.time() - start < duration:
        step = int(time.time() - start)
        mlflow.log_metric("cpu_percent", psutil.cpu_percent(), step=step)
        mlflow.log_metric("memory_percent", psutil.virtual_memory().percent, step=step)
        mlflow.log_metric("disk_percent", psutil.disk_usage('/').percent, step=step)
        time.sleep(interval)

def main():
    """data -> preprocess -> train LR & Tree -> pick best -> register."""
    mlflow.set_tracking_uri("http://ec2-44-203-72-164.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("California_Housing_Regression")

    df = load_data("data/housing.csv")

    # For LR (needs scaling) and Tree (doesn't), you can choose one:
    # Here we keep scaling=True; trees won't mind, it’s just unnecessary.
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, scale_numeric=True)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    }

    best_rmse, best_run = float("inf"), None
    for name, m in models.items():
        run_id, rmse = train_and_log_model(name, m, X_train, y_train, X_test, y_test, preprocessor)
        if rmse < best_rmse:
            best_rmse, best_run = rmse, run_id

    if best_run:
        register_best_model(best_run)

if __name__ == "__main__":
    main()
