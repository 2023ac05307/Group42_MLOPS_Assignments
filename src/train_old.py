import joblib
import mlflow
import mlflow.sklearn
import logging
import numpy as np
import psutil
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.tracking import MlflowClient
from data_preprocessing import load_data, preprocess_data

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_and_log_model(name, model, X_train, y_train, X_test, y_test, preprocessor):
    """Train a model, log metrics/artifacts to MLflow, and return (run_id, rmse).

    What gets logged:
      - Params: model_type
      - Metrics: rmse, r2_score (+ system metrics via `log_system_metrics`)
      - Artifacts: fitted preprocessor as joblib pkl
      - Model: logged as an MLflow model under the artifact path 'model'

    Note:
      - If X is sparse (e.g., due to OneHotEncoder), `infer_signature` can still work.
        If it complains, convert `X_test[0:1]` to dense via `.toarray()` first.

    Args:
        name: A short name for the run (e.g., 'LinearRegression').
        model: An sklearn regressor to fit.
        X_train, y_train, X_test, y_test: Train/test splits (preprocessed features).
        preprocessor: The fitted ColumnTransformer from your preprocessing step.

    Returns:
        Tuple(run_id: str, rmse: float)
    """
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        preprocessor_path = "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(preprocessor_path, artifact_path="datapreprocessing")

        # Save model into same folder
        # mlflow.sklearn.save_model(
        #     sk_model=model,
        #     path="models/model",  # Subdirectory
        #     input_example=X_test[0:1],
        #     signature=mlflow.models.infer_signature(X_test, predictions)
        # )

        # #Log entire folder as one MLflow artifact
        # mlflow.log_artifacts("models", artifact_path="model")

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        log_system_metrics()
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_test[0:1],
            signature=mlflow.models.infer_signature(X_test, predictions),
        )

        logging.info(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return run.info.run_id, rmse


def register_best_model(best_run_id, model_name="CaliforniaHousingModel"):
    """Register the best run's model in the MLflow Model Registry.

    Important:
      - This assumes your Tracking Server has a backend store that supports the Registry.
      - This sets a *tag* 'stage=production'. In MLflow, the canonical way to set the
        stage is `client.transition_model_version_stage(...)`. The tag is only a label.

    Args:
        best_run_id: The MLflow run_id with the best RMSE.
        model_name: Registered model name.
    """
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Wait until it's ready
    import time

    for _ in range(10):
        model_info = client.get_model_version(name=model_name, version=result.version)
        if model_info.status == "READY":
            break
        time.sleep(1)

    # Set tag to mark as "production"
    client.set_model_version_tag(
        name=model_name, version=result.version, key="stage", value="production"
    )

    logging.info(
        f"Registered and tagged model version {result.version} as 'production'"
    )


def log_system_metrics(interval=5, duration=60):
    """Log basic system metrics (cpu/memory/disk) to the *current* MLflow run.

    Args:
        interval: Seconds between samples.
        duration: Total seconds to log metrics.
    """
    start_time = time.time()
    while time.time() - start_time < duration:
        mlflow.log_metric(
            "cpu_percent", psutil.cpu_percent(), step=int(time.time() - start_time)
        )
        mlflow.log_metric(
            "memory_percent",
            psutil.virtual_memory().percent,
            step=int(time.time() - start_time),
        )
        mlflow.log_metric(
            "disk_percent",
            psutil.disk_usage("/").percent,
            step=int(time.time() - start_time),
        )
        time.sleep(interval)


def main():
    """End-to-end flow: data -> preprocess -> train 2 models -> pick best -> register."""

    mlflow.set_tracking_uri("http://ec2-44-203-72-164.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("California_Housing_Regression")

    file_path = "data/housing.csv"  # From root, since running from src/
    df = load_data(file_path)
    X, y, preprocessor = preprocess_data(df)
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    }

    best_rmse = float("inf")
    best_run_id = None

    for name, model in models.items():
        run_id, rmse = train_and_log_model(
            name, model, X_train, y_train, X_test, y_test, preprocessor
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_run_id = run_id

    if best_run_id:
        register_best_model(best_run_id)


if __name__ == "__main__":
    main()
