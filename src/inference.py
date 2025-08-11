import mlflow
import pandas as pd
import joblib
import os
from mlflow.tracking import MlflowClient

_EXPECTED_COLUMNS = [
    "MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude","OceanProximity"
]

def _ensure_input_schema(df: pd.DataFrame):
    """Defensive: make sure expected cols exist before rename/transform."""
    missing = [c for c in _EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != 1:
        raise ValueError("predict_price expects a single-row DataFrame")

def get_production_model(model_name: str = "CaliforniaHousingModel"):
    """Fetch the production-tagged model & preprocessor; ensure only ONE version has stage=production.

    Behavior:
      - Connect to the MLflow Tracking Server.
      - Find all versions tagged with {'stage': 'production'}.
      - Keep the newest production version (highest version number).
      - Remove the 'stage' tag from all other versions (so only one remains labeled production).
      - Load and return the selected model + preprocessor.

    Notes:
      - This uses a *tag* convention. If you use MLflow's official stages,
        prefer `transition_model_version_stage(..., stage="Production", archive_existing_versions=True)`.
    """
    mlflow.set_tracking_uri("http://ec2-44-203-72-164.compute-1.amazonaws.com:5000/")
    print("MLflow tracking URI set.")

    client = MlflowClient()
    print("MLflow client created.")

    versions = list(client.search_model_versions(f"name='{model_name}'"))
    print(f"Found versions: {len(versions)}")

    # Find all versions with tag stage=production
    prod_versions = []
    for v in versions:
        tags = dict(v.tags)
        print(f"Checking version {v.version}, tags: {tags}")
        if tags.get("stage") == "production":
            prod_versions.append(v)

    if not prod_versions:
        raise Exception("No production model found.")

    # Choose the newest production version (highest version number)
    prod_versions.sort(key=lambda mv: int(mv.version))
    chosen = prod_versions[-1]
    print(f"Chosen production version: {chosen.version}")

    # Ensure ONLY the chosen version keeps the stage tag; remove from others
    for v in versions:
        if v.version != chosen.version:
            try:
                client.delete_model_version_tag(
                    name=model_name,
                    version=v.version,
                    key="stage"
                )
                print(f"Removed 'stage' tag from version {v.version}")
            except Exception as e:
                # Tag may not exist or permissions may block deletion; ignore quietly
                print(f"Skipping tag removal for version {v.version}: {e}")

    # Load model and its preprocessor artifact
    artifact_uri = chosen.source
    print(f"Loading model from {artifact_uri}")
    model = mlflow.pyfunc.load_model(artifact_uri)

    preprocessor_path = client.download_artifacts(
        run_id=chosen.run_id,
        path="datapreprocessing/preprocessor.pkl"
    )
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor




def predict_price(model, preprocessor, input_df: pd.DataFrame) -> float:
    """Run a single prediction using the provided model + preprocessor.

    Steps:
        - Rename inbound feature columns to match the training schema.
        - Apply the fitted preprocessor's transform.
        - Call the model's predict and return the scalar output.

    Args:
        model: The loaded MLflow pyfunc model (wrapping the sklearn estimator).
        preprocessor: The fitted preprocessing pipeline used at train time.
        input_df: A single-row DataFrame with user-friendly column names
                  (e.g., 'MedInc', 'HouseAge', 'AveRooms', ...).

    Returns:
        Predicted price as a float (same units as the training target).

    Notes:
        - Column renaming is critical: the preprocessor expects training-time names.
        - Ensure dtypes are compatible with the preprocessor (e.g., numeric vs string).
        - If your model was logged as a full Pipeline(preprocessor+model), you could
          skip manual preprocessing and feed raw features directly to `model.predict`.
    """
    _ensure_input_schema(input_df)

    input_df = input_df.rename(columns={
        "MedInc": "median_income",
        "HouseAge": "housing_median_age",
        "AveRooms": "total_rooms",
        "AveBedrms": "total_bedrooms",
        "Population": "population",
        "AveOccup": "households",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "OceanProximity": "ocean_proximity"
    })

    #print("✅ Columns in input_df just before transformation:", input_df.columns.tolist())
    #print("✅ Data:\n", input_df)

    transformed_input = preprocessor.transform(input_df)
    prediction = model.predict(transformed_input)
    return prediction[0]