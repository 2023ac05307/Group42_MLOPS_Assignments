import mlflow
import pandas as pd
import joblib
import os
from mlflow.tracking import MlflowClient

def get_production_model(model_name="CaliforniaHousingModel"):
    mlflow.set_tracking_uri("http://localhost:5000/")
    print("MLflow tracking URI set.")
    client = MlflowClient()
    print("MLflow client created.")
    versions = client.search_model_versions(f"name='{model_name}'")
    print(f"Found versions: {len(versions)}")

    for v in versions:
        tags = dict(v.tags)
        print(f"Checking version {v.version}, tags: {tags}")
        if tags.get("stage") == "production":
            artifact_uri = v.source
            print(f"Loading model from {artifact_uri}")
            model_path = mlflow.pyfunc.load_model(artifact_uri)
            preprocessor_path = client.download_artifacts(run_id=v.run_id, path="datapreprocessing/preprocessor.pkl")
            preprocessor = joblib.load(preprocessor_path)
            return model_path, preprocessor

    raise Exception("No production model found.")


def predict_price(model, preprocessor, input_df: pd.DataFrame) -> float:
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

