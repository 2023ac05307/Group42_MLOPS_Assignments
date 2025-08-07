import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path: str):
    """Load and return the dataset."""
    df = pd.read_csv(file_path)
    return df

def get_feature_columns(df: pd.DataFrame):
    """Return numeric and categorical columns."""
    cat_features = ["ocean_proximity"]
    num_features = df.drop(columns=cat_features + ["median_house_value"]).columns.tolist()
    return num_features, cat_features

def get_preprocessor(num_features, cat_features):
    """Return fitted preprocessing pipeline."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    return preprocessor

def preprocess_data(df: pd.DataFrame):
    """Preprocess the input DataFrame."""
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    num_features, cat_features = get_feature_columns(df)
    preprocessor = get_preprocessor(num_features, cat_features)
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor
