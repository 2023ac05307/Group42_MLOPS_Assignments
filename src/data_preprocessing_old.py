import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(file_path: str):
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the dataset.
    """
    df = pd.read_csv(file_path)
    return df


def get_feature_columns(df: pd.DataFrame):
    """Determine numeric and categorical feature columns.

    Assumptions:
      - Target column is named 'median_house_value'.
      - The only categorical feature is 'ocean_proximity'.

    Args:
        df: Input DataFrame that includes features and the target.

    Returns:
        (num_features, cat_features)
        num_features: list of numeric feature column names.
        cat_features: list of categorical feature column names.
    """
    cat_features = ["ocean_proximity"]
    num_features = df.drop(
        columns=cat_features + ["median_house_value"]
    ).columns.tolist()
    return num_features, cat_features


def get_preprocessor(num_features, cat_features):
    """Build an UNFITTED preprocessing pipeline.

    - Numeric: median imputation + standard scaling.
    - Categorical: most-frequent imputation + one-hot encoding.

    Args:
        num_features: List of numeric column names.
        cat_features: List of categorical column names.

    Returns:
        A scikit-learn ColumnTransformer.
    """
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_features), ("cat", cat_pipeline, cat_features)]
    )

    return preprocessor


def preprocess_data(df: pd.DataFrame):
    """Fit the preprocessor on features and transform them.

    Splits the DataFrame into features (X) and target (y), builds the preprocessing
    pipeline, fits it on X, and returns the transformed features along with y and
    the fitted preprocessor.

    Args:
        df: Input DataFrame containing features and the 'median_house_value' target.

    Returns:
        X_processed: Transformed feature matrix (numpy array).
        y: Target Series ('median_house_value').
        preprocessor: The fitted ColumnTransformer.
    """
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    num_features, cat_features = get_feature_columns(df)
    preprocessor = get_preprocessor(num_features, cat_features)
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor
