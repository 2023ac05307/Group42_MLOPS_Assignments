import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(file_path)

def get_feature_columns(df: pd.DataFrame):
    """Return numeric and categorical columns (target: 'median_house_value')."""
    cat_features = ["ocean_proximity"]
    num_features = df.drop(columns=cat_features + ["median_house_value"]).columns.tolist()
    return num_features, cat_features

def get_preprocessor(num_features, cat_features, scale_numeric: bool = True) -> ColumnTransformer:
    """Build an UNFITTED preprocessor. Scale numeric only if requested."""
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(num_steps)

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Keep default sparse output for wide OHE; change to sparse_output=False if you need dense
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

def preprocess_data(df: pd.DataFrame, scale_numeric: bool = True):
    """Split, then fit preprocessor on train only; return transformed splits + preprocessor."""
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_features, cat_features = get_feature_columns(df)
    preprocessor = get_preprocessor(num_features, cat_features, scale_numeric=scale_numeric)

    X_train_proc = preprocessor.fit_transform(X_train)   # fit ONLY on train
    X_test_proc  = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor
