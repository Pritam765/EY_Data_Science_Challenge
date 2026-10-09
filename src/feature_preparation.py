import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def prepare_features(X):
    # Make a copy so the original data is not modified
    X = X.copy()

    # Convert Col3 to numeric
    X["Col3"] = pd.to_numeric(
        X["Col3"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    # Convert Col5 to datetime
    X["Col5"] = pd.to_datetime(
        X["Col5"],
        errors="coerce",
        format="mixed"
    )

    # Create features from Col2
    col2_text = X["Col2"].fillna("").astype(str)

    X["Col2_Length"] = col2_text.str.len()
    X["Col2_Has_Letter"] = (
        col2_text.str.contains(r"[A-Za-z]", regex=True).astype(int)
    )
    X["Col2_Has_Dash"] = (
        col2_text.str.contains("-", regex=False).astype(int)
    )

    # Track missing Col4 values before filling them
    X["Col4_Missing"] = X["Col4"].isna().astype(int)

    # Fill missing text values
    for col in ["Col1", "Col4", "Col6"]:
        X[col] = X[col].fillna("").astype(str)

    # Extract year and month from Col5
    X["Col5_Year"] = X["Col5"].dt.year
    X["Col5_Month"] = X["Col5"].dt.month

    # Return features in the order expected by the preprocessor
    return X[
        [
            "Col1", "Col2", "Col3", "Col4",
            "Col5", "Col6", "Col7",
            "Col2_Length", "Col2_Has_Letter",
            "Col2_Has_Dash", "Col4_Missing",
            "Col5_Year", "Col5_Month"
        ]
    ]


class FeaturePreparation(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return prepare_features(X)