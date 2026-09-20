from pathlib import Path
import json
import sys

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

LABELS = ["RED", "YELLOW", "GREEN", "BLACK"]
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}


def main():
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    else:
        data_path = BASE_DIR / "train_ml03.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {data_path}\n"
            "Place train_ml03.csv in backend/ or provide its path."
        )

    df = pd.read_csv(data_path)

    if "start_category" not in df.columns:
        raise ValueError("Training CSV must contain 'start_category'.")

    y = df["start_category"]
    X = df.drop(columns=["start_category"])

    if "subject_id" in X.columns:
        X = X.drop(columns=["subject_id"])

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    y_encoded = y.map(LABEL_MAP)

    if y_encoded.isna().any():
        unknown = sorted(y[y_encoded.isna()].unique())
        raise ValueError(f"Unknown labels found: {unknown}")

    pipeline.fit(X, y_encoded)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipeline,
            "classes": LABELS,
        },
        ARTIFACT_DIR / "triage_pipeline.joblib",
    )

    schema = {
        "features": X.columns.tolist(),
        "numeric": numeric_features,
        "categorical": categorical_features,
        "categories": {},
        "classes": LABELS,
    }

    for col in categorical_features:
        schema["categories"][col] = sorted(
            X[col].dropna().astype(str).unique().tolist()
        )

    with open(ARTIFACT_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print("\nModel trained successfully.")
    print(f"Training rows: {len(X)}")
    print(f"Features: {len(X.columns)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Saved: {ARTIFACT_DIR / 'triage_pipeline.joblib'}")
    print(f"Saved: {ARTIFACT_DIR / 'feature_schema.json'}")


if __name__ == "__main__":
    main()
