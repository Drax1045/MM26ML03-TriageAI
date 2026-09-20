from pathlib import Path
from typing import Any

import json
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent import analyze_triage


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_PATH = Path(
    __import__("os").environ.get(
        "TRIAGE_ARTIFACT",
        BASE_DIR / "artifacts" / "triage_pipeline.joblib",
    )
)

SCHEMA_PATH = BASE_DIR / "artifacts" / "feature_schema.json"

CLASSES = ["RED", "YELLOW", "GREEN", "BLACK"]


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="TriageAI API",
    description="Healthcare triage classification and AI explanation API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Load trained ML pipeline
# ============================================================

if not ARTIFACT_PATH.exists():
    raise RuntimeError(
        f"Model artifact not found: {ARTIFACT_PATH}"
    )


artifact = joblib.load(ARTIFACT_PATH)


if isinstance(artifact, dict):
    pipeline = artifact["pipeline"]
    MODEL_CLASSES = artifact.get("classes", CLASSES)
else:
    pipeline = artifact
    MODEL_CLASSES = CLASSES


MODEL_CLASSES = [str(x) for x in MODEL_CLASSES]


# ============================================================
# Load feature schema
# ============================================================

if not SCHEMA_PATH.exists():
    raise RuntimeError(
        f"Feature schema not found: {SCHEMA_PATH}"
    )


with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    FEATURE_SCHEMA = json.load(f)


FEATURES = FEATURE_SCHEMA.get("features", [])
NUMERIC_FEATURES = FEATURE_SCHEMA.get("numeric", [])
CATEGORICAL_FEATURES = FEATURE_SCHEMA.get("categorical", [])
CATEGORIES = FEATURE_SCHEMA.get("categories", {})


if not FEATURES:
    raise RuntimeError("No features found in feature_schema.json")


# ============================================================
# Helpers
# ============================================================

def prepare_features(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Convert incoming JSON into the exact feature structure
    expected by the trained pipeline.
    """

    row = {}

    for feature in FEATURES:
        value = payload.get(feature, None)

        if value is None or value == "":
            row[feature] = np.nan
            continue

        if feature in NUMERIC_FEATURES:
            try:
                row[feature] = float(value)
            except (TypeError, ValueError):
                row[feature] = np.nan

        elif feature in CATEGORICAL_FEATURES:
            row[feature] = str(value)

        else:
            row[feature] = value

    return pd.DataFrame([row], columns=FEATURES)


def ordered_probabilities(
    probabilities: np.ndarray,
) -> dict[str, float]:
    """
    Convert model probabilities into the fixed public class order.
    """

    result = {}

    for class_name in CLASSES:
        result[class_name] = 0.0

    for index, class_name in enumerate(MODEL_CLASSES):
        if index < len(probabilities):
            if class_name in result:
                result[class_name] = float(probabilities[index])

    return result


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifact": str(ARTIFACT_PATH),
        "model": type(pipeline).__name__,
        "feature_count": len(FEATURES),
        "classes": CLASSES,
    }


# ============================================================
# Schema
# ============================================================

@app.get("/schema")
def schema():
    return {
        "features": FEATURES,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "categories": CATEGORIES,
    }


# ============================================================
# ML Prediction
# ============================================================

@app.post("/predict")
def predict(payload: dict[str, Any]):
    try:
        X = prepare_features(payload)

        probabilities_raw = pipeline.predict_proba(X)[0]

        probabilities = ordered_probabilities(
            np.asarray(probabilities_raw)
        )

        prediction_index = int(
            np.argmax(probabilities_raw)
        )

        prediction = MODEL_CLASSES[prediction_index]

        if prediction not in CLASSES:
            raise ValueError(
                f"Unexpected model class: {prediction}"
            )

        probability_sum = sum(probabilities.values())

        if not np.isfinite(probability_sum):
            raise ValueError(
                "Model returned invalid probabilities."
            )

        return {
            "prediction": prediction,
            "probabilities": probabilities,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# ============================================================
# AI Agent
# ============================================================

@app.post("/agent/analyze")
async def agent_analyze(payload: dict[str, Any]):
    """
    Generate a model-grounded AI explanation.

    Expected payload:

    {
        "features": {...},
        "prediction": "YELLOW",
        "probabilities": {
            "RED": 0.103,
            "YELLOW": 0.896,
            "GREEN": 0.0,
            "BLACK": 0.0
        }
    }
    """

    try:
        features = payload.get("features", {})
        prediction = payload.get("prediction")
        probabilities = payload.get("probabilities", {})

        # ----------------------------------------------------
        # Validate prediction
        # ----------------------------------------------------

        if not prediction:
            raise HTTPException(
                status_code=400,
                detail="Missing model prediction.",
            )

        if prediction not in CLASSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid prediction: {prediction}",
            )

        # ----------------------------------------------------
        # Validate probabilities
        # ----------------------------------------------------

        if not isinstance(probabilities, dict):
            raise HTTPException(
                status_code=400,
                detail="Probabilities must be an object.",
            )

        missing = [
            class_name
            for class_name in CLASSES
            if class_name not in probabilities
        ]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing probabilities: {missing}",
            )

        probability_values = []

        for class_name in CLASSES:
            try:
                value = float(probabilities[class_name])
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid probability for {class_name}.",
                )

            if not np.isfinite(value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid probability for {class_name}.",
                )

            probability_values.append(value)

        probability_sum = sum(probability_values)

        if not (0.99 <= probability_sum <= 1.01):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Probabilities must sum to approximately 1. "
                    f"Received {probability_sum:.6f}."
                ),
            )

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        if not isinstance(features, dict):
            raise HTTPException(
                status_code=400,
                detail="Features must be an object.",
            )

        # ----------------------------------------------------
        # Call AI agent
        # ----------------------------------------------------

        explanation = analyze_triage(
            features=features,
            prediction=prediction,
            probabilities=probabilities,
        )

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "explanation": explanation,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI agent failed: {str(exc)}",
        )