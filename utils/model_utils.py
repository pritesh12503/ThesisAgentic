"""
utils/model_utils.py
Train, save, and load the agronomic suitability model.
Also provides the scoring normalization helpers.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

AGRO_FEATURES = [
    "Nitrogen Value",
    "Phosphorous value",
    "Potassium value",
    "pHsoil",
    "Temperature_C",
    "Humidity_%",
    "Rainfall_mm",
]


def build_training_data(agro_df: pd.DataFrame, yield_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join agronomic soil+climate data with yield crop labels to create
    a classification training set.

    Works with BOTH old and new yield datasets:
      Old dataset columns: 'Dist Name', 'Crop', 'Yield_kg_per_ha'
      New dataset columns: 'District_Name', 'Crop', 'Yield_kg_per_ha'

    Strategy:
    - For each district present in the agronomic dataset,
      find the crop with highest mean yield from the yield dataset.
    - That (soil_profile, best_crop) pair becomes one training sample.
    """
    rows = []
    yield_df  = yield_df.copy()
    agro_df   = agro_df.copy()
    agro_df["District"] = agro_df["District"].astype(str).str.lower().str.strip()

    # Detect which district column the yield dataset uses
    if "District_Name" in yield_df.columns:
        dist_col = "District_Name"
    elif "Dist Name" in yield_df.columns:
        dist_col = "Dist Name"
    else:
        raise ValueError("Yield dataset must have 'District_Name' or 'Dist Name' column.")

    yield_df[dist_col] = yield_df[dist_col].astype(str).str.lower().str.strip()

    # Must have Yield_kg_per_ha (computed in load_yield_data for new dataset)
    if "Yield_kg_per_ha" not in yield_df.columns:
        raise ValueError("Yield dataset missing 'Yield_kg_per_ha' column. "
                         "Run load_yield_data() first.")

    agro_districts = agro_df["District"].unique()

    for dist in agro_districts:
        soil = agro_df[agro_df["District"] == dist][AGRO_FEATURES].mean()
        if soil.isnull().all():
            continue

        yield_dist = yield_df[yield_df[dist_col] == dist]
        if yield_dist.empty:
            continue

        # Best crop = highest mean yield in this district
        best_crop = (
            yield_dist.groupby("Crop")["Yield_kg_per_ha"]
            .mean()
            .idxmax()
        )

        row = soil.to_dict()
        row["crop_label"] = best_crop
        row["district"]   = dist
        rows.append(row)

    if not rows:
        raise ValueError(
            "No district overlap between agronomic and yield datasets. "
            "Check that district names are normalized consistently."
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"Training data built: {len(df)} samples, "
        f"{df['crop_label'].nunique()} unique crop labels: "
        f"{sorted(df['crop_label'].unique())}"
    )
    return df


def train_agronomic_model(
    agro_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    save_path: Path,
    model_type: str = "random_forest",
) -> Tuple[object, LabelEncoder, float]:
    """
    Train a crop suitability classifier.
    Returns (model, label_encoder, test_accuracy).
    """
    training_df = build_training_data(agro_df, yield_df)

    X = training_df[AGRO_FEATURES].values
    le = LabelEncoder()
    y = le.fit_transform(training_df["crop_label"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"Model trained. Test accuracy: {acc:.3f}")

    # Save model + encoder together
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "encoder": le, "features": AGRO_FEATURES}, f)

    logger.info(f"Model saved to {save_path}")
    return model, le, acc


def load_agronomic_model(save_path: Path):
    """Load saved model bundle. Returns (model, label_encoder, feature_names)."""
    with open(save_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["encoder"], bundle["features"]


def compute_agro_scores(
    model,
    label_encoder: LabelEncoder,
    feature_vector: Dict[str, float],
    feature_names: list,
) -> Dict[str, float]:
    """
    Given a soil+climate feature vector, return normalized suitability scores
    for all crops (0–1 scale, from predict_proba).
    """
    X = np.array([[feature_vector.get(f, 0.0) for f in feature_names]])
    proba = model.predict_proba(X)[0]
    classes = label_encoder.classes_
    scores = {cls: float(p) for cls, p in zip(classes, proba)}
    return scores


def compute_shap_values(
    model,
    label_encoder: LabelEncoder,
    feature_vector: Dict[str, float],
    feature_names: list,
) -> Dict[str, Dict[str, float]]:
    """
    Compute SHAP values for the given input vector.
    Returns {crop: {feature: shap_value}}.
    """
    X = np.array([[feature_vector.get(f, 0.0) for f in feature_names]])

    try:
        if not SHAP_AVAILABLE:
            raise ImportError("shap not installed")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)  # shape: [n_classes, n_samples, n_features]

        result = {}
        classes = label_encoder.classes_
        for i, crop in enumerate(classes):
            if isinstance(shap_vals, list):
                vals = shap_vals[i][0]
            else:
                vals = shap_vals[0]
            result[crop] = {fname: float(v) for fname, v in zip(feature_names, vals)}
        return result
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}. Returning feature importances instead.")
        importances = model.feature_importances_
        fallback = {}
        for crop in label_encoder.classes_:
            fallback[crop] = {fname: float(v) for fname, v in zip(feature_names, importances)}
        return fallback


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a dict of scores to [0, 1]."""
    if not scores:
        return scores
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
