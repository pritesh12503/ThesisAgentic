"""
utils/model_utils.py
Train, save, and load the agronomic suitability model.

IMPROVED TRAINING STRATEGY:
- Uses ALL 119,784 soil rows (not just 48 district means)
- Assigns crop labels from yield data per district
- Top-3 crops per district used as training labels (more diversity)
- StandardScaler normalisation applied before training
- GridSearchCV hyperparameter tuning (focused grid, ~10 min on laptop)
- Kaggle crop name → SUPPORTED_CROPS mapping applied
- District name normalisation applied on both sides before joining
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

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

# Map Kaggle yield dataset crop names → our SUPPORTED_CROPS canonical names
KAGGLE_TO_SUPPORTED = {
    "Rice":                 "Rice",
    "Wheat":                "Wheat",
    "Maize":                "Maize",
    "Jowar":                "Jowar",
    "Bajra":                "Bajra",
    "Ragi":                 "Ragi",
    "Barley":               "Barley",
    "Arhar/Tur":            "Arhar/Tur",
    "Moong(Green Gram)":    "Moong(Green Gram)",
    "Urad":                 "Urad",
    "Gram":                 "Gram",
    "Masoor":               "Masoor",
    "Groundnut":            "Groundnut",
    "Rapeseed &Mustard":    "Rapeseed &Mustard",
    "Sunflower":            "Sunflower",
    "Soyabean":             "Soyabean",
    "Sesamum":              "Sesamum",
    "Linseed":              "Linseed",
    "Safflower":            "Safflower",
    "Cotton(lint)":         "Cotton(lint)",
    "Sugarcane":            "Sugarcane",
    "Coconut":              "Coconut",
    "Arecanut":             "Arecanut",
    "Coffee":               "Coffee",
    "Black pepper":         "Black pepper",
    "Potato":               "Potato",
    "Onion":                "Onion",
    "Tomato":               "Tomato",
    "Garlic":               "Garlic",
    "Ginger":               "Ginger",
    "Turmeric":             "Turmeric",
    "Tapioca":              "Tapioca",
    "Banana":               "Banana",
    "Mango":                "Mango",
    "Grapes":               "Grapes",
    "Papaya":               "Papaya",
    "Pineapple":            "Pineapple",
    "Guava":                "Guava",
    "Lemon":                "Lemon",
    "Orange":               "Orange Fruit",
    "Castor seed":          "Castor Seed",
    "Tobacco":              "Tobacco",
    "Brinjal":              "Brinjal",
    "Cabbage":              "Cabbage",
    "Cauliflower":          "Cauliflower",
    "Sweet potato":         "Sweet Potato",
    "Bitter Gourd":         "Bitter Gourd",
    "Bottle Gourd":         "Bottle Gourd",
    "Pumpkin":              "Pumpkin",
    "Apple":                "Apple",
    "Cowpea(Lobia)":        "Cowpea",
    "Cowpea":               "Cowpea",
}


def _normalise_kaggle_crop(crop_name: str) -> str:
    return KAGGLE_TO_SUPPORTED.get(crop_name.strip(), "")


def build_training_data(agro_df: pd.DataFrame, yield_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build training data using ALL soil rows with top-3 crop labels per district.
    Returns DataFrame with AGRO_FEATURES columns + 'crop_label' + 'district'.
    """
    agro_df  = agro_df.copy()
    yield_df = yield_df.copy()

    agro_df["District"] = agro_df["District"].astype(str).str.lower().str.strip()

    if "District_Name" in yield_df.columns:
        dist_col = "District_Name"
    elif "Dist Name" in yield_df.columns:
        dist_col = "Dist Name"
    else:
        raise ValueError("Yield dataset must have 'District_Name' or 'Dist Name' column.")

    yield_df[dist_col] = yield_df[dist_col].astype(str).str.lower().str.strip()

    if "Yield_kg_per_ha" not in yield_df.columns:
        raise ValueError("Yield dataset missing 'Yield_kg_per_ha'. Run load_yield_data() first.")

    yield_df["Crop_Canonical"] = yield_df["Crop"].apply(_normalise_kaggle_crop)
    yield_df = yield_df[yield_df["Crop_Canonical"] != ""]

    agro_districts = agro_df["District"].unique()

    district_crop_map: Dict[str, list] = {}
    matched = 0
    for dist in agro_districts:
        yield_dist = yield_df[yield_df[dist_col] == dist]
        if yield_dist.empty:
            continue
        matched += 1
        top_crops = (
            yield_dist.groupby("Crop_Canonical")["Yield_kg_per_ha"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        district_crop_map[dist] = top_crops

    logger.info(f"Matched {matched}/{len(agro_districts)} agro districts to yield data")

    if not district_crop_map:
        raise ValueError("No district overlap between agronomic and yield datasets.")

    rows = []
    for dist, top_crops in district_crop_map.items():
        dist_rows = agro_df[agro_df["District"] == dist][AGRO_FEATURES].dropna()
        if dist_rows.empty:
            continue
        for crop in top_crops:
            for _, soil_row in dist_rows.iterrows():
                row = soil_row.to_dict()
                row["crop_label"] = crop
                row["district"]   = dist
                rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        f"Training data: {len(df):,} samples, "
        f"{df['crop_label'].nunique()} crop classes: "
        f"{sorted(df['crop_label'].unique())}"
    )
    return df


def train_agronomic_model(
    agro_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    save_path: Path,
    model_type: str = "random_forest",
    use_grid_search: bool = True,
) -> Tuple[object, LabelEncoder, StandardScaler, float]:
    """
    Train crop suitability classifier with GridSearchCV tuning.

    Pipeline:
      1. Build training data (all soil rows × top-3 crop labels)
      2. StandardScaler normalisation
      3. GridSearchCV (focused grid, 5-fold stratified CV)
      4. Evaluate on held-out test set
      5. Save model + scaler + encoder together

    Returns: (model, label_encoder, scaler, test_accuracy)
    """
    training_df = build_training_data(agro_df, yield_df)

    X = training_df[AGRO_FEATURES].values
    le = LabelEncoder()
    y  = le.fit_transform(training_df["crop_label"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── StandardScaler normalisation ──────────────────────────────────────────
    # Fit on train only, apply to both — prevents data leakage
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── GridSearchCV ──────────────────────────────────────────────────────────
    # Focused grid: 3×3×2×2 = 36 combos × 5 folds = 180 fits
    # On 350k rows this takes ~8-12 min. Set use_grid_search=False for quick retrains.
    if use_grid_search:
        logger.info("Running GridSearchCV (this takes ~8-12 minutes)...")

        param_grid = {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [8, 12, 20],
            "min_samples_leaf": [3, 5],
            "max_features":     ["sqrt", "log2"],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train_sc, y_train)

        best_params = grid.best_params_
        best_cv_score = grid.best_score_
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best CV accuracy: {best_cv_score:.3f}")

        model = grid.best_estimator_

    else:
        logger.info("Skipping GridSearch — training with fixed best params...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_sc, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.3f}")
    logger.info("\n" + classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    # ── Save everything ───────────────────────────────────────────────────────
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":    model,
        "encoder":  le,
        "scaler":   scaler,
        "features": AGRO_FEATURES,
    }
    if use_grid_search:
        bundle["best_params"]   = grid.best_params_
        bundle["best_cv_score"] = grid.best_score_

    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)

    logger.info(f"Model saved to {save_path}")
    return model, le, scaler, acc


def load_agronomic_model(save_path: Path):
    """
    Load saved model bundle.
    Returns (model, label_encoder, scaler, feature_names).
    Backward-compatible: if no scaler in bundle, returns None for scaler.
    """
    with open(save_path, "rb") as f:
        bundle = pickle.load(f)
    scaler = bundle.get("scaler", None)
    return bundle["model"], bundle["encoder"], scaler, bundle["features"]


def compute_agro_scores(
    model,
    label_encoder: LabelEncoder,
    feature_vector: Dict[str, float],
    feature_names: list,
    scaler: StandardScaler = None,
) -> Dict[str, float]:
    """Return predict_proba scores keyed by crop canonical name."""
    X = np.array([[feature_vector.get(f, 0.0) for f in feature_names]])
    if scaler is not None:
        X = scaler.transform(X)
    proba  = model.predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(label_encoder.classes_, proba)}


def compute_shap_values(
    model,
    label_encoder: LabelEncoder,
    feature_vector: Dict[str, float],
    feature_names: list,
    scaler: StandardScaler = None,
) -> Dict[str, Dict[str, float]]:
    """Compute SHAP values. Returns {crop: {feature: shap_value}}."""
    X = np.array([[feature_vector.get(f, 0.0) for f in feature_names]])
    if scaler is not None:
        X = scaler.transform(X)

    try:
        if not SHAP_AVAILABLE:
            raise ImportError("shap not installed")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        result = {}
        for i, crop in enumerate(label_encoder.classes_):
            if isinstance(shap_vals, list):
                vals = shap_vals[i][0]
            else:
                vals = shap_vals[0]
            result[crop] = {fname: float(v) for fname, v in zip(feature_names, vals)}
        return result
    except Exception as e:
        logger.warning(f"SHAP failed: {e}. Using feature importances.")
        importances = model.feature_importances_
        return {
            crop: {fname: float(v) for fname, v in zip(feature_names, importances)}
            for crop in label_encoder.classes_
        }


def compute_lime_explanation(
    model,
    label_encoder: LabelEncoder,
    feature_vector: Dict[str, float],
    feature_names: list,
    training_data: np.ndarray,
    crop: str,
    scaler: StandardScaler = None,
) -> str:
    """
    Compute LIME local explanation for the recommended crop.
    Returns a plain-text summary of top 5 features (not a chart — saves UI space).
    Used to validate SHAP explanations for thesis methodology.
    """
    if not LIME_AVAILABLE:
        return "LIME not available (pip install lime)"

    try:
        X_instance = np.array([feature_vector.get(f, 0.0) for f in feature_names])

        # Scale training data and instance if scaler available
        if scaler is not None:
            training_data_sc = scaler.transform(training_data)
            X_instance_sc    = scaler.transform(X_instance.reshape(1, -1))[0]
        else:
            training_data_sc = training_data
            X_instance_sc    = X_instance

        explainer = LimeTabularExplainer(
            training_data_sc,
            feature_names=feature_names,
            class_names=label_encoder.classes_,
            mode="classification",
            random_state=42,
        )

        crop_idx = list(label_encoder.classes_).index(crop) \
                   if crop in label_encoder.classes_ else 0

        exp = explainer.explain_instance(
            X_instance_sc,
            model.predict_proba,
            num_features=5,
            labels=[crop_idx],
        )

        # Format as readable text — no chart needed
        lines = [f"LIME explanation for {crop}:"]
        for feat, weight in exp.as_list(label=crop_idx):
            direction = "supports" if weight > 0 else "reduces"
            lines.append(f"  • {feat}: {direction} suitability ({weight:+.4f})")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"LIME explanation failed: {e}")
        return f"LIME unavailable: {str(e)}"


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a dict of scores to [0, 1]."""
    if not scores:
        return scores
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
