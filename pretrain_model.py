"""
pretrain_model.py
Run ONCE before starting the app to train the agronomic model.

Usage:
    python pretrain_model.py              # full GridSearchCV (~10 min)
    python pretrain_model.py --fast       # skip GridSearch (~2 min)
"""

import sys
import os
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from config import AGRONOMIC_CSV, YIELD_CSV, SUPPORTED_CROPS
from utils.data_utils import load_agronomic_data, load_yield_data
from utils.model_utils import (
    train_agronomic_model, load_agronomic_model,
    compute_agro_scores, KAGGLE_TO_SUPPORTED
)

MODEL_PATH = Path(__file__).parent / "models" / "agronomic_model.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip GridSearchCV, use fixed params (faster)")
    args = parser.parse_args()
    use_grid = not args.fast

    print("=" * 65)
    print("  AgriAdvisor AI — Agronomic Model Training")
    print(f"  Mode: {'GridSearchCV (thorough)' if use_grid else 'Fast (fixed params)'}")
    print("=" * 65)

    # ── 1. Check data files ────────────────────────────────────────────────────
    missing = [p for p in [AGRONOMIC_CSV, YIELD_CSV] if not Path(p).exists()]
    if missing:
        for p in missing:
            logger.error(f"Missing: {p}")
        sys.exit(1)

    # ── 2. Load data ───────────────────────────────────────────────────────────
    logger.info("Loading agronomic dataset (soil + climate)...")
    agro_df = load_agronomic_data(AGRONOMIC_CSV)
    logger.info(f"  {len(agro_df):,} rows | "
                f"{agro_df['District'].nunique()} districts | "
                f"{agro_df['State'].nunique()} states")

    logger.info("Loading yield dataset (Kaggle crop production)...")
    yield_df = load_yield_data(YIELD_CSV)
    logger.info(f"  {len(yield_df):,} rows | "
                f"{yield_df['District_Name'].nunique()} districts | "
                f"{yield_df['Crop'].nunique()} crops")

    # Crop mapping stats
    kaggle_in_yield = set(yield_df["Crop"].unique())
    mappable = [k for k in KAGGLE_TO_SUPPORTED
                if k in kaggle_in_yield and KAGGLE_TO_SUPPORTED[k] in SUPPORTED_CROPS]
    logger.info(f"  Kaggle crops mapped to SUPPORTED_CROPS: {len(mappable)}/51")
    logger.info(f"  Mapped: {sorted([KAGGLE_TO_SUPPORTED[k] for k in mappable])}")

    # ── 3. Train ───────────────────────────────────────────────────────────────
    print()
    if use_grid:
        logger.info("GridSearchCV: 36 param combos × 5 folds = 180 fits")
        logger.info("Expected time: 8-15 minutes depending on CPU...")
    else:
        logger.info("Fast mode: training with proven fixed params...")

    model, le, scaler, acc = train_agronomic_model(
        agro_df, yield_df,
        save_path=MODEL_PATH,
        model_type="random_forest",
        use_grid_search=use_grid,
    )

    # ── 4. Results ─────────────────────────────────────────────────────────────
    print()
    print(f"  ✅ Model trained successfully!")
    print(f"  📊 Test accuracy  : {acc:.1%}")
    print(f"  🌱 Crops learned  : {len(le.classes_)}")
    print(f"  📋 Crop list      : {sorted(le.classes_.tolist())}")
    print(f"  🔧 Scaler applied : {'Yes (StandardScaler)' if scaler else 'No'}")
    print(f"  💾 Saved to       : {MODEL_PATH}")

    # Show best GridSearch params if available
    bundle_path = MODEL_PATH
    import pickle
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    if "best_params" in bundle:
        print(f"  🎯 Best params    : {bundle['best_params']}")
        print(f"  🎯 Best CV acc    : {bundle['best_cv_score']:.1%}")

    # ── 5. Sanity check on 3 district profiles ─────────────────────────────────
    print()
    logger.info("Sanity check on 3 district soil profiles...")
    model_l, le_l, scaler_l, features_l = load_agronomic_model(MODEL_PATH)

    test_profiles = {
        "Bangalore (Karnataka, tropical)": {
            "Nitrogen Value": 44.0, "Phosphorous value": 70.0,
            "Potassium value": 47.0, "pHsoil": 6.9,
            "Temperature_C": 22.4, "Humidity_%": 70.5, "Rainfall_mm": 880.0,
        },
        "Jamnagar (Gujarat, semi-arid)": {
            "Nitrogen Value": 22.0, "Phosphorous value": 55.0,
            "Potassium value": 65.0, "pHsoil": 7.8,
            "Temperature_C": 27.5, "Humidity_%": 62.0, "Rainfall_mm": 450.0,
        },
        "Thanjavur (Tamil Nadu, delta)": {
            "Nitrogen Value": 28.0, "Phosphorous value": 60.0,
            "Potassium value": 70.0, "pHsoil": 6.7,
            "Temperature_C": 28.5, "Humidity_%": 75.0, "Rainfall_mm": 1050.0,
        },
    }

    for label, profile in test_profiles.items():
        scores = compute_agro_scores(model_l, le_l, profile, features_l, scaler_l)
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  {label}:")
        for crop, score in top3:
            bar = "█" * int(score * 30)
            print(f"    {crop:25s}: {score:.4f}  {bar}")

    print()
    print("  ✅ Done. Run: streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
