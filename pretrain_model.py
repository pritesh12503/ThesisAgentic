"""
pretrain_model.py
Run this ONCE before starting the app to train and save the agronomic model.
This avoids the training delay on first Streamlit load.

Usage:
    python pretrain_model.py
"""

import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from config import AGRONOMIC_CSV, YIELD_CSV
from utils.data_utils import load_agronomic_data, load_yield_data
from utils.model_utils import train_agronomic_model, load_agronomic_model, compute_agro_scores

MODEL_PATH = Path(__file__).parent / "models" / "agronomic_model.pkl"


def main():
    print("=" * 60)
    print("  AgriAdvisor AI — Agronomic Model Pretraining")
    print("=" * 60)

    # ── Check data files exist ─────────────────────────────────────────────────
    for path in [AGRONOMIC_CSV, YIELD_CSV]:
        if not Path(path).exists():
            logger.error(f"Missing data file: {path}")
            logger.error("Please copy your CSV files into the data/ folder.")
            sys.exit(1)

    logger.info("Loading datasets...")
    agro_df  = load_agronomic_data(AGRONOMIC_CSV)
    yield_df = load_yield_data(YIELD_CSV)

    logger.info(f"Agronomic dataset: {len(agro_df):,} rows, "
                f"{agro_df['District'].nunique()} districts")
    logger.info(f"Yield dataset    : {len(yield_df):,} rows, "
                f"{yield_df['District_Name'].nunique()} districts, "
                f"{yield_df['Crop'].nunique()} crops")

    # ── Train model ────────────────────────────────────────────────────────────
    logger.info("Training Random Forest model (this may take ~30 seconds)...")
    model, le, acc = train_agronomic_model(
        agro_df, yield_df,
        save_path=MODEL_PATH,
        model_type="random_forest"
    )

    print()
    print(f"  ✅ Model trained successfully!")
    print(f"  📊 Test accuracy: {acc:.1%}")
    print(f"  🏷️  Crops: {le.classes_.tolist()}")
    print(f"  💾 Saved to: {MODEL_PATH}")

    # ── Quick sanity check ─────────────────────────────────────────────────────
    logger.info("Running sanity check on a sample district...")
    model_loaded, le_loaded, features = load_agronomic_model(MODEL_PATH)

    sample_profile = {
        "Nitrogen Value":    65.0,
        "Phosphorous value": 35.0,
        "Potassium value":   45.0,
        "pHsoil":            6.8,
        "Temperature_C":     28.0,
        "Humidity_%":        72.0,
        "Rainfall_mm":       950.0,
    }

    scores = compute_agro_scores(model_loaded, le_loaded, sample_profile, features)
    print()
    print("  Sanity check — scores for sample profile:")
    for crop, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 20)
        print(f"    {crop:10s}: {score:.4f}  {bar}")

    print()
    print("  ✅ Pretraining complete. You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
