"""
utils/data_utils.py — UPDATED for new yield dataset

KEY CHANGE:
  load_yield_data() now reads crop_production_india.csv (Kaggle dataset).
  Columns: State_Name, District_Name, Crop_Year, Season, Crop, Area, Production
  Yield is MISSING in the raw file — we compute it here:
      Yield_kg_per_ha = (Production / Area) * 1000
      (Production is in Tonnes, Area in Hectares → multiply by 1000 for kg/ha)

  Rows where Area==0 or Production is NaN are dropped.
  Crop names are preserved exactly as-is (Title Case) to match CROP_TO_COMMODITY keys.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DISTRICT_ALIASES = {
    "bengaluru": "bangalore",
    "bengaluru urban": "bangalore",
    "bengaluru rural": "bangalore",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "prayagraj": "allahabad",
    "mysore": "mysuru",
    "tumkur": "tumakuru",
    "bellary": "ballari",
    "bijapur": "vijayapura",
    "gulbarga": "kalaburagi",
    "shimoga": "shivamogga",
    "belgaum": "belagavi",
    "dharwar": "dharwad",
}


def normalize_district(name: str) -> str:
    name = str(name).lower().strip()
    return DISTRICT_ALIASES.get(name, name)


# ── Agronomic CSV (unchanged) ─────────────────────────────────────────────────
def load_agronomic_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["District"] = df["District"].astype(str).apply(normalize_district)
    df["State"]    = df["State"].astype(str).str.lower().str.strip()
    return df


# ── NEW: Yield CSV (crop_production_india.csv) ────────────────────────────────
def load_yield_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean the Kaggle crop production dataset.

    Raw columns : State_Name, District_Name, Crop_Year, Season, Crop, Area, Production
    Added column: Yield_kg_per_ha = (Production / Area) * 1000

    Cleaning steps:
      1. Drop rows where Area == 0 or Area is NaN (cannot compute yield)
      2. Drop rows where Production is NaN
      3. Compute yield
      4. Drop rows where computed yield is infinite or NaN
      5. Normalize district names
      6. Preserve Crop names in their original Title Case
    """
    df = pd.read_csv(csv_path)

    # Standardize column names (handle any spacing issues)
    df.columns = df.columns.str.strip()

    # Drop unusable rows
    df = df[df["Area"].notna() & (df["Area"] > 0)]
    df = df[df["Production"].notna()]

    # Compute yield: Production (Tonnes) / Area (Ha) × 1000 = kg/ha
    df["Yield_kg_per_ha"] = (df["Production"] / df["Area"]) * 1000

    # Drop infinite / NaN yields (happens if Production=0 which is valid but yields 0)
    df = df[np.isfinite(df["Yield_kg_per_ha"])]

    # Remove extreme outliers (yields > 200 tonnes/ha are data errors)
    df = df[df["Yield_kg_per_ha"] < 200_000]

    # Normalize district names for matching
    df["District_Name"] = df["District_Name"].astype(str).apply(normalize_district)
    df["State_Name"]    = df["State_Name"].astype(str).str.strip()

    # Preserve Crop exactly as-is — must match CROP_TO_COMMODITY keys
    df["Crop"] = df["Crop"].astype(str).str.strip()

    logger.info(
        f"Yield dataset loaded: {len(df):,} rows, "
        f"{df['District_Name'].nunique()} districts, "
        f"{df['Crop'].nunique()} crops"
    )
    return df


# ── Price CSV (unchanged) ─────────────────────────────────────────────────────
def load_price_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["District"]  = df["District"].astype(str).apply(normalize_district)
    df["State"]     = df["State"].astype(str).str.lower().str.strip()
    df["Commodity"] = df["Commodity"].astype(str).str.strip()
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True, errors="coerce")
    return df


# ── Filtering functions ───────────────────────────────────────────────────────
def filter_agronomic(df: pd.DataFrame, district: str) -> pd.DataFrame:
    district = normalize_district(district)
    filtered = df[df["District"] == district]
    if filtered.empty:
        logger.warning(f"No agronomic data for '{district}'. Using full dataset mean.")
        return df
    return filtered


def filter_yield(df: pd.DataFrame, district: str) -> pd.DataFrame:
    """Filter yield data by district. Falls back to state/national if empty."""
    district = normalize_district(district)
    filtered = df[df["District_Name"] == district]
    if filtered.empty:
        logger.warning(f"No yield data for district '{district}'. Using national average.")
        return df
    return filtered


def filter_price(df: pd.DataFrame, commodities: list) -> pd.DataFrame:
    return df[df["Commodity"].isin(commodities)]


# ── Profile / stats helpers ───────────────────────────────────────────────────
def get_district_soil_profile(agro_df: pd.DataFrame, district: str) -> dict:
    features = [
        "Nitrogen Value", "Phosphorous value", "Potassium value",
        "pHsoil", "Temperature_C", "Humidity_%", "Rainfall_mm"
    ]
    filtered = filter_agronomic(agro_df, district)
    return {f: float(filtered[f].mean()) for f in features if f in filtered.columns}


def get_yield_stats(yield_df: pd.DataFrame, district: str, crop: str) -> dict:
    """
    Returns yield statistics for a crop in a district.
    Uses district data if available, falls back to national crop average.
    """
    filtered = filter_yield(yield_df, district)
    crop_data = filtered[filtered["Crop"] == crop]
    if crop_data.empty:
        crop_data = yield_df[yield_df["Crop"] == crop]
    if crop_data.empty:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    return {
        "mean":  float(crop_data["Yield_kg_per_ha"].mean()),
        "std":   float(crop_data["Yield_kg_per_ha"].std()),
        "count": int(len(crop_data)),
    }


def get_price_stats(price_df: pd.DataFrame, commodities: list) -> dict:
    """Returns mean modal price across matched commodities (Rs/quintal → Rs/kg)."""
    filtered = filter_price(price_df, commodities)
    if filtered.empty:
        return {"mean_per_kg": 0.0, "min_per_kg": 0.0, "max_per_kg": 0.0}
    return {
        "mean_per_kg": float(filtered["Modal Price"].mean()) / 100.0,
        "min_per_kg":  float(filtered["Min Price"].mean())   / 100.0,
        "max_per_kg":  float(filtered["Max Price"].mean())   / 100.0,
    }


def get_available_crops_for_district(yield_df: pd.DataFrame, district: str,
                                     supported_crops: list) -> list:
    """
    Returns which supported crops have yield data for the given district.
    Used to restrict recommendation to crops with local data.
    Falls back to all supported crops if none found locally.
    """
    district = normalize_district(district)
    local = yield_df[yield_df["District_Name"] == district]["Crop"].unique().tolist()
    available = [c for c in supported_crops if c in local]
    if not available:
        logger.warning(f"No local crop data for '{district}'. Using all supported crops.")
        return supported_crops
    return available
