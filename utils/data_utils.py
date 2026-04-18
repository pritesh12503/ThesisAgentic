"""
utils/data_utils.py

KEY FIXES IN THIS VERSION:

Fix 1 — Rainfall scale mismatch (ROOT CAUSE of wrong crop scores):
  Crop_recommendation.csv uses rainfall in mm/month scale (~20-299mm)
  Our STATE_SOIL_PROFILES had annual rainfall (~400-3500mm)
  When passed to the StandardScaler trained on monthly data,
  annual values produce z-scores of +22 — completely out of distribution.
  Fix: STATE_SOIL_PROFILES now stores rainfall as monthly average
       (annual / 10 growing months). This matches the training scale exactly.

Fix 2 — K (Potassium) scale mismatch:
  Crop_recommendation.csv K range: 5-205 kg/ha (mean ~48)
  Our K values were 140-250 → z-score +2.8 to +4.0 (out of range)
  Fix: K values corrected to match real Soil Health Card data in kg/ha
       which happens to align with the base paper dataset range.

Fix 3 — State soil profiles now use SAME scale as training data.

Fix 4 — get_district_soil_profile() accepts state parameter for level-2 fallback.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DISTRICT_ALIASES = {
    "bengaluru": "bangalore", "bengaluru urban": "bangalore",
    "bengaluru rural": "bangalore", "bombay": "mumbai",
    "calcutta": "kolkata", "madras": "chennai",
    "prayagraj": "allahabad", "mysore": "mysuru",
    "tumkur": "tumakuru", "bellary": "ballari",
    "bijapur": "vijayapura", "gulbarga": "kalaburagi",
    "shimoga": "shivamogga", "belgaum": "belagavi",
    "dharwar": "dharwad",
}

def normalize_district(name: str) -> str:
    name = str(name).lower().strip()
    return DISTRICT_ALIASES.get(name, name)


# ── State soil profiles — CORRECTED SCALE ─────────────────────────────────────
# IMPORTANT: All values use the SAME scale as Crop_recommendation.csv training data:
#   N, P, K  : kg/ha (range ~0-140, matching base paper dataset)
#   pH       : standard soil pH (3.5-9.0)
#   Temp     : Celsius
#   Humidity : percentage
#   Rainfall : mm/MONTH average during growing season (NOT annual total)
#              = annual_mm / 10 (approximate growing months)
#              e.g. Chhattisgarh 1300mm annual → 130mm/month
#
# This is critical: the model was trained on monthly rainfall values (20-299mm range).
# Passing annual values (400-3500mm) produces z-scores 10-22x outside training range.

STATE_SOIL_PROFILES = {
    # Format: N, P, K (kg/ha), pH, Temp(C), Humidity(%), Rainfall(mm/month)
    "uttar pradesh":     {"Nitrogen Value":55, "Phosphorous value":25, "Potassium value":45, "pHsoil":7.8, "Temperature_C":25, "Humidity_%":65, "Rainfall_mm":90},
    "bihar":             {"Nitrogen Value":50, "Phosphorous value":22, "Potassium value":40, "pHsoil":7.5, "Temperature_C":26, "Humidity_%":70, "Rainfall_mm":110},
    "west bengal":       {"Nitrogen Value":45, "Phosphorous value":20, "Potassium value":35, "pHsoil":6.5, "Temperature_C":27, "Humidity_%":78, "Rainfall_mm":160},
    "punjab":            {"Nitrogen Value":65, "Phosphorous value":30, "Potassium value":50, "pHsoil":8.0, "Temperature_C":22, "Humidity_%":58, "Rainfall_mm":60},
    "haryana":           {"Nitrogen Value":60, "Phosphorous value":28, "Potassium value":48, "pHsoil":8.1, "Temperature_C":24, "Humidity_%":55, "Rainfall_mm":55},
    "rajasthan":         {"Nitrogen Value":30, "Phosphorous value":15, "Potassium value":38, "pHsoil":7.8, "Temperature_C":28, "Humidity_%":40, "Rainfall_mm":40},
    "madhya pradesh":    {"Nitrogen Value":45, "Phosphorous value":25, "Potassium value":42, "pHsoil":7.2, "Temperature_C":27, "Humidity_%":55, "Rainfall_mm":110},
    "chhattisgarh":      {"Nitrogen Value":42, "Phosphorous value":20, "Potassium value":40, "pHsoil":6.8, "Temperature_C":27, "Humidity_%":72, "Rainfall_mm":130},
    "jharkhand":         {"Nitrogen Value":40, "Phosphorous value":18, "Potassium value":35, "pHsoil":5.8, "Temperature_C":26, "Humidity_%":75, "Rainfall_mm":130},
    "odisha":            {"Nitrogen Value":43, "Phosphorous value":20, "Potassium value":38, "pHsoil":6.2, "Temperature_C":28, "Humidity_%":78, "Rainfall_mm":150},
    "andhra pradesh":    {"Nitrogen Value":48, "Phosphorous value":28, "Potassium value":45, "pHsoil":6.8, "Temperature_C":29, "Humidity_%":68, "Rainfall_mm":95},
    "telangana":         {"Nitrogen Value":45, "Phosphorous value":25, "Potassium value":42, "pHsoil":7.0, "Temperature_C":30, "Humidity_%":62, "Rainfall_mm":95},
    "karnataka":         {"Nitrogen Value":44, "Phosphorous value":22, "Potassium value":40, "pHsoil":6.8, "Temperature_C":25, "Humidity_%":65, "Rainfall_mm":90},
    "kerala":            {"Nitrogen Value":55, "Phosphorous value":35, "Potassium value":32, "pHsoil":5.8, "Temperature_C":28, "Humidity_%":85, "Rainfall_mm":250},
    "tamil nadu":        {"Nitrogen Value":42, "Phosphorous value":20, "Potassium value":35, "pHsoil":7.0, "Temperature_C":29, "Humidity_%":72, "Rainfall_mm":95},
    "maharashtra":       {"Nitrogen Value":48, "Phosphorous value":22, "Potassium value":40, "pHsoil":7.2, "Temperature_C":27, "Humidity_%":62, "Rainfall_mm":90},
    "gujarat":           {"Nitrogen Value":38, "Phosphorous value":18, "Potassium value":42, "pHsoil":7.8, "Temperature_C":28, "Humidity_%":58, "Rainfall_mm":70},
    "goa":               {"Nitrogen Value":55, "Phosphorous value":30, "Potassium value":32, "pHsoil":5.8, "Temperature_C":27, "Humidity_%":82, "Rainfall_mm":220},
    "assam":             {"Nitrogen Value":60, "Phosphorous value":28, "Potassium value":35, "pHsoil":5.5, "Temperature_C":25, "Humidity_%":85, "Rainfall_mm":180},
    "manipur":           {"Nitrogen Value":52, "Phosphorous value":22, "Potassium value":30, "pHsoil":5.5, "Temperature_C":22, "Humidity_%":80, "Rainfall_mm":150},
    "meghalaya":         {"Nitrogen Value":58, "Phosphorous value":25, "Potassium value":32, "pHsoil":5.2, "Temperature_C":20, "Humidity_%":88, "Rainfall_mm":250},
    "mizoram":           {"Nitrogen Value":55, "Phosphorous value":24, "Potassium value":30, "pHsoil":5.3, "Temperature_C":22, "Humidity_%":82, "Rainfall_mm":220},
    "nagaland":          {"Nitrogen Value":53, "Phosphorous value":22, "Potassium value":30, "pHsoil":5.4, "Temperature_C":21, "Humidity_%":82, "Rainfall_mm":200},
    "tripura":           {"Nitrogen Value":55, "Phosphorous value":25, "Potassium value":32, "pHsoil":5.5, "Temperature_C":26, "Humidity_%":82, "Rainfall_mm":200},
    "arunachal pradesh": {"Nitrogen Value":60, "Phosphorous value":28, "Potassium value":32, "pHsoil":5.5, "Temperature_C":20, "Humidity_%":85, "Rainfall_mm":280},
    "sikkim":            {"Nitrogen Value":58, "Phosphorous value":26, "Potassium value":30, "pHsoil":5.3, "Temperature_C":18, "Humidity_%":85, "Rainfall_mm":280},
    "himachal pradesh":  {"Nitrogen Value":52, "Phosphorous value":28, "Potassium value":38, "pHsoil":6.5, "Temperature_C":15, "Humidity_%":65, "Rainfall_mm":140},
    "uttarakhand":       {"Nitrogen Value":50, "Phosphorous value":25, "Potassium value":36, "pHsoil":6.8, "Temperature_C":18, "Humidity_%":68, "Rainfall_mm":150},
    "jammu and kashmir": {"Nitrogen Value":48, "Phosphorous value":22, "Potassium value":35, "pHsoil":7.0, "Temperature_C":14, "Humidity_%":58, "Rainfall_mm":110},
    "delhi":             {"Nitrogen Value":55, "Phosphorous value":25, "Potassium value":45, "pHsoil":7.8, "Temperature_C":25, "Humidity_%":60, "Rainfall_mm":65},
    "puducherry":        {"Nitrogen Value":42, "Phosphorous value":20, "Potassium value":35, "pHsoil":7.0, "Temperature_C":29, "Humidity_%":76, "Rainfall_mm":110},
    "chandigarh":        {"Nitrogen Value":62, "Phosphorous value":30, "Potassium value":50, "pHsoil":7.9, "Temperature_C":22, "Humidity_%":58, "Rainfall_mm":60},
    "ladakh":            {"Nitrogen Value":25, "Phosphorous value":12, "Potassium value":28, "pHsoil":7.5, "Temperature_C":5,  "Humidity_%":30, "Rainfall_mm":10},
    "andaman and nicobar islands": {"Nitrogen Value":58, "Phosphorous value":30, "Potassium value":32, "pHsoil":5.8, "Temperature_C":29, "Humidity_%":88, "Rainfall_mm":280},
}

NATIONAL_AVERAGE_PROFILE = {
    "Nitrogen Value":    50.0,
    "Phosphorous value": 25.0,
    "Potassium value":   40.0,
    "pHsoil":            7.0,
    "Temperature_C":     26.0,
    "Humidity_%":        68.0,
    "Rainfall_mm":       103.0,   # matches Crop_recommendation.csv mean
}


# ── CSV loaders ────────────────────────────────────────────────────────────────
def load_agronomic_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["District"] = df["District"].astype(str).apply(normalize_district)
    df["State"]    = df["State"].astype(str).str.lower().str.strip()
    return df


def load_yield_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df[df["Area"].notna() & (df["Area"] > 0)]
    df = df[df["Production"].notna()]
    df["Yield_kg_per_ha"] = (df["Production"] / df["Area"]) * 1000
    df = df[np.isfinite(df["Yield_kg_per_ha"])]
    df = df[df["Yield_kg_per_ha"] < 200_000]
    df["District_Name"] = df["District_Name"].astype(str).apply(normalize_district)
    df["State_Name"]    = df["State_Name"].astype(str).str.strip()
    df["Crop"]          = df["Crop"].astype(str).str.strip()
    logger.info(f"Yield dataset: {len(df):,} rows, {df['District_Name'].nunique()} districts")
    return df


def load_price_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["District"]     = df["District"].astype(str).apply(normalize_district)
    df["State"]        = df["State"].astype(str).str.lower().str.strip()
    df["Commodity"]    = df["Commodity"].astype(str).str.strip()
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True, errors="coerce")
    return df


# ── Filter helpers ─────────────────────────────────────────────────────────────
def filter_agronomic(df: pd.DataFrame, district: str) -> pd.DataFrame:
    district = normalize_district(district)
    filtered = df[df["District"] == district]
    if filtered.empty:
        return pd.DataFrame()
    return filtered


def filter_yield(df: pd.DataFrame, district: str) -> pd.DataFrame:
    district = normalize_district(district)
    filtered = df[df["District_Name"] == district]
    if filtered.empty:
        logger.warning(f"No yield data for '{district}'. Using national average.")
        return df
    return filtered


def filter_price(df: pd.DataFrame, commodities: list) -> pd.DataFrame:
    return df[df["Commodity"].isin(commodities)]


# ── Soil profile lookup ────────────────────────────────────────────────────────
def get_district_soil_profile(agro_df: pd.DataFrame, district: str,
                               state: str = "") -> dict:
    """
    Returns 7-feature soil+climate profile for a district.
    All values are in the SAME scale as Crop_recommendation.csv training data.
    Rainfall is mm/month (NOT annual total).

    Priority:
      1. Exact district in thesis_agronomic_dataset_clean.csv
         NOTE: This dataset has annual rainfall. We divide by 10 to convert
               to monthly scale before returning.
      2. STATE_SOIL_PROFILES (already in correct monthly scale)
      3. National average (monthly scale)
    """
    features = ["Nitrogen Value", "Phosphorous value", "Potassium value",
                "pHsoil", "Temperature_C", "Humidity_%", "Rainfall_mm"]

    # Level 1: district dataset
    filtered = filter_agronomic(agro_df, district)
    if not filtered.empty:
        profile = {f: float(filtered[f].mean()) for f in features if f in filtered.columns}
        if profile:
            # thesis_agronomic_dataset_clean.csv has annual rainfall (600-1200mm)
            # Convert to monthly scale to match training data
            if "Rainfall_mm" in profile:
                profile["Rainfall_mm"] = profile["Rainfall_mm"] / 10.0
            logger.info(f"Soil profile for '{district}': district dataset (rainfall scaled to monthly)")
            return profile

    # Level 2: state-level profile (already in monthly scale)
    state_key = state.lower().strip() if state else ""
    if state_key in STATE_SOIL_PROFILES:
        logger.info(f"Soil profile for '{district}': state average for '{state_key}'")
        return dict(STATE_SOIL_PROFILES[state_key])

    # Level 3: national average
    logger.warning(f"Soil profile for '{district}': using national average")
    return dict(NATIONAL_AVERAGE_PROFILE)


# ── Stats helpers ──────────────────────────────────────────────────────────────
def get_yield_stats(yield_df: pd.DataFrame, district: str, crop: str) -> dict:
    filtered  = filter_yield(yield_df, district)
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
    district  = normalize_district(district)
    local     = yield_df[yield_df["District_Name"] == district]["Crop"].unique().tolist()
    available = [c for c in supported_crops if c in local]
    if not available:
        return supported_crops
    return available
