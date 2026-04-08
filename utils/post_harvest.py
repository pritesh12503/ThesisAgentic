"""
utils/post_harvest.py

All logic for the post-harvest advisory module.
Three independent sub-modules:
  1. Seasonal analyser  — when to sell (price seasonality tables)
  2. Storage analyser   — can the farmer afford to wait (cost vs uplift)
  3. Weather signal     — 7-day forecast override via Open-Meteo (free, no key)
  4. Channel selector   — local-only: APMC / e-NAM / FCI / processor

This module is completely independent of Score_agro and Score_economic.
It runs after the crop is recommended and answers: "now what?"
"""

import requests
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# ── District coordinate lookup ─────────────────────────────────────────────────
COORDS_CSV = Path(__file__).parent.parent / "data" / "district_coordinates.csv"

def get_district_coordinates(district: str, state: str) -> Optional[tuple]:
    """Returns (lat, lon) for a district or None if not found."""
    try:
        df = pd.read_csv(COORDS_CSV)
        df["district_lower"] = df["district"].str.lower().str.strip()
        df["state_lower"]    = df["state"].str.lower().str.strip()
        match = df[
            (df["district_lower"] == district.lower().strip()) &
            (df["state_lower"]    == state.lower().strip())
        ]
        if match.empty:
            # Try district only — different state spelling possible
            match = df[df["district_lower"] == district.lower().strip()]
        if match.empty:
            return None
        row = match.iloc[0]
        return float(row["lat"]), float(row["lon"])
    except Exception as e:
        logger.warning(f"Could not load coordinates for {district}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TABLES
# ═══════════════════════════════════════════════════════════════════════════════

# Month-wise price index per crop
# 100 = harvest-month price (baseline). Values > 100 = price above harvest month.
# harvest_months: calendar months when crop is typically harvested
# peak_month: calendar month with highest price historically
# peak_uplift_pct: % above harvest-month price at peak
PRICE_SEASONALITY = {
    # ── Cereals ────────────────────────────────────────────────────────────────
    "Rice": {
        "harvest_months": [10, 11], "peak_month": 2, "peak_uplift_pct": 18,
        "monthly_index": {1:114, 2:118, 3:115, 4:110, 5:107, 6:104,
                          7:102, 8:101, 9:100, 10:100, 11:95, 12:105},
    },
    "Wheat": {
        "harvest_months": [3, 4], "peak_month": 11, "peak_uplift_pct": 20,
        "monthly_index": {1:108, 2:104, 3:100, 4:96, 5:98, 6:100,
                          7:104, 8:107, 9:110, 10:112, 11:120, 12:115},
    },
    "Maize": {
        "harvest_months": [10, 11], "peak_month": 5, "peak_uplift_pct": 22,
        "monthly_index": {1:110, 2:112, 3:114, 4:116, 5:122, 6:118,
                          7:112, 8:108, 9:104, 10:100, 11:96, 12:105},
    },
    "Jowar": {
        "harvest_months": [10, 11], "peak_month": 3, "peak_uplift_pct": 15,
        "monthly_index": {1:110, 2:112, 3:115, 4:112, 5:108, 6:104,
                          7:102, 8:100, 9:100, 10:100, 11:96, 12:105},
    },
    "Bajra": {
        "harvest_months": [10, 11], "peak_month": 4, "peak_uplift_pct": 16,
        "monthly_index": {1:110, 2:112, 3:114, 4:116, 5:112, 6:108,
                          7:104, 8:101, 9:100, 10:100, 11:96, 12:106},
    },
    "Ragi": {
        "harvest_months": [11, 12], "peak_month": 5, "peak_uplift_pct": 14,
        "monthly_index": {1:108, 2:110, 3:112, 4:114, 5:114, 6:110,
                          7:106, 8:104, 9:102, 10:100, 11:98, 12:100},
    },
    "Barley": {
        "harvest_months": [3, 4], "peak_month": 10, "peak_uplift_pct": 17,
        "monthly_index": {1:106, 2:103, 3:100, 4:97, 5:99, 6:102,
                          7:106, 8:108, 9:112, 10:117, 11:115, 12:110},
    },
    # ── Pulses ─────────────────────────────────────────────────────────────────
    "Arhar/Tur": {
        "harvest_months": [1, 2], "peak_month": 7, "peak_uplift_pct": 25,
        "monthly_index": {1:100, 2:98, 3:100, 4:104, 5:110, 6:118,
                          7:125, 8:122, 9:116, 10:110, 11:106, 12:102},
    },
    "Moong(Green Gram)": {
        "harvest_months": [10, 11], "peak_month": 6, "peak_uplift_pct": 20,
        "monthly_index": {1:112, 2:114, 3:116, 4:118, 5:118, 6:120,
                          7:116, 8:112, 9:106, 10:100, 11:96, 12:104},
    },
    "Urad": {
        "harvest_months": [10, 11], "peak_month": 6, "peak_uplift_pct": 22,
        "monthly_index": {1:112, 2:114, 3:116, 4:118, 5:120, 6:122,
                          7:118, 8:112, 9:106, 10:100, 11:96, 12:104},
    },
    "Gram": {
        "harvest_months": [3, 4], "peak_month": 8, "peak_uplift_pct": 22,
        "monthly_index": {1:106, 2:103, 3:100, 4:96, 5:99, 6:108,
                          7:116, 8:122, 9:118, 10:114, 11:110, 12:108},
    },
    "Masoor": {
        "harvest_months": [3, 4], "peak_month": 9, "peak_uplift_pct": 18,
        "monthly_index": {1:106, 2:103, 3:100, 4:97, 5:100, 6:106,
                          7:112, 8:116, 9:118, 10:114, 11:110, 12:108},
    },
    "Cowpea": {
        "harvest_months": [10, 11], "peak_month": 5, "peak_uplift_pct": 16,
        "monthly_index": {1:108, 2:110, 3:112, 4:114, 5:116, 6:114,
                          7:110, 8:106, 9:103, 10:100, 11:97, 12:104},
    },
    # ── Oilseeds ───────────────────────────────────────────────────────────────
    "Groundnut": {
        "harvest_months": [10, 11], "peak_month": 4, "peak_uplift_pct": 18,
        "monthly_index": {1:108, 2:110, 3:112, 4:118, 5:116, 6:112,
                          7:108, 8:104, 9:101, 10:100, 11:97, 12:104},
    },
    "Rapeseed &Mustard": {
        "harvest_months": [2, 3], "peak_month": 10, "peak_uplift_pct": 20,
        "monthly_index": {1:104, 2:101, 3:100, 4:98, 5:100, 6:104,
                          7:108, 8:112, 9:116, 10:120, 11:118, 12:112},
    },
    "Sunflower": {
        "harvest_months": [3, 4], "peak_month": 10, "peak_uplift_pct": 18,
        "monthly_index": {1:106, 2:103, 3:100, 4:97, 5:99, 6:104,
                          7:108, 8:112, 9:116, 10:118, 11:115, 12:110},
    },
    "Soyabean": {
        "harvest_months": [10, 11], "peak_month": 5, "peak_uplift_pct": 20,
        "monthly_index": {1:108, 2:112, 3:114, 4:116, 5:120, 6:118,
                          7:114, 8:108, 9:104, 10:100, 11:96, 12:104},
    },
    "Sesamum": {
        "harvest_months": [10, 11], "peak_month": 4, "peak_uplift_pct": 16,
        "monthly_index": {1:108, 2:110, 3:112, 4:116, 5:114, 6:110,
                          7:106, 8:103, 9:101, 10:100, 11:97, 12:104},
    },
    "Linseed": {
        "harvest_months": [3, 4], "peak_month": 10, "peak_uplift_pct": 15,
        "monthly_index": {1:106, 2:103, 3:100, 4:97, 5:100, 6:104,
                          7:108, 8:110, 9:113, 10:115, 11:112, 12:108},
    },
    "Safflower": {
        "harvest_months": [3, 4], "peak_month": 10, "peak_uplift_pct": 15,
        "monthly_index": {1:106, 2:103, 3:100, 4:97, 5:100, 6:104,
                          7:108, 8:110, 9:113, 10:115, 11:112, 12:108},
    },
    "Castor Seed": {
        "harvest_months": [1, 2], "peak_month": 8, "peak_uplift_pct": 18,
        "monthly_index": {1:100, 2:98, 3:100, 4:104, 5:108, 6:112,
                          7:116, 8:118, 9:115, 10:110, 11:106, 12:102},
    },
    # ── Commercial ─────────────────────────────────────────────────────────────
    "Cotton(lint)": {
        "harvest_months": [10, 11], "peak_month": 2, "peak_uplift_pct": 20,
        "monthly_index": {1:112, 2:120, 3:116, 4:110, 5:106, 6:102,
                          7:101, 8:100, 9:100, 10:100, 11:97, 12:108},
    },
    "Sugarcane": {
        "harvest_months": [11, 12, 1], "peak_month": 0, "peak_uplift_pct": 0,
        "monthly_index": {1:100, 2:100, 3:100, 4:100, 5:100, 6:100,
                          7:100, 8:100, 9:100, 10:100, 11:100, 12:100},
        "note": "Fixed FRP — sell directly to sugar mill. No seasonal variation.",
    },
    "Tobacco": {
        "harvest_months": [2, 3], "peak_month": 7, "peak_uplift_pct": 12,
        "monthly_index": {1:104, 2:100, 3:98, 4:100, 5:104, 6:108,
                          7:112, 8:110, 9:108, 10:106, 11:105, 12:104},
    },
    "Coconut": {
        "harvest_months": [12, 1, 2], "peak_month": 7, "peak_uplift_pct": 15,
        "monthly_index": {1:100, 2:98, 3:100, 4:104, 5:108, 6:112,
                          7:115, 8:113, 9:110, 10:108, 11:104, 12:100},
    },
    "Arecanut": {
        "harvest_months": [11, 12], "peak_month": 6, "peak_uplift_pct": 20,
        "monthly_index": {1:108, 2:110, 3:112, 4:116, 5:118, 6:120,
                          7:116, 8:112, 9:108, 10:104, 11:100, 12:97},
    },
    "Coffee": {
        "harvest_months": [11, 12, 1], "peak_month": 8, "peak_uplift_pct": 14,
        "monthly_index": {1:100, 2:100, 3:102, 4:106, 5:108, 6:110,
                          7:112, 8:114, 9:112, 10:110, 11:106, 12:100},
    },
    "Black pepper": {
        "harvest_months": [7, 8], "peak_month": 3, "peak_uplift_pct": 18,
        "monthly_index": {1:108, 2:110, 3:118, 4:116, 5:112, 6:108,
                          7:100, 8:100, 9:102, 10:104, 11:106, 12:108},
    },
    # ── Vegetables (perishable — sell quickly) ─────────────────────────────────
    "Potato": {
        "harvest_months": [1, 2, 3], "peak_month": 9, "peak_uplift_pct": 25,
        "monthly_index": {1:100, 2:96, 3:94, 4:98, 5:105, 6:112,
                          7:118, 8:122, 9:125, 10:118, 11:110, 12:104},
    },
    "Onion": {
        "harvest_months": [11, 12, 1], "peak_month": 8, "peak_uplift_pct": 60,
        "monthly_index": {1:100, 2:98, 3:95, 4:100, 5:120, 6:140,
                          7:155, 8:160, 9:148, 10:130, 11:110, 12:100},
        "note": "Onion prices highly volatile. Cold storage (5°C) extends shelf life to 5 months.",
    },
    "Tomato":      {"harvest_months": [11, 12], "peak_month": 0, "peak_uplift_pct": 0,
                    "note": "Highly perishable — sell within 3–5 days of harvest."},
    "Brinjal":     {"harvest_months": [10, 11], "peak_month": 0, "peak_uplift_pct": 0,
                    "note": "Perishable — sell within 5–7 days."},
    "Cabbage":     {"harvest_months": [11, 12], "peak_month": 0, "peak_uplift_pct": 0,
                    "note": "Perishable — sell within 1–2 weeks."},
    "Cauliflower": {"harvest_months": [11, 12], "peak_month": 0, "peak_uplift_pct": 0,
                    "note": "Perishable — sell within 1 week."},
    "Garlic": {
        "harvest_months": [3, 4], "peak_month": 11, "peak_uplift_pct": 35,
        "monthly_index": {1:108, 2:104, 3:100, 4:96, 5:100, 6:108,
                          7:116, 8:122, 9:128, 10:132, 11:135, 12:120},
    },
    "Ginger":     {"harvest_months": [12, 1], "peak_month": 0, "peak_uplift_pct": 0,
                   "note": "Fresh ginger — sell within 2 weeks. Dry ginger can be stored 6–8 months."},
    "Turmeric": {
        "harvest_months": [1, 2], "peak_month": 9, "peak_uplift_pct": 22,
        "monthly_index": {1:100, 2:97, 3:100, 4:104, 5:108, 6:112,
                          7:116, 8:120, 9:122, 10:118, 11:112, 12:106},
    },
    "Tapioca":     {"harvest_months": [12, 1, 2], "peak_month": 0, "peak_uplift_pct": 0,
                    "note": "Perishable — process or sell within 2–3 days of harvest."},
    "Sweet Potato":{"harvest_months": [11, 12], "peak_month": 5, "peak_uplift_pct": 14,
                    "monthly_index": {1:108, 2:110, 3:112, 4:114, 5:114, 6:112,
                                      7:108, 8:104, 9:102, 10:100, 11:98, 12:100}},
    "Bitter Gourd":{"harvest_months": [5, 6], "peak_month": 0, "peak_uplift_pct": 0,
                    "note": "Perishable — sell within 5 days."},
    "Bottle Gourd":{"harvest_months": [5, 6], "peak_month": 0, "peak_uplift_pct": 0,
                    "note": "Perishable — sell within 5 days."},
    "Pumpkin": {
        "harvest_months": [10, 11], "peak_month": 5, "peak_uplift_pct": 12,
        "monthly_index": {1:108, 2:110, 3:112, 4:112, 5:112, 6:110,
                          7:108, 8:104, 9:102, 10:100, 11:98, 12:104},
    },
    # ── Fruits ─────────────────────────────────────────────────────────────────
    "Banana":  {"harvest_months": [3, 4, 5], "peak_month": 0, "peak_uplift_pct": 0,
                "note": "Perishable — sell within 5–7 days of harvest. Ripens quickly post-harvest."},
    "Mango": {
        "harvest_months": [4, 5, 6], "peak_month": 3, "peak_uplift_pct": 20,
        "monthly_index": {1:112, 2:116, 3:120, 4:108, 5:100, 6:96,
                          7:100, 8:104, 9:106, 10:108, 11:110, 12:112},
        "note": "Best prices in early season (March) before main crop arrives. Sell early variety first.",
    },
    "Grapes": {
        "harvest_months": [2, 3], "peak_month": 12, "peak_uplift_pct": 18,
        "monthly_index": {1:112, 2:100, 3:96, 4:100, 5:104, 6:108,
                          7:110, 8:112, 9:114, 10:116, 11:116, 12:118},
    },
    "Papaya":  {"harvest_months": [10, 11], "peak_month": 0, "peak_uplift_pct": 0,
                "note": "Perishable — sell within 5–7 days."},
    "Orange Fruit": {
        "harvest_months": [11, 12], "peak_month": 7, "peak_uplift_pct": 16,
        "monthly_index": {1:104, 2:100, 3:98, 4:102, 5:106, 6:110,
                          7:116, 8:114, 9:112, 10:108, 11:100, 12:98},
    },
    "Pineapple": {"harvest_months": [4, 5], "peak_month": 0, "peak_uplift_pct": 0,
                  "note": "Perishable — sell within 3–5 days."},
    "Guava": {
        "harvest_months": [10, 11], "peak_month": 5, "peak_uplift_pct": 14,
        "monthly_index": {1:108, 2:110, 3:112, 4:114, 5:114, 6:112,
                          7:108, 8:104, 9:102, 10:100, 11:98, 12:104},
    },
    "Lemon": {
        "harvest_months": [3, 4], "peak_month": 10, "peak_uplift_pct": 18,
        "monthly_index": {1:108, 2:104, 3:100, 4:97, 5:100, 6:106,
                          7:110, 8:112, 9:115, 10:118, 11:115, 12:110},
    },
    "Apple": {
        "harvest_months": [8, 9], "peak_month": 3, "peak_uplift_pct": 22,
        "monthly_index": {1:108, 2:110, 3:122, 4:118, 5:112, 6:108,
                          7:104, 8:100, 9:100, 10:104, 11:106, 12:108},
    },
}

# Crops where waiting to sell is not viable (perishable / fixed-price)
PERISHABLE_CROPS = {
    "Tomato", "Brinjal", "Cabbage", "Cauliflower", "Bitter Gourd",
    "Bottle Gourd", "Banana", "Papaya", "Pineapple", "Tapioca", "Ginger"
}

# Crops sold directly to processor — channel is fixed
PROCESSOR_CROPS = {
    "Sugarcane":   "Local sugar mill (cooperative or private) — brings crop directly; mill sets FRP price",
    "Cotton(lint)":"CCI (Cotton Corporation of India) ginning centre in your district",
    "Arecanut":    "Local arecanut processing cooperative or private trader",
    "Coffee":      "Coffee Board curing works or registered estate buyer",
    "Rubber":      "Rubber Board processing centre or RSS dealer",
    "Tobacco":     "Tobacco Board registered auction platform or local buyer",
}

# States with strong/active MSP procurement infrastructure
STRONG_PROCUREMENT_STATES = {
    "punjab", "haryana", "chhattisgarh", "odisha",
    "madhya pradesh", "telangana", "andhra pradesh",
    "west bengal", "tamil nadu", "karnataka", "kerala"
}

# MSP-covered crops (central government)
MSP_CROPS = {
    "Rice", "Wheat", "Maize", "Jowar", "Bajra", "Ragi", "Barley",
    "Arhar/Tur", "Moong(Green Gram)", "Urad", "Gram", "Masoor",
    "Groundnut", "Rapeseed &Mustard", "Sunflower", "Soyabean",
    "Sesamum", "Linseed", "Safflower", "Cotton(lint)", "Sugarcane"
}

# Storage parameters: cost_per_month_per_quintal (Rs), max shelf life (months)
# Source: WDRA warehousing rates + ICAR post-harvest guidelines
STORAGE_PARAMS = {
    "Rice":               {"cost_per_month": 8,  "shelf_months": 12, "type": "Dry warehouse / gunny bags, moisture < 14%"},
    "Wheat":              {"cost_per_month": 7,  "shelf_months": 15, "type": "Dry silo or gunny bags, moisture < 12%"},
    "Maize":              {"cost_per_month": 8,  "shelf_months": 9,  "type": "Hermetic bags / metal silo, moisture < 13%"},
    "Jowar":              {"cost_per_month": 6,  "shelf_months": 10, "type": "Dry gunny bags, cool ventilated store"},
    "Bajra":              {"cost_per_month": 6,  "shelf_months": 8,  "type": "Dry gunny bags — sensitive to moisture"},
    "Ragi":               {"cost_per_month": 5,  "shelf_months": 12, "type": "Dry gunny bags — naturally pest resistant"},
    "Barley":             {"cost_per_month": 6,  "shelf_months": 12, "type": "Dry warehouse, moisture < 12%"},
    "Arhar/Tur":          {"cost_per_month": 7,  "shelf_months": 12, "type": "Dry hermetic bags, moisture < 10%"},
    "Moong(Green Gram)":  {"cost_per_month": 7,  "shelf_months": 9,  "type": "Hermetic storage — prone to weevils"},
    "Urad":               {"cost_per_month": 7,  "shelf_months": 9,  "type": "Hermetic storage — prone to weevils"},
    "Gram":               {"cost_per_month": 6,  "shelf_months": 14, "type": "Cool dry storage, hermetic bags"},
    "Masoor":             {"cost_per_month": 6,  "shelf_months": 12, "type": "Dry hermetic bags"},
    "Cowpea":             {"cost_per_month": 6,  "shelf_months": 8,  "type": "Hermetic bags — highly susceptible to weevils"},
    "Groundnut":          {"cost_per_month": 8,  "shelf_months": 6,  "type": "Dry ventilated store — prone to aflatoxin if damp"},
    "Rapeseed &Mustard":  {"cost_per_month": 6,  "shelf_months": 10, "type": "Dry hermetic bags, moisture < 8%"},
    "Sunflower":          {"cost_per_month": 7,  "shelf_months": 8,  "type": "Dry store, moisture < 9%"},
    "Soyabean":           {"cost_per_month": 7,  "shelf_months": 8,  "type": "Cool dry storage, moisture < 12%"},
    "Sesamum":            {"cost_per_month": 6,  "shelf_months": 10, "type": "Dry hermetic bags, moisture < 6%"},
    "Linseed":            {"cost_per_month": 5,  "shelf_months": 12, "type": "Dry sealed bags"},
    "Safflower":          {"cost_per_month": 5,  "shelf_months": 12, "type": "Dry sealed bags"},
    "Castor Seed":        {"cost_per_month": 5,  "shelf_months": 12, "type": "Dry ventilated store"},
    "Cotton(lint)":       {"cost_per_month": 4,  "shelf_months": 18, "type": "Dry covered storage in bale form"},
    "Sugarcane":          {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — send to mill within 24 hrs"},
    "Tobacco":            {"cost_per_month": 6,  "shelf_months": 24, "type": "Cured leaf — dry ventilated shed"},
    "Coconut":            {"cost_per_month": 5,  "shelf_months": 3,  "type": "Cool dry place — dehusked coconut only"},
    "Arecanut":           {"cost_per_month": 5,  "shelf_months": 18, "type": "Dry ventilated store, dried form"},
    "Coffee":             {"cost_per_month": 6,  "shelf_months": 12, "type": "Dry hermetic bags after curing"},
    "Black pepper":       {"cost_per_month": 7,  "shelf_months": 24, "type": "Dry hermetic bags after drying"},
    "Potato":             {"cost_per_month": 18, "shelf_months": 5,  "type": "Cold storage (4–8°C) — essential"},
    "Onion":              {"cost_per_month": 12, "shelf_months": 5,  "type": "Well-ventilated cool store or cold storage"},
    "Tomato":             {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — sell within 3–5 days"},
    "Brinjal":            {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — sell within 5–7 days"},
    "Cabbage":            {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — sell within 1–2 weeks"},
    "Cauliflower":        {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — sell within 1 week"},
    "Garlic":             {"cost_per_month": 8,  "shelf_months": 6,  "type": "Cool dry ventilated store"},
    "Ginger":             {"cost_per_month": 999,"shelf_months": 0,  "type": "Fresh ginger — sell within 2 weeks"},
    "Turmeric":           {"cost_per_month": 6,  "shelf_months": 12, "type": "Dry boiled/dried turmeric in gunny bags"},
    "Tapioca":            {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — process within 2–3 days"},
    "Sweet Potato":       {"cost_per_month": 8,  "shelf_months": 2,  "type": "Cool dry store, max 2 months"},
    "Bitter Gourd":       {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — sell within 5 days"},
    "Bottle Gourd":       {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — sell within 5 days"},
    "Pumpkin":            {"cost_per_month": 5,  "shelf_months": 4,  "type": "Cool dry store, whole fruit stores well"},
    "Banana":             {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — sell within 5–7 days"},
    "Mango":              {"cost_per_month": 999,"shelf_months": 0,  "type": "Perishable — cold chain needed for >1 week"},
    "Grapes":             {"cost_per_month": 999,"shelf_months": 0,  "type": "Perishable — cold chain or sell within 5 days"},
    "Papaya":             {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — sell within 5 days"},
    "Orange Fruit":       {"cost_per_month": 8,  "shelf_months": 2,  "type": "Cool store, max 2 months"},
    "Pineapple":          {"cost_per_month": 999,"shelf_months": 0,  "type": "Cannot store — sell within 3–5 days"},
    "Guava":              {"cost_per_month": 999,"shelf_months": 0,  "type": "Perishable — sell within 5–7 days"},
    "Lemon":              {"cost_per_month": 6,  "shelf_months": 2,  "type": "Cool dry store, up to 2 months"},
    "Apple":              {"cost_per_month": 10, "shelf_months": 6,  "type": "Cold storage (0–4°C) essential for long storage"},
}

MONTH_NAMES = {
    1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June",
    7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"
}


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-MODULE 1: SEASONAL ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_seasonal(crop: str, harvest_season: str,
                     current_price_per_kg: float) -> dict:
    """
    Returns best selling month, projected price, and storage recommendation
    based on historical price seasonality for the crop.
    """
    data = PRICE_SEASONALITY.get(crop)

    # Perishable — no waiting possible
    if crop in PERISHABLE_CROPS or data is None:
        return {
            "recommendation":  "SELL_IMMEDIATELY",
            "best_month":      "immediately",
            "peak_uplift_pct": 0,
            "current_price":   current_price_per_kg,
            "projected_price": current_price_per_kg,
            "months_to_wait":  0,
            "note": data.get("note", "Perishable crop — sell as soon as possible.")
                    if data else "No seasonality data. Sell immediately.",
        }

    # Fixed price crop (Sugarcane)
    if data.get("peak_uplift_pct", 0) == 0:
        return {
            "recommendation":  "SELL_IMMEDIATELY",
            "best_month":      "immediately",
            "peak_uplift_pct": 0,
            "current_price":   current_price_per_kg,
            "projected_price": current_price_per_kg,
            "months_to_wait":  0,
            "note": data.get("note", "Fixed price crop. Sell when ready."),
        }

    peak_month    = data["peak_month"]
    uplift_pct    = data["peak_uplift_pct"]
    harvest_month = data["harvest_months"][0]   # primary harvest month

    # Calculate months to wait from harvest
    months_to_wait = (peak_month - harvest_month) % 12
    if months_to_wait == 0:
        months_to_wait = 12

    projected_price = round(current_price_per_kg * (1 + uplift_pct / 100), 2)

    return {
        "recommendation":  "WAIT" if uplift_pct >= 10 else "SELL_SOON",
        "best_month":      MONTH_NAMES[peak_month],
        "peak_uplift_pct": uplift_pct,
        "current_price":   current_price_per_kg,
        "projected_price": projected_price,
        "months_to_wait":  months_to_wait,
        "note": data.get("note", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-MODULE 2: STORAGE ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_storage(crop: str, months_to_wait: int,
                    current_price_per_kg: float,
                    projected_price_per_kg: float) -> dict:
    """
    Returns whether waiting is financially viable after accounting for storage cost.
    All price calculations in Rs/quintal (1 quintal = 100 kg).
    """
    params = STORAGE_PARAMS.get(crop, {"cost_per_month": 10, "shelf_months": 6,
                                        "type": "Dry storage"})

    shelf_months       = params["shelf_months"]
    cost_per_month     = params["cost_per_month"]
    storage_type       = params["type"]

    # Instantly not viable
    if shelf_months == 0 or cost_per_month == 999:
        return {
            "viable":           False,
            "reason":           "Perishable — storage not possible",
            "storage_type":     storage_type,
            "storage_cost":     0,
            "price_gain":       0,
            "net_gain_quintal": 0,
            "shelf_months":     0,
        }

    # Can farmer actually wait long enough?
    actual_wait = min(months_to_wait, shelf_months)
    total_storage_cost = cost_per_month * actual_wait   # Rs/quintal

    # Price gain in Rs/quintal
    price_gain = (projected_price_per_kg - current_price_per_kg) * 100

    net_gain = round(price_gain - total_storage_cost, 2)
    viable   = (net_gain > 0) and (actual_wait <= shelf_months)

    return {
        "viable":           viable,
        "reason":           "Profitable to wait" if viable else
                            ("Storage cost exceeds price gain"
                             if price_gain < total_storage_cost
                             else "Shelf life too short"),
        "storage_type":     storage_type,
        "storage_cost":     total_storage_cost,    # Rs/quintal
        "price_gain":       round(price_gain, 2),  # Rs/quintal
        "net_gain_quintal": net_gain,               # Rs/quintal after storage
        "shelf_months":     shelf_months,
        "actual_wait":      actual_wait,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-MODULE 3: WEATHER SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

GRAIN_CROPS_SET = {
    "Rice", "Wheat", "Maize", "Jowar", "Bajra", "Ragi", "Barley",
    "Arhar/Tur", "Moong(Green Gram)", "Urad", "Gram", "Masoor",
    "Groundnut", "Soyabean", "Sesamum"
}

VEGETABLE_FRUIT_SET = {
    "Tomato", "Brinjal", "Cabbage", "Cauliflower", "Bitter Gourd",
    "Bottle Gourd", "Banana", "Papaya", "Pineapple", "Guava",
    "Onion", "Mango", "Grapes", "Tapioca", "Ginger"
}


def fetch_weather_forecast(lat: float, lon: float) -> Optional[dict]:
    """
    Fetches 7-day daily forecast from Open-Meteo (free, no API key required).
    Returns dict with daily precipitation_sum and precipitation_probability_max,
    or None if the request fails.
    """
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "daily": [
                "precipitation_sum",
                "precipitation_probability_max",
                "temperature_2m_max",
                "windspeed_10m_max",
            ],
            "forecast_days": 7,
            "timezone":      "Asia/Kolkata",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Weather API failed: {e}")
        return None


def analyse_weather_signal(forecast: dict, crop: str) -> dict:
    """
    Analyses 7-day forecast and returns one of 5 signal types:
      PERISHABLE_CRASH    — heavy rain → rush-sell perishables before glut
      SURGE_OPPORTUNITY   — rain damages standing crops → stored stock becomes scarcer
      TRANSPORT_DISRUPTION— very heavy rain → roads flood → local price spike
      STORAGE_RISK        — sustained rain → humidity → grain quality degradation
      NORMAL              — no significant weather event
    """
    if not forecast or "daily" not in forecast:
        return {
            "signal_type": "NORMAL",
            "urgency":     "LOW",
            "headline":    "Weather data unavailable — following seasonal advice",
            "detail":      "Could not fetch weather forecast. Proceed with seasonal recommendation.",
            "action":      "FOLLOW_SEASONAL_ADVICE",
        }

    daily        = forecast["daily"]
    rain_7day    = daily.get("precipitation_sum", [0]*7)
    rain_7day    = [r if r else 0 for r in rain_7day]   # replace None with 0

    total_rain   = sum(rain_7day)
    rain_2day    = sum(rain_7day[:2])
    heavy_days   = sum(1 for r in rain_7day if r > 20)  # >20mm/day = heavy

    # Signal 1: Perishable crash
    # Heavy rain → all perishable farmers rush to mandi → supply glut → prices crash
    if crop in VEGETABLE_FRUIT_SET and total_rain > 40:
        return {
            "signal_type": "PERISHABLE_CRASH",
            "urgency":     "HIGH",
            "headline":    f"URGENT: Sell {crop} immediately — rain will crash mandi prices",
            "detail":      (
                f"{total_rain:.0f}mm of rain forecast over 7 days. Heavy rain causes all "
                f"perishable farmers to rush to the mandi simultaneously, flooding supply "
                f"and crashing prices by 30–50%. Sell today or tomorrow before the glut hits. "
                f"Even a 10% discount today is better than a 40% crash in 3 days."
            ),
            "action":      "SELL_IMMEDIATELY",
        }

    # Signal 2: Supply shock / surge opportunity for stored grain
    # Rain damages other farmers' standing crops → regional supply falls → your stored stock more valuable
    if crop in GRAIN_CROPS_SET and heavy_days >= 3 and rain_2day > 25:
        return {
            "signal_type": "SURGE_OPPORTUNITY",
            "urgency":     "MEDIUM",
            "headline":    "Rain damaging standing crops nearby — prices likely to rise",
            "detail":      (
                f"Heavy rainfall ({rain_2day:.0f}mm) forecast in the next 2 days with "
                f"{heavy_days} heavy-rain days this week. This will damage unharvested crops "
                f"in surrounding areas, reducing regional supply. If your crop is already "
                f"safely harvested and stored, consider waiting 1–2 extra weeks — "
                f"post-rain supply shocks typically push prices up 10–20% above normal."
            ),
            "action":      "WAIT_SHORT_TERM",
        }

    # Signal 3: Transport disruption — local mandi price spike
    # Flooded roads → fewer trucks reach mandi → local buyers compete → short spike
    if rain_2day > 50:
        return {
            "signal_type": "TRANSPORT_DISRUPTION",
            "urgency":     "MEDIUM",
            "headline":    "Heavy rain may block roads — local mandi prices will spike",
            "detail":      (
                f"Very heavy rain ({rain_2day:.0f}mm) expected in the next 2 days. "
                f"Road flooding typically cuts mandi arrivals by 40–60% for 3–7 days, "
                f"pushing local prices up 8–15% as buyers compete for available stock. "
                f"If your crop is safely stored and accessible, selling during this window "
                f"(before roads clear) can give a short-term price advantage."
            ),
            "action":      "SELL_THIS_WEEK",
        }

    # Signal 4: Storage risk — sustained rain raises humidity, damages grain
    # Grain absorbs moisture → fungal growth → grade downgrading at mandi
    if crop in GRAIN_CROPS_SET and total_rain > 70:
        return {
            "signal_type": "STORAGE_RISK",
            "urgency":     "HIGH",
            "headline":    "Storage risk — sustained rain will damage grain quality",
            "detail":      (
                f"{total_rain:.0f}mm of sustained rainfall forecast over 7 days. "
                f"This raises ambient humidity significantly, causing grain stored in "
                f"non-hermetic bags to absorb moisture above safe levels (>14% for rice, "
                f">12% for wheat). Fungal growth and grade downgrading at the mandi can "
                f"reduce effective price by 15–25%. Either sell now or immediately "
                f"transfer to hermetic/sealed storage."
            ),
            "action":      "SELL_OR_SEAL_STORAGE",
        }

    # Normal conditions
    return {
        "signal_type": "NORMAL",
        "urgency":     "LOW",
        "headline":    f"No significant weather event forecast (total rain: {total_rain:.0f}mm/week)",
        "detail":      "Weather conditions are normal for the next 7 days. Follow seasonal price recommendation.",
        "action":      "FOLLOW_SEASONAL_ADVICE",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-MODULE 4: CHANNEL SELECTOR (LOCAL ONLY)
# ═══════════════════════════════════════════════════════════════════════════════

def get_local_channel(crop: str, district: str, state: str) -> dict:
    """
    Returns the best local selling channel for the crop.
    All channels are district-level or within ~50km — no cross-state recommendations.
    Priority: dedicated processor > MSP procurement > e-NAM > local APMC
    """
    state_lower = state.lower().strip()

    # Priority 1: crop has a dedicated processor co-located in producing districts
    if crop in PROCESSOR_CROPS:
        return {
            "channel":  PROCESSOR_CROPS[crop],
            "type":     "dedicated_processor",
            "note":     "This processor is located within your district or nearby tehsil. "
                        "No long-distance travel needed.",
        }

    # Priority 2: MSP-covered crop + state has active procurement
    if crop in MSP_CROPS and state_lower in STRONG_PROCUREMENT_STATES:
        return {
            "channel":  "FCI / State Civil Supplies Corporation procurement centre",
            "type":     "msp_procurement",
            "note":     (
                f"{state.title()} has active MSP procurement. Register your crop at the nearest "
                f"Primary Agricultural Credit Society (PACS) or FCI procurement centre. "
                f"Bring: Aadhaar card, bank passbook, land records."
            ),
        }

    # Priority 3: e-NAM availability — check by known e-NAM district list
    # (In production this reads from data/enam_mandis.csv)
    # For now: fallback to a known-active state list
    ENAM_ACTIVE_STATES = {
        "uttar pradesh", "madhya pradesh", "rajasthan", "haryana",
        "maharashtra", "telangana", "andhra pradesh", "gujarat",
        "karnataka", "himachal pradesh", "uttarakhand", "odisha",
        "punjab", "chhattisgarh", "tamil nadu"
    }
    if state_lower in ENAM_ACTIVE_STATES:
        return {
            "channel":  f"e-NAM mandi — {district.title()} APMC (if registered) or nearest e-NAM mandi",
            "type":     "enam",
            "note":     (
                "e-NAM is available in your state. It uses the same physical APMC mandi "
                "but online competitive bidding typically gives 3–8% better prices. "
                "Register free at enam.gov.in with Aadhaar and bank account."
            ),
        }

    # Priority 4: Default — local APMC mandi
    return {
        "channel":  f"Local APMC mandi — {district.title()} or nearest district mandi",
        "type":     "apmc",
        "note":     "Sell at the nearest APMC mandi. Arrive early morning for best prices. "
                    "Check Agmarknet (agmarknet.gov.in) for today's mandi prices before travelling.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ADVISORY ENGINE — combines all 3 sub-modules
# ═══════════════════════════════════════════════════════════════════════════════

def build_post_harvest_advisory(
    crop: str,
    district: str,
    state: str,
    season: str,
    current_price_per_kg: float,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> dict:
    """
    Runs all 3 sub-modules and applies the weather override logic.
    Returns a structured advisory dict ready for Groq narration.
    """
    # Sub-module 1: seasonal
    seasonal = analyse_seasonal(crop, season, current_price_per_kg)

    # Sub-module 2: storage
    storage = analyse_storage(
        crop,
        seasonal["months_to_wait"],
        current_price_per_kg,
        seasonal["projected_price"],
    )

    # Sub-module 3: weather
    weather_raw = None
    if lat and lon:
        weather_raw = fetch_weather_forecast(lat, lon)
    weather = analyse_weather_signal(weather_raw, crop)

    # Sub-module 4: channel
    channel = get_local_channel(crop, district, state)

    # ── Override logic ────────────────────────────────────────────────────────
    # Weather HIGH urgency always overrides seasonal
    if weather["urgency"] == "HIGH":
        final_action      = weather["action"]
        final_sell_month  = "immediately" if "IMMEDIATE" in weather["action"] else seasonal["best_month"]
        override_applied  = True

    # Weather MEDIUM + seasonal says WAIT → evaluate
    elif weather["urgency"] == "MEDIUM" and seasonal["recommendation"] == "WAIT":
        if weather["action"] == "SELL_IMMEDIATELY":
            # Weather says sell now despite seasonal saying wait
            final_action     = "SELL_IMMEDIATELY"
            final_sell_month = "immediately"
            override_applied = True
        else:
            # Weather confirms waiting (SURGE or DISRUPTION supports the wait strategy)
            final_action     = "WAIT"
            final_sell_month = seasonal["best_month"]
            override_applied = False

    # No significant weather → follow seasonal + storage
    else:
        if not storage["viable"]:
            final_action     = "SELL_IMMEDIATELY"
            final_sell_month = "immediately"
        else:
            final_action     = seasonal["recommendation"]
            final_sell_month = seasonal["best_month"]
        override_applied = False

    return {
        # Final decision
        "final_action":        final_action,
        "sell_month":          final_sell_month,
        "weather_override":    override_applied,

        # Seasonal data
        "best_month":          seasonal["best_month"],
        "peak_uplift_pct":     seasonal["peak_uplift_pct"],
        "current_price":       current_price_per_kg,
        "projected_price":     seasonal["projected_price"],
        "months_to_wait":      seasonal["months_to_wait"],

        # Storage data
        "storage_viable":      storage["viable"],
        "storage_type":        storage["storage_type"],
        "storage_cost":        storage["storage_cost"],
        "net_gain_quintal":    storage["net_gain_quintal"],
        "shelf_months":        storage["shelf_months"],

        # Weather data
        "weather_signal":      weather["signal_type"],
        "weather_urgency":     weather["urgency"],
        "weather_headline":    weather["headline"],
        "weather_detail":      weather["detail"],

        # Channel data
        "channel":             channel["channel"],
        "channel_type":        channel["type"],
        "channel_note":        channel["note"],

        # Raw modules for Groq prompt
        "_seasonal":  seasonal,
        "_storage":   storage,
        "_weather":   weather,
        "_channel":   channel,
    }
