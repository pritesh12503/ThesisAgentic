"""
config.py — UPDATED for new yield dataset (crop_production_india.csv)

DATASET CHANGE:
  Old: Custom_Crops_yield_Historical_Dataset.csv  (4 crops only)
  New: crop_production_india.csv                  (124 crops, 646 districts)
  Source: https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
  Columns: State_Name, District_Name, Crop_Year, Season, Crop, Area, Production
  NOTE: Yield column ABSENT — computed as (Production/Area)*1000 during loading
"""

import os
from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

AGRONOMIC_CSV = DATA_DIR / "thesis_agronomic_dataset_clean.csv"
YIELD_CSV     = DATA_DIR / "crop_production_india.csv"          # NEW
PRICE_CSV     = DATA_DIR / "Price_Agriculture_commodities_Week.csv"

GROQ_MODEL = "llama-3.1-8b-instant"

# New yield dataset column names
YIELD_COL_STATE    = "State_Name"
YIELD_COL_DISTRICT = "District_Name"
YIELD_COL_YEAR     = "Crop_Year"
YIELD_COL_SEASON   = "Season"
YIELD_COL_CROP     = "Crop"
YIELD_COL_AREA     = "Area"
YIELD_COL_PROD     = "Production"
YIELD_COL_YIELD    = "Yield_kg_per_ha"   # computed column

KAGGLE_SEASONS = ["Kharif", "Rabi", "Whole Year", "Summer", "Autumn", "Winter"]
UI_SEASONS     = ["Kharif", "Rabi", "Zaid", "Annual"]

AGRO_FEATURES = [
    "Nitrogen Value", "Phosphorous value", "Potassium value",
    "pHsoil", "Temperature_C", "Humidity_%", "Rainfall_mm",
]

AGRO_FEATURE_LABELS = {
    "Nitrogen Value":    "Nitrogen (N)",
    "Phosphorous value": "Phosphorus (P)",
    "Potassium value":   "Potassium (K)",
    "pHsoil":            "Soil pH",
    "Temperature_C":     "Temperature (°C)",
    "Humidity_%":        "Humidity (%)",
    "Rainfall_mm":       "Rainfall (mm)",
}

DEFAULT_W1 = 0.5
DEFAULT_W2 = 0.5

# Crop names EXACTLY as they appear in crop_production_india.csv
# mapped to commodity names in Price_Agriculture_commodities_Week.csv
CROP_TO_COMMODITY = {
    # Cereals
    "Rice":               ["Rice", "Paddy(Dhan)(Common)", "Paddy(Dhan)(Basmati)"],
    "Wheat":              ["Wheat"],
    "Maize":              ["Maize"],
    "Jowar":              ["Jowar(Sorghum)"],
    "Bajra":              ["Bajra(Pearl Millet/Cumbu)"],
    "Ragi":               ["Ragi (Finger Millet)"],
    "Barley":             ["Barley (Jau)"],
    # Pulses
    "Arhar/Tur":          ["Arhar (Tur/Red Gram)(Whole)", "Arhar Dal(Tur Dal)"],
    "Moong(Green Gram)":  ["Green Gram (Moong)(Whole)", "Green Gram Dal (Moong Dal)"],
    "Urad":               ["Black Gram (Urd Beans)(Whole)", "Black Gram Dal (Urd Dal)"],
    "Gram":               ["Bengal Gram(Gram)(Whole)", "Bengal Gram Dal (Chana Dal)", "Kabuli Chana(Chickpeas-White)"],
    "Masoor":             ["Lentil (Masur)(Whole)", "Masur Dal"],
    "Cowpea":             ["Cowpea (Lobia/Karamani)"],
    # Oilseeds
    "Groundnut":          ["Groundnut", "Ground Nut Seed", "Groundnut pods (raw)"],
    "Rapeseed &Mustard":  ["Mustard"],
    "Sunflower":          ["Sunflower"],
    "Soyabean":           ["Soyabean"],
    "Sesamum":            ["Sesamum(Sesame,Gingelly,Til)"],
    "Linseed":            ["Linseed"],
    "Safflower":          ["Safflower"],
    "Castor Seed":        ["Castor Seed"],
    # Commercial
    "Cotton(lint)":       ["Cotton"],
    "Sugarcane":          ["Gur(Jaggery)", "Sugar"],
    "Tobacco":            ["Tobacco"],
    "Coconut":            ["Coconut", "Coconut Seed", "Tender Coconut"],
    "Arecanut":           ["Arecanut(Betelnut/Supari)"],
    "Coffee":             ["Coffee"],
    "Black pepper":       ["Black pepper", "Pepper ungarbled"],
    # Vegetables
    "Potato":             ["Potato"],
    "Onion":              ["Onion", "Onion Green"],
    "Tomato":             ["Tomato"],
    "Brinjal":            ["Brinjal"],
    "Cabbage":            ["Cabbage"],
    "Cauliflower":        ["Cauliflower"],
    "Garlic":             ["Garlic"],
    "Ginger":             ["Ginger(Green)", "Ginger(Dry)"],
    "Turmeric":           ["Turmeric", "Turmeric (raw)"],
    "Tapioca":            ["Tapioca"],
    "Sweet Potato":       ["Sweet Potato"],
    "Bitter Gourd":       ["Bitter gourd"],
    "Bottle Gourd":       ["Bottle gourd"],
    "Pumpkin":            ["Pumpkin"],
    # Fruits
    "Banana":             ["Banana", "Banana - Green"],
    "Mango":              ["Mango", "Mango (Raw-Ripe)"],
    "Grapes":             ["Grapes"],
    "Papaya":             ["Papaya"],
    "Orange Fruit":       ["Orange"],
    "Pineapple":          ["Pineapple"],
    "Guava":              ["Guava"],
    "Lemon":              ["Lemon", "Lime"],
    "Apple":              ["Apple"],
}

SUPPORTED_CROPS = list(CROP_TO_COMMODITY.keys())

# Approximate input costs per hectare (Rs/ha) for profit calculation
INPUT_COSTS = {
    "Rice": 25000, "Wheat": 22000, "Maize": 20000, "Jowar": 14000,
    "Bajra": 12000, "Ragi": 13000, "Barley": 16000,
    "Arhar/Tur": 18000, "Moong(Green Gram)": 14000, "Urad": 14000,
    "Gram": 18000, "Masoor": 16000, "Cowpea": 13000,
    "Groundnut": 28000, "Rapeseed &Mustard": 18000, "Sunflower": 20000,
    "Soyabean": 18000, "Sesamum": 16000, "Linseed": 15000,
    "Safflower": 16000, "Castor Seed": 20000,
    "Cotton(lint)": 35000, "Sugarcane": 40000, "Tobacco": 45000,
    "Coconut": 25000, "Arecanut": 30000, "Coffee": 35000, "Black pepper": 40000,
    "Potato": 60000, "Onion": 45000, "Tomato": 50000, "Brinjal": 35000,
    "Cabbage": 40000, "Cauliflower": 42000, "Garlic": 55000, "Ginger": 80000,
    "Turmeric": 60000, "Tapioca": 25000, "Sweet Potato": 25000,
    "Bitter Gourd": 40000, "Bottle Gourd": 35000, "Pumpkin": 25000,
    "Banana": 50000, "Mango": 30000, "Grapes": 90000, "Papaya": 40000,
    "Orange Fruit": 35000, "Pineapple": 45000, "Guava": 30000,
    "Lemon": 28000, "Apple": 80000,
}

POLICY_KNOWLEDGE = """
NATIONAL POLICY — CENTRAL MSP AND SCHEMES (2024-25)

CROP: RICE (Paddy)
  Central MSP 2024-25: Rs 23.00/kg (Common); Rs 23.20/kg (Grade A)
  PMFBY Insurance Premium: 2% Kharif
  PM-KISAN: Rs 2000/season
  Selling advice: Avoid Oct-Nov glut. Best prices Dec-Feb or May-Jun.

CROP: WHEAT
  Central MSP 2024-25: Rs 21.75/kg
  PMFBY: 1.5% Rabi. Selling advice: Best Jun-Aug.

CROP: MAIZE
  Central MSP 2024-25: Rs 22.25/kg. PMFBY: 2% Kharif.
  Selling advice: Best Apr-Jun (poultry/starch demand).

CROP: JOWAR
  Central MSP 2024-25: Rs 33.71/kg (Hybrid). PMFBY: 2%.

CROP: BAJRA
  Central MSP 2024-25: Rs 25.25/kg. PMFBY: 2%.

CROP: RAGI
  Central MSP 2024-25: Rs 40.29/kg. PMFBY: 2%.

CROP: ARHAR/TUR
  Central MSP 2024-25: Rs 70.00/kg. PMFBY: 2%.
  PM-AASHA: 100% procurement guarantee up to 2028-29.
  Selling advice: Best May-Aug.

CROP: MOONG
  Central MSP 2024-25: Rs 86.82/kg. PMFBY: 2%.

CROP: URAD
  Central MSP 2024-25: Rs 71.00/kg. PMFBY: 2%.

CROP: GRAM (Chickpea)
  Central MSP 2024-25: Rs 54.40/kg. PMFBY: 1.5% Rabi.
  Selling advice: Best May-Aug.

CROP: MASOOR (Lentil)
  Central MSP 2024-25: Rs 61.00/kg. PMFBY: 1.5% Rabi.

CROP: GROUNDNUT
  Central MSP 2024-25: Rs 65.82/kg. PMFBY: 2%.

CROP: MUSTARD/RAPESEED
  Central MSP 2024-25: Rs 57.50/kg. PMFBY: 1.5%.

CROP: SUNFLOWER
  Central MSP 2024-25: Rs 70.80/kg. PMFBY: 2%.

CROP: SOYABEAN
  Central MSP 2024-25: Rs 49.92/kg. PMFBY: 2%.

CROP: SESAMUM
  Central MSP 2024-25: Rs 90.97/kg. PMFBY: 2%.

CROP: SAFFLOWER
  Central MSP 2024-25: Rs 58.00/kg. PMFBY: 1.5%.

CROP: COTTON
  Central MSP 2024-25: Rs 70.21/kg Medium Staple; Rs 74.21/kg Long Staple.
  PMFBY: 5% (commercial). Selling advice: Best Jan-Mar.

CROP: SUGARCANE
  FRP 2024-25: Rs 3.40/kg. Sold directly to mills.

STATE-WISE BONUS OVER CENTRAL MSP:
  Kerala (Rice):    +Rs 6.37/kg → Effective Rs 29.37/kg
  Telangana (Rice): +Rs 5.00/kg → Effective Rs 28.00/kg
  Maharashtra:      Namo Shetkari — extra Rs 2000/season
  Punjab/Haryana:   Near-100% procurement efficiency at central MSP
  UP:               Low procurement (~3% of rice reaches MSP)

GENERAL SCHEMES:
  PM-KISAN: Rs 6000/year (Rs 2000/season) direct to all farmer families.
  PMFBY: Farmer pays 2% Kharif, 1.5% Rabi, 5% commercial crops.
  KCC: Crop loan at 4% subsidized interest.
  e-NAM: Online price discovery across mandis.
"""
