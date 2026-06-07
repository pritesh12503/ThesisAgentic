"""
SHC Fallback Builder — FIXED VERSION
======================================
Fixes:
  1. 38/48 bug: thesis CSV districts that overlap with hardcoded SHC list
     were being counted as SHC_Published and skipping thesis values.
     Fix: thesis CSV values OVERRIDE hardcoded SHC values (your lab data
     is more specific than generic hardcoded values).

  2. J&K / Delhi / UTs skipped: added those state profiles.

USAGE:
  python shc_fallback_builder.py
  python shc_fallback_builder.py --districts-csv data/district_coordinates.csv --thesis-csv data/thesis_agronomic_dataset_clean.csv --output data/shc_district_soil.csv
"""

import pandas as pd
import argparse
from pathlib import Path

# ── ICAR state profiles ────────────────────────────────────────────────────────
STATE_PROFILES = {
    "andhra pradesh":          {"N": 42, "P": 18, "K": 200, "pH": 7.5, "T": 29.0, "H": 72, "R": 83},
    "arunachal pradesh":       {"N": 55, "P": 25, "K": 180, "pH": 6.0, "T": 18.0, "H": 80, "R": 220},
    "assam":                   {"N": 48, "P": 20, "K": 160, "pH": 5.8, "T": 24.0, "H": 82, "R": 170},
    "bihar":                   {"N": 38, "P": 15, "K": 175, "pH": 7.8, "T": 25.0, "H": 70, "R": 105},
    "chhattisgarh":            {"N": 42, "P": 20, "K": 185, "pH": 6.8, "T": 27.0, "H": 72, "R": 130},
    "goa":                     {"N": 50, "P": 22, "K": 140, "pH": 6.2, "T": 27.5, "H": 80, "R": 250},
    "gujarat":                 {"N": 35, "P": 16, "K": 220, "pH": 7.9, "T": 28.0, "H": 60, "R": 75},
    "haryana":                 {"N": 40, "P": 17, "K": 240, "pH": 8.0, "T": 25.0, "H": 57, "R": 60},
    "himachal pradesh":        {"N": 52, "P": 24, "K": 160, "pH": 6.5, "T": 15.0, "H": 65, "R": 120},
    "jharkhand":               {"N": 44, "P": 19, "K": 170, "pH": 6.2, "T": 26.0, "H": 68, "R": 115},
    "karnataka":               {"N": 45, "P": 21, "K": 190, "pH": 7.2, "T": 26.5, "H": 70, "R": 110},
    "kerala":                  {"N": 55, "P": 28, "K": 145, "pH": 5.9, "T": 27.0, "H": 85, "R": 262},
    "madhya pradesh":          {"N": 40, "P": 16, "K": 195, "pH": 7.5, "T": 27.0, "H": 64, "R": 110},
    "maharashtra":             {"N": 43, "P": 19, "K": 210, "pH": 7.4, "T": 27.5, "H": 65, "R": 100},
    "manipur":                 {"N": 50, "P": 23, "K": 155, "pH": 6.0, "T": 20.0, "H": 78, "R": 150},
    "meghalaya":               {"N": 54, "P": 26, "K": 150, "pH": 5.7, "T": 18.0, "H": 84, "R": 240},
    "mizoram":                 {"N": 52, "P": 25, "K": 148, "pH": 5.8, "T": 21.0, "H": 80, "R": 195},
    "nagaland":                {"N": 53, "P": 24, "K": 152, "pH": 5.9, "T": 20.0, "H": 79, "R": 165},
    "odisha":                  {"N": 44, "P": 20, "K": 180, "pH": 6.5, "T": 27.0, "H": 75, "R": 140},
    "punjab":                  {"N": 38, "P": 16, "K": 235, "pH": 8.1, "T": 24.0, "H": 60, "R": 55},
    "rajasthan":               {"N": 32, "P": 14, "K": 225, "pH": 8.2, "T": 28.0, "H": 45, "R": 38},
    "sikkim":                  {"N": 55, "P": 27, "K": 155, "pH": 6.0, "T": 15.0, "H": 78, "R": 185},
    "tamil nadu":              {"N": 46, "P": 22, "K": 200, "pH": 7.3, "T": 28.5, "H": 75, "R": 95},
    "telangana":               {"N": 41, "P": 18, "K": 195, "pH": 7.4, "T": 29.0, "H": 68, "R": 88},
    "tripura":                 {"N": 48, "P": 21, "K": 158, "pH": 6.1, "T": 24.5, "H": 82, "R": 185},
    "uttar pradesh":           {"N": 39, "P": 16, "K": 220, "pH": 7.9, "T": 25.5, "H": 65, "R": 90},
    "uttarakhand":             {"N": 50, "P": 23, "K": 175, "pH": 6.8, "T": 20.0, "H": 68, "R": 130},
    "west bengal":             {"N": 46, "P": 21, "K": 168, "pH": 6.4, "T": 26.0, "H": 78, "R": 145},
    # ── UTs and Special Regions ──
    "jammu and kashmir":       {"N": 48, "P": 22, "K": 155, "pH": 7.2, "T": 14.0, "H": 65, "R": 95},
    "ladakh":                  {"N": 30, "P": 12, "K": 140, "pH": 7.8, "T":  5.0, "H": 35, "R": 10},
    "delhi":                   {"N": 36, "P": 15, "K": 210, "pH": 8.0, "T": 25.0, "H": 60, "R": 65},
    "puducherry":              {"N": 45, "P": 21, "K": 195, "pH": 7.4, "T": 28.5, "H": 76, "R": 110},
    "chandigarh":              {"N": 40, "P": 17, "K": 235, "pH": 8.0, "T": 23.0, "H": 60, "R": 62},
    "andaman and nicobar islands": {"N": 52, "P": 24, "K": 150, "pH": 6.5, "T": 28.0, "H": 80, "R": 280},
    "lakshadweep":             {"N": 50, "P": 22, "K": 145, "pH": 6.8, "T": 28.5, "H": 85, "R": 160},
    "dadra and nagar haveli":  {"N": 44, "P": 20, "K": 175, "pH": 6.9, "T": 27.5, "H": 75, "R": 190},
    "daman and diu":           {"N": 38, "P": 17, "K": 180, "pH": 7.5, "T": 28.0, "H": 70, "R": 100},
}

# ── Hardcoded SHC district data ────────────────────────────────────────────────
# These are LOWER PRIORITY than thesis CSV (thesis has your lab-verified values)
SHC_DISTRICTS = [
    {"district": "raigarh",     "state": "chhattisgarh", "N": 42, "P": 20, "K": 185, "pH": 6.8},
    {"district": "raipur",      "state": "chhattisgarh", "N": 40, "P": 18, "K": 180, "pH": 7.0},
    {"district": "bilaspur",    "state": "chhattisgarh", "N": 44, "P": 22, "K": 190, "pH": 6.7},
    {"district": "durg",        "state": "chhattisgarh", "N": 41, "P": 19, "K": 182, "pH": 6.9},
    {"district": "korba",       "state": "chhattisgarh", "N": 38, "P": 17, "K": 175, "pH": 6.5},
    {"district": "pune",        "state": "maharashtra",  "N": 45, "P": 21, "K": 215, "pH": 7.5},
    {"district": "nashik",      "state": "maharashtra",  "N": 43, "P": 20, "K": 208, "pH": 7.3},
    {"district": "nagpur",      "state": "maharashtra",  "N": 41, "P": 18, "K": 200, "pH": 7.6},
    {"district": "aurangabad",  "state": "maharashtra",  "N": 39, "P": 17, "K": 210, "pH": 7.8},
    {"district": "solapur",     "state": "maharashtra",  "N": 36, "P": 15, "K": 218, "pH": 7.9},
    {"district": "ludhiana",    "state": "punjab",       "N": 40, "P": 17, "K": 240, "pH": 8.0},
    {"district": "amritsar",    "state": "punjab",       "N": 38, "P": 16, "K": 235, "pH": 8.1},
    {"district": "jalandhar",   "state": "punjab",       "N": 39, "P": 16, "K": 238, "pH": 8.0},
    {"district": "patiala",     "state": "punjab",       "N": 37, "P": 15, "K": 230, "pH": 8.2},
    {"district": "varanasi",    "state": "uttar pradesh","N": 41, "P": 17, "K": 225, "pH": 7.8},
    {"district": "lucknow",     "state": "uttar pradesh","N": 40, "P": 16, "K": 220, "pH": 7.9},
    {"district": "agra",        "state": "uttar pradesh","N": 37, "P": 15, "K": 215, "pH": 8.1},
    {"district": "kanpur",      "state": "uttar pradesh","N": 39, "P": 16, "K": 222, "pH": 8.0},
    {"district": "allahabad",   "state": "uttar pradesh","N": 40, "P": 17, "K": 218, "pH": 7.9},
    {"district": "dharwad",     "state": "karnataka",    "N": 46, "P": 22, "K": 192, "pH": 7.1},
    {"district": "mysuru",      "state": "karnataka",    "N": 44, "P": 21, "K": 188, "pH": 7.2},
    {"district": "belagavi",    "state": "karnataka",    "N": 45, "P": 20, "K": 190, "pH": 7.3},
    {"district": "coimbatore",  "state": "tamil nadu",   "N": 47, "P": 23, "K": 202, "pH": 7.2},
    {"district": "thanjavur",   "state": "tamil nadu",   "N": 48, "P": 24, "K": 205, "pH": 7.0},
    {"district": "salem",       "state": "tamil nadu",   "N": 45, "P": 22, "K": 198, "pH": 7.3},
    {"district": "indore",      "state": "madhya pradesh","N":41, "P": 17, "K": 198, "pH": 7.6},
    {"district": "bhopal",      "state": "madhya pradesh","N":40, "P": 16, "K": 195, "pH": 7.5},
    {"district": "jabalpur",    "state": "madhya pradesh","N":42, "P": 18, "K": 200, "pH": 7.4},
    {"district": "murshidabad", "state": "west bengal",  "N": 47, "P": 22, "K": 170, "pH": 6.3},
    {"district": "nadia",       "state": "west bengal",  "N": 46, "P": 21, "K": 168, "pH": 6.4},
]
# Build lowercase lookup
SHC_LOOKUP = {d["district"].lower(): d for d in SHC_DISTRICTS}


def build(districts_csv, thesis_csv=None, output="shc_district_soil.csv"):

    # ── Load district list ─────────────────────────────────────────────────────
    try:
        dist_df = pd.read_csv(districts_csv)
    except FileNotFoundError:
        print(f"ERROR: {districts_csv} not found.")
        return

    dist_df.columns = [c.lower().strip() for c in dist_df.columns]
    dist_col  = next((c for c in dist_df.columns if "district" in c), None)
    state_col = next((c for c in dist_df.columns if "state" in c), None)
    if not dist_col or not state_col:
        print(f"ERROR: Cannot find district/state columns. Got: {list(dist_df.columns)}")
        return

    print(f"Loaded {len(dist_df)} districts from {districts_csv}")

    # ── Load thesis CSV ────────────────────────────────────────────────────────
    # KEY FIX: thesis values go into a PRIORITY lookup
    # They override SHC hardcoded values because your thesis CSV
    # has verified, specific values for those districts
    thesis_lookup = {}
    if thesis_csv and Path(thesis_csv).exists():
        t_df = pd.read_csv(thesis_csv)
        t_df.columns = [c.lower().strip() for c in t_df.columns]

        # Find column names flexibly (handles N/nitrogen/Nitrogen etc.)
        def find_col(df, options):
            for o in options:
                if o in df.columns:
                    return o
            return None

        d_col  = find_col(t_df, ["district","District","DISTRICT"])
        n_col  = find_col(t_df, ["n","nitrogen","N"])
        p_col  = find_col(t_df, ["p","phosphorous","phosphorus","P"])
        k_col  = find_col(t_df, ["k","potassium","K"])
        ph_col = find_col(t_df, ["ph","pH","PH"])
        t_col  = find_col(t_df, ["temperature","temp","t","T"])
        h_col  = find_col(t_df, ["humidity","h","H"])
        r_col  = find_col(t_df, ["rainfall","rain","r","R"])

        for _, row in t_df.iterrows():
            key = str(row.get(d_col, "")).lower().strip()
            if not key:
                continue
            thesis_lookup[key] = {
                "N":   row[n_col]  if n_col  else None,
                "P":   row[p_col]  if p_col  else None,
                "K":   row[k_col]  if k_col  else None,
                "pH":  row[ph_col] if ph_col else None,
                "T":   row[t_col]  if t_col  else None,
                "H":   row[h_col]  if h_col  else None,
                "R":   row[r_col]  if r_col  else None,
            }

        print(f"Loaded {len(thesis_lookup)} districts from thesis CSV")

    # ── Build rows ─────────────────────────────────────────────────────────────
    rows = []
    sources = {"Thesis_CSV": 0, "SHC_Published": 0, "ICAR_State": 0, "Skipped": 0}

    for _, row in dist_df.iterrows():
        dist_name  = str(row[dist_col]).strip()
        state_name = str(row[state_col]).strip()

        # Skip rows where state is NaN (bad rows in CSV)
        if state_name.lower() in ("nan", "", "none"):
            sources["Skipped"] += 1
            continue

        dist_key  = dist_name.lower()
        state_key = state_name.lower()

        sp = STATE_PROFILES.get(state_key, {})

        # ── PRIORITY 1: Thesis CSV (your lab-verified values) ──────────────────
        if dist_key in thesis_lookup:
            t = thesis_lookup[dist_key]
            rows.append({
                "district":         dist_name,
                "state":            state_name,
                "N":                t["N"]  or sp.get("N"),
                "P":                t["P"]  or sp.get("P"),
                "K":                t["K"]  or sp.get("K"),
                "pH":               t["pH"] or sp.get("pH"),
                "Temperature":      t["T"]  or sp.get("T", 25.0),
                "Humidity":         t["H"]  or sp.get("H", 70.0),
                "Rainfall_monthly": t["R"]  or sp.get("R", 100.0),
                "data_source":      "Thesis_CSV",
            })
            sources["Thesis_CSV"] += 1

        # ── PRIORITY 2: Hardcoded SHC data ────────────────────────────────────
        elif dist_key in SHC_LOOKUP:
            d = SHC_LOOKUP[dist_key]
            rows.append({
                "district":         dist_name,
                "state":            state_name,
                "N":   d["N"],  "P": d["P"],
                "K":   d["K"],  "pH": d.get("pH"),
                "Temperature":      sp.get("T", 25.0),
                "Humidity":         sp.get("H", 70.0),
                "Rainfall_monthly": sp.get("R", 100.0),
                "data_source":      "SHC_Published",
            })
            sources["SHC_Published"] += 1

        # ── PRIORITY 3: ICAR state average ────────────────────────────────────
        elif sp:
            rows.append({
                "district":         dist_name,
                "state":            state_name,
                "N":   sp["N"],  "P": sp["P"],
                "K":   sp["K"],  "pH": sp["pH"],
                "Temperature":      sp["T"],
                "Humidity":         sp["H"],
                "Rainfall_monthly": sp["R"],
                "data_source":      "ICAR_State",
            })
            sources["ICAR_State"] += 1

        # ── No data at all ─────────────────────────────────────────────────────
        else:
            print(f"  SKIP: {dist_name}, {state_name} — state not in profile table")
            sources["Skipped"] += 1

    # ── Save ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["district", "state"])
    df.to_csv(output, index=False)

    print(f"\n{'='*55}")
    print(f"  Built {len(df)} district profiles → {output}")
    print(f"  Thesis CSV (your verified data): {sources['Thesis_CSV']} districts")
    print(f"  SHC Published (hardcoded):       {sources['SHC_Published']} districts")
    print(f"  ICAR State average:              {sources['ICAR_State']} districts")
    print(f"  Skipped (UTs/bad rows):          {sources['Skipped']}")
    print(f"{'='*55}")
    print(f"\nSample (first 10 rows):")
    print(df.head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--districts-csv", default="data/district_coordinates.csv")
    parser.add_argument("--thesis-csv",    default="data/thesis_agronomic_dataset_clean.csv")
    parser.add_argument("--output",        default="data/shc_district_soil.csv")
    args = parser.parse_args()

    build(
        districts_csv = args.districts_csv,
        thesis_csv    = args.thesis_csv,
        output        = args.output,
    )