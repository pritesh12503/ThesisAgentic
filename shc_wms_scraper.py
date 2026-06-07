"""
SHC WMS Farm-Level Soil Scraper — Production Version
======================================================
Improvements over v3:
  1. Retry logic with exponential backoff for connection errors
  2. Smaller tile size (0.03 deg) for better farm coverage
  3. Session reconnect on SSL/connection failure
  4. Progress saved after every district (safe to Ctrl+C and resume)
  5. Resume mode: skips already-scraped districts if CSV exists
  6. Rate limiting: auto-slows down if server starts erroring

INSTALL:
  pip install requests pandas

USAGE:
  # Test with 3 districts
  python shc_wms_scraper.py --wms-base "YOUR_URL" --states "CHHATTISGARH" --max-districts 3

  # Run all Chhattisgarh districts
  python shc_wms_scraper.py --wms-base "YOUR_URL" --states "CHHATTISGARH"

  # Resume interrupted run (skips already done districts)
  python shc_wms_scraper.py --wms-base "YOUR_URL" --states "CHHATTISGARH" --resume

YOUR WMS URL:
  https://soilhealth.dac.gov.in/jW8X3zM5Y7pQvLr4K2Tn6...YOURTOKEN.../shc/wms/wms
  (get from DevTools -> Network -> wms?service=WMS request -> Request URL -> keep up to /shc/wms/wms)
"""

import requests
import pandas as pd
import statistics
import time
import logging
import argparse
import sys
import math
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("shc_wms.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# ── PASTE YOUR WMS URL HERE (or pass via --wms-base) ──────────────────────────
WMS_BASE    = "PASTE_YOUR_WMS_URL_HERE"
LAYERS_BASE = "https://soilhealth.dac.gov.in/q8ZdH3f0mX1y7nJrP2K5BvW9aQpLb-6TsFcYzC4oUtN_MwRiDgGZ0xVsJe7Xy8nMk2TjPqFbD1C5LvOr9WQ6Xa3lYsN7V1sRmJez4OtUbY0Qn9hPk6WfLd2Y8oSvK3UtGmX5C7bT9Pn6xV0JfZ1TzQm8LrV9aGkM2JpXeN4fUoQ8SwCiRzN7VtPk1XgW5/public/layers"

STATE_CLIMATE = {
    "ANDHRA PRADESH":    {"T": 29.0, "H": 72, "R": 83},
    "ARUNACHAL PRADESH": {"T": 18.0, "H": 80, "R": 220},
    "ASSAM":             {"T": 24.0, "H": 82, "R": 170},
    "BIHAR":             {"T": 25.0, "H": 70, "R": 105},
    "CHHATTISGARH":      {"T": 27.0, "H": 72, "R": 130},
    "GOA":               {"T": 27.5, "H": 80, "R": 250},
    "GUJARAT":           {"T": 28.0, "H": 60, "R": 75},
    "HARYANA":           {"T": 25.0, "H": 57, "R": 60},
    "HIMACHAL PRADESH":  {"T": 15.0, "H": 65, "R": 120},
    "JHARKHAND":         {"T": 26.0, "H": 68, "R": 115},
    "KARNATAKA":         {"T": 26.5, "H": 70, "R": 110},
    "KERALA":            {"T": 27.0, "H": 85, "R": 262},
    "MADHYA PRADESH":    {"T": 27.0, "H": 64, "R": 110},
    "MAHARASHTRA":       {"T": 27.5, "H": 65, "R": 100},
    "MANIPUR":           {"T": 20.0, "H": 78, "R": 150},
    "MEGHALAYA":         {"T": 18.0, "H": 84, "R": 240},
    "MIZORAM":           {"T": 21.0, "H": 80, "R": 195},
    "NAGALAND":          {"T": 20.0, "H": 79, "R": 165},
    "ODISHA":            {"T": 27.0, "H": 75, "R": 140},
    "PUNJAB":            {"T": 24.0, "H": 60, "R": 55},
    "RAJASTHAN":         {"T": 28.0, "H": 45, "R": 38},
    "SIKKIM":            {"T": 15.0, "H": 78, "R": 185},
    "TAMIL NADU":        {"T": 28.5, "H": 75, "R": 95},
    "TELANGANA":         {"T": 29.0, "H": 68, "R": 88},
    "TRIPURA":           {"T": 24.5, "H": 82, "R": 185},
    "UTTAR PRADESH":     {"T": 25.5, "H": 65, "R": 90},
    "UTTARAKHAND":       {"T": 20.0, "H": 68, "R": 130},
    "WEST BENGAL":       {"T": 26.0, "H": 78, "R": 145},
    "JAMMU & KASHMIR":   {"T": 14.0, "H": 65, "R": 95},
    "LADAKH":            {"T":  5.0, "H": 35, "R": 10},
}

STATE_NPK_FALLBACK = {
    "ANDHRA PRADESH":  {"N": 42, "P": 18, "K": 200, "pH": 7.5},
    "ASSAM":           {"N": 48, "P": 20, "K": 160, "pH": 5.8},
    "BIHAR":           {"N": 38, "P": 15, "K": 175, "pH": 7.8},
    "CHHATTISGARH":    {"N": 42, "P": 20, "K": 185, "pH": 6.8},
    "GUJARAT":         {"N": 35, "P": 16, "K": 220, "pH": 7.9},
    "HARYANA":         {"N": 40, "P": 17, "K": 240, "pH": 8.0},
    "KARNATAKA":       {"N": 45, "P": 21, "K": 190, "pH": 7.2},
    "KERALA":          {"N": 55, "P": 28, "K": 145, "pH": 5.9},
    "MADHYA PRADESH":  {"N": 40, "P": 16, "K": 195, "pH": 7.5},
    "MAHARASHTRA":     {"N": 43, "P": 19, "K": 210, "pH": 7.4},
    "ODISHA":          {"N": 44, "P": 20, "K": 180, "pH": 6.5},
    "PUNJAB":          {"N": 38, "P": 16, "K": 235, "pH": 8.1},
    "RAJASTHAN":       {"N": 32, "P": 14, "K": 225, "pH": 8.2},
    "TAMIL NADU":      {"N": 46, "P": 22, "K": 200, "pH": 7.3},
    "TELANGANA":       {"N": 41, "P": 18, "K": 195, "pH": 7.4},
    "UTTAR PRADESH":   {"N": 39, "P": 16, "K": 220, "pH": 7.9},
    "WEST BENGAL":     {"N": 46, "P": 21, "K": 168, "pH": 6.4},
}

STATES = [
    {"name": "ANDAMAN & NICOBAR",    "code": "35"},
    {"name": "ANDHRA PRADESH",       "code": "28"},
    {"name": "ARUNACHAL PRADESH",    "code": "12"},
    {"name": "ASSAM",                "code": "18"},
    {"name": "BIHAR",                "code": "10"},
    {"name": "CHHATTISGARH",         "code": "22"},
    {"name": "GOA",                  "code": "30"},
    {"name": "GUJARAT",              "code": "24"},
    {"name": "HARYANA",              "code": "6"},
    {"name": "HIMACHAL PRADESH",     "code": "2"},
    {"name": "JAMMU & KASHMIR",      "code": "1"},
    {"name": "JHARKHAND",            "code": "20"},
    {"name": "KARNATAKA",            "code": "29"},
    {"name": "KERALA",               "code": "32"},
    {"name": "LADAKH",               "code": "37"},
    {"name": "MADHYA PRADESH",       "code": "23"},
    {"name": "MAHARASHTRA",          "code": "27"},
    {"name": "MANIPUR",              "code": "14"},
    {"name": "MEGHALAYA",            "code": "17"},
    {"name": "MIZORAM",              "code": "15"},
    {"name": "NAGALAND",             "code": "13"},
    {"name": "ODISHA",               "code": "21"},
    {"name": "PUNJAB",               "code": "3"},
    {"name": "RAJASTHAN",            "code": "8"},
    {"name": "SIKKIM",               "code": "11"},
    {"name": "TAMIL NADU",           "code": "33"},
    {"name": "TELANGANA",            "code": "36"},
    {"name": "TRIPURA",              "code": "16"},
    {"name": "UTTAR PRADESH",        "code": "9"},
    {"name": "UTTARAKHAND",          "code": "5"},
    {"name": "WEST BENGAL",          "code": "19"},
]

CG_DISTRICTS = [
    {"name": "BALOD",           "code": "646"},
    {"name": "BALODA BAZAR",    "code": "644"},
    {"name": "BASTAR",          "code": "374"},
    {"name": "BEMETARA",        "code": "650"},
    {"name": "BIJAPUR",         "code": "636"},
    {"name": "BALRAMPUR",       "code": "649"},
    {"name": "BILASPUR",        "code": "375"},
    {"name": "DANTEWADA",       "code": "376"},
    {"name": "DHAMTARI",        "code": "377"},
    {"name": "DURG",            "code": "378"},
    {"name": "GARIYABAND",      "code": "645"},
    {"name": "GAURELLA PENDRA MARWAHI", "code": "734"},
    {"name": "JANJGIR-CHAMPA",  "code": "379"},
    {"name": "JASHPUR",         "code": "380"},
    {"name": "KABIRDHAM",       "code": "382"},
    {"name": "KANKER",          "code": "381"},
    {"name": "KONDAGAON",       "code": "643"},
    {"name": "KORBA",           "code": "383"},
    {"name": "KOREA",           "code": "384"},
    {"name": "MAHASAMUND",      "code": "385"},
    {"name": "MUNGELI",         "code": "647"},
    {"name": "NARAYANPUR",      "code": "637"},
    {"name": "RAIGARH",         "code": "386"},
    {"name": "RAIPUR",          "code": "387"},
    {"name": "RAJNANDGAON",     "code": "388"},
    {"name": "SAKTI",           "code": "762"},
    {"name": "SARANGARH BILAIGARH", "code": "763"},
    {"name": "SUKMA",           "code": "642"},
    {"name": "SURAJPUR",        "code": "648"},
    {"name": "SURGUJA",         "code": "389"},
]

# ── Adaptive delay tracker ─────────────────────────────────────────────────────
class AdaptiveDelay:
    """Automatically increases delay when server errors occur."""
    def __init__(self, base=0.4, max_delay=5.0):
        self.delay     = base
        self.base      = base
        self.max_delay = max_delay
        self.errors    = 0

    def ok(self):
        self.errors = max(0, self.errors - 1)
        self.delay  = max(self.base, self.delay * 0.95)

    def error(self):
        self.errors += 1
        self.delay   = min(self.max_delay, self.delay * 1.5)
        log.info(f"    [Rate limiter] Delay increased to {self.delay:.1f}s")

    def wait(self):
        time.sleep(self.delay)


def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
        "Accept":     "application/json, */*",
        "Referer":    "https://soilhealth.dac.gov.in/",
    })
    return s


def get_district_info(session, state_code, district_code, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(
                LAYERS_BASE,
                params={"state_code": state_code, "district_code": district_code},
                timeout=15
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.warning(f"    layers error attempt {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def wms_get_features(session, wms_base, layer_name,
                     minx, miny, maxx, maxy, throttle, retries=3):
    """WMS GetFeatureInfo with retry and adaptive throttling."""
    params = {
        "service":       "WMS",
        "version":       "1.1.1",
        "request":       "GetFeatureInfo",
        "format":        "image/png",
        "transparent":   "true",
        "layers":        layer_name,
        "query_layers":  layer_name,
        "exceptions":    "application/vnd.ogc.se_inimage",
        "srs":           "EPSG:4326",
        "width":         "101",
        "height":        "101",
        "X":             "50",
        "Y":             "50",
        "feature_count": "500",
        "info_format":   "application/json",
        "bbox":          f"{minx},{miny},{maxx},{maxy}",
        "HIDE_GEOMETRY": "true",
    }

    for attempt in range(retries):
        try:
            r = session.get(wms_base, params=params, timeout=25)
            if r.status_code == 200:
                throttle.ok()
                data = r.json()
                return [
                    {"id": f.get("id", ""), "props": f.get("properties", {})}
                    for f in data.get("features", [])
                    if f.get("properties")
                ]
            else:
                throttle.error()
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            throttle.error()
            wait = 3 * (attempt + 1)
            log.warning(f"    WMS error attempt {attempt+1}: "
                        f"{str(e)[:60]}... retrying in {wait}s")
            # Reconnect session on SSL errors
            if "SSL" in str(e) or "EOF" in str(e):
                session.close()
                time.sleep(wait)
                session.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
                    "Accept":     "application/json, */*",
                    "Referer":    "https://soilhealth.dac.gov.in/",
                })
            else:
                time.sleep(wait)

    return []


def safe_float(val):
    try:
        v = float(str(val).strip())
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def scrape_district(session, wms_base, state_code, district_code,
                    district_name, tile_size=0.03):
    """
    Scrape farm-level soil data for one district.
    tile_size=0.03 degrees (~3.3km) gives better coverage than 0.05.
    Tries all available SHC cycles, picks the one with most farms.
    """
    throttle = AdaptiveDelay(base=0.4)

    info = get_district_info(session, state_code, district_code)
    if not info:
        return None

    bbox = info.get("bbox", {})
    minx = bbox.get("minx")
    miny = bbox.get("miny")
    maxx = bbox.get("maxx")
    maxy = bbox.get("maxy")

    if None in [minx, miny, maxx, maxy]:
        return None

    layers = info.get("shcLayers", [])
    if not layers:
        return None

    # Sort cycles: most recent first
    cycle_order = ["2025-26", "2024-25", "2023-24", "2022-23",
                   "2021-22", "2020-21", "2019-21", "2017-19", "2015-17"]
    sorted_layers = sorted(
        layers,
        key=lambda x: cycle_order.index(x) if x in cycle_order else 99
    )

    best_farms  = 0
    best_result = None
    best_cycle  = None

    for layer_year in sorted_layers:
        layer_name  = f"{state_code}_{district_code}_shc_{layer_year}"
        n_x         = max(1, math.ceil((maxx - minx) / tile_size))
        n_y         = max(1, math.ceil((maxy - miny) / tile_size))
        x_step      = (maxx - minx) / n_x
        y_step      = (maxy - miny) / n_y
        total_tiles = n_x * n_y

        log.info(f"    Layer: {layer_name} | "
                 f"Grid: {n_x}x{n_y}={total_tiles} tiles")

        all_farms = {}
        tiles_with_data = 0

        for xi in range(n_x):
            for yi in range(n_y):
                tx0 = minx + xi * x_step
                ty0 = miny + yi * y_step
                tx1 = tx0 + x_step
                ty1 = ty0 + y_step

                features = wms_get_features(
                    session, wms_base, layer_name,
                    tx0, ty0, tx1, ty1, throttle
                )

                if features:
                    tiles_with_data += 1
                    for f in features:
                        fid = f["id"]
                        if fid and fid not in all_farms:
                            all_farms[fid] = f["props"]

                throttle.wait()

        log.info(f"    Cycle {layer_year}: {len(all_farms)} farms "
                 f"({tiles_with_data}/{total_tiles} tiles had data)")

        if len(all_farms) > best_farms:
            best_farms  = len(all_farms)
            best_result = dict(all_farms)
            best_cycle  = layer_year

        # Stop if we have enough farms
        if best_farms >= 30:
            log.info(f"    Good coverage in {layer_year}. Stopping.")
            break

        time.sleep(1.0)

    if not best_result:
        return None

    # Compute district medians from all collected farm readings
    vals = defaultdict(list)
    for props in best_result.values():
        for key in ["N", "P", "K", "pH", "OC", "B", "Fe", "Zn", "S"]:
            v = safe_float(props.get(key))
            if v is not None:
                vals[key].append(v)

    result = {}
    for key in ["N", "P", "K", "pH", "OC"]:
        v_list = vals.get(key, [])
        if v_list:
            result[key]      = round(statistics.median(v_list), 2)
            result[f"{key}_n"] = len(v_list)

    result["farms_sampled"] = best_farms
    result["cycle"]         = best_cycle

    log.info(f"    FINAL: N={result.get('N')}, P={result.get('P')}, "
             f"K={result.get('K')}, pH={result.get('pH')}, "
             f"OC={result.get('OC')} | farms={best_farms}")
    return result


def save(rows, output):
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["district", "state"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    log.info(f"Saved {len(df)} rows -> {output}")


def load_existing(output):
    """Load already-scraped districts for resume mode."""
    if Path(output).exists():
        try:
            df = pd.read_csv(output)
            done = set(df["district"].str.upper().tolist())
            log.info(f"Resume mode: {len(done)} districts already done")
            return df.to_dict("records"), done
        except Exception:
            pass
    return [], set()


def scrape(wms_base, target_states=None, max_districts=None,
           output="data/shc_district_soil.csv", resume=False):

    session = make_session()
    rows, already_done = load_existing(output) if resume else ([], set())
    count = len(already_done)

    states = STATES
    if target_states:
        up     = [s.upper().strip() for s in target_states]
        states = [s for s in states if s["name"].upper() in up]
        log.info(f"Running for: {[s['name'] for s in states]}")

    for si, state in enumerate(states, 1):
        sname = state["name"]
        scode = state["code"]
        clim  = STATE_CLIMATE.get(sname, {"T": 25, "H": 70, "R": 100})
        fb    = STATE_NPK_FALLBACK.get(sname,
                                       {"N": 40, "P": 18, "K": 190, "pH": 7.2})

        log.info(f"\n[{si}/{len(states)}] {sname} (code={scode})")

        if sname == "CHHATTISGARH":
            districts = CG_DISTRICTS
        else:
            log.warning(f"  District list not added for {sname} yet")
            continue

        for di, dist in enumerate(districts, 1):
            if max_districts and count >= (len(already_done) + max_districts):
                log.info(f"Reached max {max_districts} new districts. Stopping.")
                save(rows, output)
                return rows

            dname = dist["name"]
            dcode = dist["code"]

            # Skip if already scraped (resume mode)
            if dname.upper() in already_done:
                log.info(f"  [{di}/{len(districts)}] {dname} -- already done, skipping")
                continue

            log.info(f"\n  [{di}/{len(districts)}] {dname} (code={dcode})")

            nutrients = scrape_district(
                session, wms_base, scode, dcode, dname,
                tile_size=0.03
            )

            farms = nutrients.get("farms_sampled", 0) if nutrients else 0

            if farms >= 20:
                source = "SHC_WMS_Farms"
            elif farms >= 5:
                source = "SHC_WMS_LowConf"
                log.warning(f"  Low confidence: only {farms} farms")
            else:
                nutrients = fb.copy()
                nutrients.update({"farms_sampled": 0, "cycle": ""})
                source = "ICAR_State"
                log.warning(f"  Using ICAR fallback for {dname}")

            rows.append({
                "district":         dname.title(),
                "state":            sname.title(),
                "N":                nutrients.get("N"),
                "P":                nutrients.get("P"),
                "K":                nutrients.get("K"),
                "pH":               nutrients.get("pH"),
                "OC":               nutrients.get("OC"),
                "Temperature":      clim["T"],
                "Humidity":         clim["H"],
                "Rainfall_monthly": clim["R"],
                "data_source":      source,
                "farms_sampled":    nutrients.get("farms_sampled", 0),
                "cycle":            nutrients.get("cycle", ""),
                "district_code":    dcode,
                "state_code":       scode,
            })
            count += 1

            # Save after every district
            save(rows, output)
            time.sleep(1.0)

        log.info(f"  State done. Total: {count}")

    save(rows, output)
    log.info(f"\nComplete. {count} districts in output file.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wms-base",      default=WMS_BASE)
    ap.add_argument("--states",        default=None,
                    help='e.g. "CHHATTISGARH,MAHARASHTRA"')
    ap.add_argument("--max-districts", type=int, default=None,
                    help="Max NEW districts to scrape this run")
    ap.add_argument("--output",        default="data/shc_district_soil.csv")
    ap.add_argument("--resume",        action="store_true",
                    help="Skip districts already in output CSV")
    args = ap.parse_args()

    wms = args.wms_base
    if wms == "PASTE_YOUR_WMS_URL_HERE":
        print("ERROR: Provide --wms-base URL")
        print("Get from DevTools -> Network -> wms?service=WMS request")
        print("Copy Request URL, keep only up to /shc/wms/wms")
        sys.exit(1)

    target = [s.strip() for s in args.states.split(",")] if args.states else None

    print("=" * 62)
    print("  SHC WMS Farm-Level Soil Scraper — Production")
    print(f"  States:        {target or 'All'}")
    print(f"  Max new:       {args.max_districts or 'No limit'}")
    print(f"  Tile size:     0.03 degrees (~3.3 km)")
    print(f"  Resume mode:   {args.resume}")
    print(f"  Output:        {args.output}")
    print("=" * 62)
    print()
    print("Tip: Press Ctrl+C anytime -- progress is saved after each district.")
    print()

    rows = scrape(wms, target, args.max_districts, args.output, args.resume)

    if rows:
        df = pd.read_csv(args.output)
        show = ["district", "state", "N", "P", "K", "pH", "OC",
                "farms_sampled", "cycle", "data_source"]
        print(f"\n  {len(df)} districts in {args.output}")
        print()
        print(df[show].to_string(index=False))
    else:
        print("\nNo data. Check shc_wms.log")