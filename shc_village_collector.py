"""
SHC Village-Level Soil Data Collector
=======================================
Instead of computing district medians, this saves EVERY village
as a separate row. The AgriAdvisor UI then lets the farmer select
their village for exact soil values.

Output CSV columns:
  village, district, state, N, P, K, pH, OC, B, Fe, Zn, S,
  Temperature, Humidity, Rainfall_monthly,
  feature_id, cycle, district_code, state_code

This gives real farm-level data instead of statistical aggregates.

USAGE:
  python shc_village_collector.py --wms-base "YOUR_URL" --states "CHHATTISGARH"
  python shc_village_collector.py --wms-base "YOUR_URL" --states "CHHATTISGARH" --resume
"""

import requests
import pandas as pd
import time
import logging
import argparse
import sys
import math
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("shc_village.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

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


class AdaptiveDelay:
    def __init__(self, base=0.4):
        self.delay = base
        self.base  = base

    def ok(self):
        self.delay = max(self.base, self.delay * 0.95)

    def error(self):
        self.delay = min(5.0, self.delay * 1.6)
        log.info(f"    Rate limit: delay -> {self.delay:.1f}s")

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


def get_district_info(session, state_code, district_code):
    for attempt in range(3):
        try:
            r = session.get(
                LAYERS_BASE,
                params={"state_code": state_code,
                        "district_code": district_code},
                timeout=15
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.warning(f"    layers error: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def wms_get_features(session, wms_base, layer_name,
                     minx, miny, maxx, maxy, throttle):
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
    for attempt in range(3):
        try:
            r = session.get(wms_base, params=params, timeout=25)
            if r.status_code == 200:
                throttle.ok()
                data = r.json()
                return [
                    {"id": f.get("id", ""),
                     "props": f.get("properties", {})}
                    for f in data.get("features", [])
                    if f.get("properties")
                ]
            throttle.error()
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            throttle.error()
            wait = 3 * (attempt + 1)
            if "SSL" in str(e) or "EOF" in str(e) or "abort" in str(e).lower():
                log.warning(f"    Connection error, reconnecting...")
                session.close()
            else:
                log.warning(f"    WMS error: {str(e)[:50]}")
            time.sleep(wait)
    return []


def safe_float(val):
    try:
        v = float(str(val).strip())
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def collect_district_villages(session, wms_base, state_code,
                               district_code, district_name,
                               state_name, tile_size=0.03):
    """
    Collect ALL village soil readings for a district.
    Returns list of village dicts (one row per village/farm).
    """
    throttle = AdaptiveDelay(base=0.4)

    info = get_district_info(session, state_code, district_code)
    if not info:
        return []

    bbox = info.get("bbox", {})
    minx = bbox.get("minx")
    miny = bbox.get("miny")
    maxx = bbox.get("maxx")
    maxy = bbox.get("maxy")

    if None in [minx, miny, maxx, maxy]:
        return []

    layers = info.get("shcLayers", [])
    if not layers:
        return []

    # Pick best cycle
    cycle_order = ["2025-26", "2024-25", "2023-24", "2022-23",
                   "2021-22", "2020-21", "2019-21", "2017-19", "2015-17"]
    sorted_layers = sorted(
        layers,
        key=lambda x: cycle_order.index(x) if x in cycle_order else 99
    )

    best_farms  = {}
    best_cycle  = None

    for layer_year in sorted_layers:
        layer_name  = f"{state_code}_{district_code}_shc_{layer_year}"
        n_x         = max(1, math.ceil((maxx - minx) / tile_size))
        n_y         = max(1, math.ceil((maxy - miny) / tile_size))
        x_step      = (maxx - minx) / n_x
        y_step      = (maxy - miny) / n_y

        log.info(f"    Layer: {layer_name} | Grid: {n_x}x{n_y}")

        all_farms = {}

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
                for f in features:
                    fid = f["id"]
                    if fid and fid not in all_farms:
                        all_farms[fid] = f["props"]

                throttle.wait()

        log.info(f"    Cycle {layer_year}: {len(all_farms)} farms found")

        if len(all_farms) > len(best_farms):
            best_farms = dict(all_farms)
            best_cycle = layer_year

        if len(best_farms) >= 30:
            break
        time.sleep(1.0)

    if not best_farms:
        log.warning(f"    No data for {district_name}")
        return []

    # Build village rows
    clim = STATE_CLIMATE.get(state_name, {"T": 25, "H": 70, "R": 100})
    rows = []

    for fid, props in best_farms.items():
        village = str(props.get("village", "")).strip().title()
        if not village:
            village = "Unknown"

        row = {
            "village":          village,
            "district":         district_name.title(),
            "state":            state_name.title(),
            "N":                safe_float(props.get("N")),
            "P":                safe_float(props.get("P")),
            "K":                safe_float(props.get("K")),
            "pH":               safe_float(props.get("pH")),
            "OC":               safe_float(props.get("OC")),
            "B":                safe_float(props.get("B")),
            "Fe":               safe_float(props.get("Fe")),
            "Zn":               safe_float(props.get("Zn")),
            "S":                safe_float(props.get("S")),
            "Temperature":      clim["T"],
            "Humidity":         clim["H"],
            "Rainfall_monthly": clim["R"],
            "cycle":            best_cycle,
            "feature_id":       fid,
            "district_code":    district_code,
            "state_code":       state_code,
        }
        rows.append(row)

    log.info(f"    Collected {len(rows)} village records for {district_name}")
    return rows


def save(rows, output):
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Sort by state, district, village for clean lookup
    df = df.sort_values(["state", "district", "village"]).reset_index(drop=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    log.info(f"Saved {len(df)} village records -> {output}")


def get_done_districts(output):
    """Get set of already-scraped districts for resume mode."""
    if Path(output).exists():
        try:
            df = pd.read_csv(output)
            done = set(df["district"].str.upper().tolist())
            log.info(f"Resume: {len(done)} districts already scraped")
            return done
        except Exception:
            pass
    return set()


def scrape(wms_base, target_states=None, max_districts=None,
           output="data/shc_village_soil.csv", resume=False):

    session  = make_session()
    all_rows = []
    done     = set()

    # Load existing data if resuming
    if resume and Path(output).exists():
        try:
            existing = pd.read_csv(output)
            all_rows = existing.to_dict("records")
            done     = set(existing["district"].str.upper().tolist())
            log.info(f"Loaded {len(all_rows)} existing records, "
                     f"{len(done)} districts done")
        except Exception as e:
            log.warning(f"Could not load existing: {e}")

    states = [{"name": "CHHATTISGARH", "code": "22"}]  # extend as needed
    if target_states:
        # Filter to requested states
        all_s  = [
            {"name": "ANDHRA PRADESH", "code": "28"},
            {"name": "ASSAM", "code": "18"},
            {"name": "BIHAR", "code": "10"},
            {"name": "CHHATTISGARH", "code": "22"},
            {"name": "GUJARAT", "code": "24"},
            {"name": "HARYANA", "code": "6"},
            {"name": "JHARKHAND", "code": "20"},
            {"name": "KARNATAKA", "code": "29"},
            {"name": "KERALA", "code": "32"},
            {"name": "MADHYA PRADESH", "code": "23"},
            {"name": "MAHARASHTRA", "code": "27"},
            {"name": "ODISHA", "code": "21"},
            {"name": "PUNJAB", "code": "3"},
            {"name": "RAJASTHAN", "code": "8"},
            {"name": "TAMIL NADU", "code": "33"},
            {"name": "TELANGANA", "code": "36"},
            {"name": "UTTAR PRADESH", "code": "9"},
            {"name": "UTTARAKHAND", "code": "5"},
            {"name": "WEST BENGAL", "code": "19"},
        ]
        up     = [s.upper().strip() for s in target_states]
        states = [s for s in all_s if s["name"].upper() in up]

    new_count = 0

    for si, state in enumerate(states, 1):
        sname = state["name"]
        scode = state["code"]
        log.info(f"\n[{si}/{len(states)}] {sname}")

        if sname == "CHHATTISGARH":
            districts = CG_DISTRICTS
        else:
            log.warning(f"  No district list for {sname}")
            continue

        for di, dist in enumerate(districts, 1):
            if max_districts and new_count >= max_districts:
                log.info(f"Reached max {max_districts}. Stopping.")
                save(all_rows, output)
                return all_rows

            dname = dist["name"]
            dcode = dist["code"]

            if resume and dname.upper() in done:
                log.info(f"  [{di}/{len(districts)}] {dname} -- skipping (done)")
                continue

            log.info(f"\n  [{di}/{len(districts)}] {dname} (code={dcode})")

            village_rows = collect_district_villages(
                session, wms_base, scode, dcode, dname, sname,
                tile_size=0.03
            )

            if village_rows:
                all_rows.extend(village_rows)
                log.info(f"  Added {len(village_rows)} villages from {dname}")
            else:
                log.warning(f"  No village data for {dname}")

            new_count += 1
            # Save after every district
            save(all_rows, output)
            time.sleep(1.0)

        log.info(f"  State {sname} done.")

    save(all_rows, output)
    total = len(pd.read_csv(output)) if Path(output).exists() else 0
    log.info(f"\nDone. {total} village records in {output}")
    return all_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wms-base",      default=WMS_BASE)
    ap.add_argument("--states",        default=None)
    ap.add_argument("--max-districts", type=int, default=None)
    ap.add_argument("--output",        default="data/shc_village_soil.csv")
    ap.add_argument("--resume",        action="store_true")
    args = ap.parse_args()

    wms = args.wms_base
    if wms == "PASTE_YOUR_WMS_URL_HERE":
        print("ERROR: Provide --wms-base URL")
        sys.exit(1)

    target = [s.strip() for s in args.states.split(",")] if args.states else None

    print("=" * 62)
    print("  SHC Village-Level Soil Collector")
    print(f"  States:        {target or 'All'}")
    print(f"  Max districts: {args.max_districts or 'No limit'}")
    print(f"  Output:        {args.output}")
    print(f"  Resume:        {args.resume}")
    print("=" * 62)
    print()
    print("Each village saved as a SEPARATE ROW.")
    print("UI will have: State -> District -> Village dropdowns.")
    print("Ctrl+C safe -- saves after every district.")
    print()

    rows = scrape(wms, target, args.max_districts, args.output, args.resume)

    if rows:
        df = pd.read_csv(args.output)
        print(f"\n  Total: {len(df)} village records")
        print(f"  Districts covered: {df['district'].nunique()}")
        print(f"  Villages covered: {df['village'].nunique()}")
        print()
        show = ["village", "district", "state", "N", "P", "K", "pH", "OC", "cycle"]
        print(df[show].head(15).to_string(index=False))
    else:
        print("\nNo data collected.")