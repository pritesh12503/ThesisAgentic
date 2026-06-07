"""
SHC Multi-State Village Collector
===================================
Collects village-level soil data for multiple states.
Add district lists as you collect them from the portal.

USAGE:
  python shc_multistate_collector.py --wms-base "YOUR_URL" --states "MAHARASHTRA"
  python shc_multistate_collector.py --wms-base "YOUR_URL" --states "CHHATTISGARH,MAHARASHTRA"
  python shc_multistate_collector.py --wms-base "YOUR_URL"   (all available states)
  python shc_multistate_collector.py --wms-base "YOUR_URL" --states "MAHARASHTRA" --resume
"""

import requests, pandas as pd, time, math, logging, argparse, sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("shc_multistate.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

WMS_BASE    = "PASTE_YOUR_WMS_URL_HERE"
LAYERS_BASE = "https://soilhealth.dac.gov.in/q8ZdH3f0mX1y7nJrP2K5BvW9aQpLb-6TsFcYzC4oUtN_MwRiDgGZ0xVsJe7Xy8nMk2TjPqFbD1C5LvOr9WQ6Xa3lYsN7V1sRmJez4OtUbY0Qn9hPk6WfLd2Y8oSvK3UtGmX5C7bT9Pn6xV0JfZ1TzQm8LrV9aGkM2JpXeN4fUoQ8SwCiRzN7VtPk1XgW5/public/layers"

STATE_CLIMATE = {
    "ANDHRA PRADESH":    {"T": 29.0, "H": 72, "R": 83},
    "ASSAM":             {"T": 24.0, "H": 82, "R": 170},
    "BIHAR":             {"T": 25.0, "H": 70, "R": 105},
    "CHHATTISGARH":      {"T": 27.0, "H": 72, "R": 130},
    "GUJARAT":           {"T": 28.0, "H": 60, "R": 75},
    "HARYANA":           {"T": 25.0, "H": 57, "R": 60},
    "JHARKHAND":         {"T": 26.0, "H": 68, "R": 115},
    "KARNATAKA":         {"T": 26.5, "H": 70, "R": 110},
    "KERALA":            {"T": 27.0, "H": 85, "R": 262},
    "MADHYA PRADESH":    {"T": 27.0, "H": 64, "R": 110},
    "MAHARASHTRA":       {"T": 27.5, "H": 65, "R": 100},
    "ODISHA":            {"T": 27.0, "H": 75, "R": 140},
    "PUNJAB":            {"T": 24.0, "H": 60, "R": 55},
    "RAJASTHAN":         {"T": 28.0, "H": 45, "R": 38},
    "TAMIL NADU":        {"T": 28.5, "H": 75, "R": 95},
    "TELANGANA":         {"T": 29.0, "H": 68, "R": 88},
    "UTTAR PRADESH":     {"T": 25.5, "H": 65, "R": 90},
    "UTTARAKHAND":       {"T": 20.0, "H": 68, "R": 130},
    "WEST BENGAL":       {"T": 26.0, "H": 78, "R": 145},
}

# ── District lists per state ───────────────────────────────────────────────────

DISTRICTS = {

    "CHHATTISGARH": {
        "code": "22",
        "districts": [
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
    },

    "MAHARASHTRA": {
        "code": "27",
        "districts": [
            {"name": "AHILYANAGAR",               "code": "466"},
            {"name": "AKOLA",                     "code": "467"},
            {"name": "AMRAVATI",                  "code": "468"},
            {"name": "BEED",                      "code": "470"},
            {"name": "BHANDARA",                  "code": "471"},
            {"name": "BULDHANA",                  "code": "472"},
            {"name": "CHANDRAPUR",                "code": "473"},
            {"name": "CHHATRAPATI SAMBHAJINAGAR", "code": "469"},
            {"name": "DHARASHIV",                 "code": "488"},
            {"name": "DHULE",                     "code": "474"},
            {"name": "GADCHIROLI",                "code": "475"},
            {"name": "GONDIA",                    "code": "476"},
            {"name": "HINGOLI",                   "code": "477"},
            {"name": "JALGAON",                   "code": "478"},
            {"name": "JALNA",                     "code": "479"},
            {"name": "KOLHAPUR",                  "code": "480"},
            {"name": "LATUR",                     "code": "481"},
            {"name": "MUMBAI SUBURBAN",           "code": "483"},
            {"name": "MUMBAI",                    "code": "482"},
            {"name": "NAGPUR",                    "code": "484"},
            {"name": "NANDED",                    "code": "485"},
            {"name": "NANDURBAR",                 "code": "486"},
            {"name": "NASHIK",                    "code": "487"},
            {"name": "PALGHAR",                   "code": "665"},
            {"name": "PARBHANI",                  "code": "489"},
            {"name": "PUNE",                      "code": "490"},
            {"name": "RAIGAD",                    "code": "491"},
            {"name": "RATNAGIRI",                 "code": "492"},
            {"name": "SANGLI",                    "code": "493"},
            {"name": "SATARA",                    "code": "494"},
            {"name": "SINDHUDURG",                "code": "495"},
            {"name": "SOLAPUR",                   "code": "496"},
            {"name": "THANE",                     "code": "497"},
            {"name": "WARDHA",                    "code": "498"},
            {"name": "WASHIM",                    "code": "499"},
            {"name": "YAVATMAL",                  "code": "500"},
        ]
    },

    # ── Add more states here as you collect district data ──────────────────────
    # "PUNJAB": {"code": "3", "districts": [...]},
    # "UTTAR PRADESH": {"code": "9", "districts": [...]},
    # etc.

}


class AdaptiveDelay:
    def __init__(self, base=0.4):
        self.delay = base
        self.base  = base

    def ok(self):
        self.delay = max(self.base, self.delay * 0.95)

    def error(self):
        self.delay = min(5.0, self.delay * 1.6)
        log.info(f"    Rate limit -> {self.delay:.1f}s")

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
                params={"state_code": state_code, "district_code": district_code},
                timeout=15
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.warning(f"    layers error: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def wms_get_features(session, wms_base, layer_name, minx, miny, maxx, maxy, throttle):
    params = {
        "service":       "WMS", "version": "1.1.1",
        "request":       "GetFeatureInfo", "format": "image/png",
        "transparent":   "true", "layers": layer_name,
        "query_layers":  layer_name,
        "exceptions":    "application/vnd.ogc.se_inimage",
        "srs":           "EPSG:4326", "width": "101", "height": "101",
        "X":             "50", "Y": "50", "feature_count": "500",
        "info_format":   "application/json",
        "bbox":          f"{minx},{miny},{maxx},{maxy}",
        "HIDE_GEOMETRY": "true",
    }
    for attempt in range(3):
        try:
            r = session.get(wms_base, params=params, timeout=25)
            if r.status_code == 200:
                throttle.ok()
                return [
                    {"id": f.get("id", ""), "props": f.get("properties", {})}
                    for f in r.json().get("features", [])
                    if f.get("properties")
                ]
            throttle.error()
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            throttle.error()
            wait = 3 * (attempt + 1)
            log.warning(f"    WMS error: {str(e)[:60]}")
            if "SSL" in str(e) or "EOF" in str(e) or "abort" in str(e).lower():
                session.close()
            time.sleep(wait)
    return []


def safe_float(val):
    try:
        v = float(str(val).strip())
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def collect_district(session, wms_base, state_code, district_code,
                     district_name, state_name, tile_size=0.03):
    throttle = AdaptiveDelay(base=0.4)
    info = get_district_info(session, state_code, district_code)
    if not info:
        return []

    bbox = info.get("bbox", {})
    minx, miny = bbox.get("minx"), bbox.get("miny")
    maxx, maxy = bbox.get("maxx"), bbox.get("maxy")
    if None in [minx, miny, maxx, maxy]:
        return []

    layers = info.get("shcLayers", [])
    if not layers:
        return []

    cycle_order = ["2025-26", "2024-25", "2023-24", "2022-23",
                   "2021-22", "2020-21", "2019-21", "2017-19", "2015-17"]
    sorted_layers = sorted(layers,
                           key=lambda x: cycle_order.index(x)
                           if x in cycle_order else 99)

    best_farms = {}
    best_cycle = None

    for layer_year in sorted_layers:
        layer_name = f"{state_code}_{district_code}_shc_{layer_year}"
        n_x = max(1, math.ceil((maxx - minx) / tile_size))
        n_y = max(1, math.ceil((maxy - miny) / tile_size))
        x_step = (maxx - minx) / n_x
        y_step = (maxy - miny) / n_y
        log.info(f"    Layer: {layer_name} | Grid: {n_x}x{n_y}")

        all_farms = {}
        for xi in range(n_x):
            for yi in range(n_y):
                tx0 = minx + xi * x_step
                ty0 = miny + yi * y_step
                features = wms_get_features(
                    session, wms_base, layer_name,
                    tx0, ty0, tx0 + x_step, ty0 + y_step, throttle
                )
                for f in features:
                    fid = f["id"]
                    if fid and fid not in all_farms:
                        all_farms[fid] = f["props"]
                throttle.wait()

        log.info(f"    Cycle {layer_year}: {len(all_farms)} farms")
        if len(all_farms) > len(best_farms):
            best_farms = dict(all_farms)
            best_cycle = layer_year
        if len(best_farms) >= 30:
            break
        time.sleep(1.0)

    if not best_farms:
        return []

    clim = STATE_CLIMATE.get(state_name, {"T": 25, "H": 70, "R": 100})
    rows = []
    for fid, props in best_farms.items():
        village = str(props.get("village", "")).strip().title() or "Unknown"
        rows.append({
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
        })

    log.info(f"    Collected {len(rows)} farm records for {district_name}")
    return rows


def save(rows, output):
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["state", "district", "village"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    log.info(f"Saved {len(df)} records -> {output}")


def get_done_districts(output):
    if Path(output).exists():
        try:
            df = pd.read_csv(output)
            done = set((df["district"] + "|" + df["state"]).str.upper().tolist())
            log.info(f"Resume: {len(done)} district-state combinations already done")
            return done
        except Exception:
            pass
    return set()


def scrape(wms_base, target_states=None, max_districts=None,
           output="data/shc_village_soil.csv", resume=False):

    session  = make_session()
    all_rows = []
    done     = set()

    if resume and Path(output).exists():
        try:
            existing = pd.read_csv(output)
            all_rows = existing.to_dict("records")
            done     = set((existing["district"] + "|" +
                            existing["state"]).str.upper().tolist())
            log.info(f"Loaded {len(all_rows)} existing records, "
                     f"{len(done)} districts done")
        except Exception as e:
            log.warning(f"Could not load existing: {e}")

    available_states = list(DISTRICTS.keys())
    if target_states:
        up     = [s.upper().strip() for s in target_states]
        states = [s for s in available_states if s.upper() in up]
        missing = [s for s in up if s not in [x.upper() for x in available_states]]
        if missing:
            log.warning(f"States not yet in district list: {missing}")
            log.warning("Collect district data from portal and add to DISTRICTS dict")
    else:
        states = available_states

    new_count = 0

    for si, sname in enumerate(states, 1):
        sinfo  = DISTRICTS[sname]
        scode  = sinfo["code"]
        dists  = sinfo["districts"]
        log.info(f"\n[{si}/{len(states)}] {sname} (code={scode}, "
                 f"{len(dists)} districts)")

        for di, dist in enumerate(dists, 1):
            if max_districts and new_count >= max_districts:
                log.info(f"Reached max {max_districts}. Stopping.")
                save(all_rows, output)
                return all_rows

            dname    = dist["name"]
            dcode    = dist["code"]
            done_key = f"{dname}|{sname}".upper()

            if resume and done_key in done:
                log.info(f"  [{di}/{len(dists)}] {dname} -- skipping")
                continue

            log.info(f"\n  [{di}/{len(dists)}] {dname} (code={dcode})")

            rows = collect_district(
                session, wms_base, scode, dcode, dname, sname,
                tile_size=0.03
            )

            if rows:
                all_rows.extend(rows)
                log.info(f"  Added {len(rows)} records from {dname}")
            else:
                log.warning(f"  No data for {dname}")

            new_count += 1
            save(all_rows, output)
            time.sleep(1.0)

        log.info(f"  {sname} done.")

    save(all_rows, output)
    total = len(pd.read_csv(output)) if Path(output).exists() else 0
    log.info(f"\nComplete. {total} total farm records.")
    return all_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wms-base",      default=WMS_BASE)
    ap.add_argument("--states",        default=None,
                    help='e.g. "MAHARASHTRA,PUNJAB"')
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
    print("  SHC Multi-State Village Collector")
    print(f"  Available states: {list(DISTRICTS.keys())}")
    print(f"  Running for:      {target or 'All available'}")
    print(f"  Max districts:    {args.max_districts or 'No limit'}")
    print(f"  Resume:           {args.resume}")
    print(f"  Output:           {args.output}")
    print("=" * 62)

    rows = scrape(wms, target, args.max_districts, args.output, args.resume)

    if rows:
        df = pd.read_csv(args.output)
        print(f"\n  Total: {len(df)} farm records")
        print(f"  States:    {df['state'].nunique()}")
        print(f"  Districts: {df['district'].nunique()}")
        print(f"  Villages:  {df['village'].nunique()}")
    else:
        print("\nNo data collected.")