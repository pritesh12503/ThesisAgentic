"""
SHC Scraper — Cookie Auth Version
====================================
The soilhealth4.dac.gov.in GraphQL API requires a browser session.
This script uses YOUR browser's cookie to authenticate.

HOW TO GET YOUR COOKIE (do this once):
  1. Open Chrome and go to: https://soilhealth.dac.gov.in/slusi-visualisation/
  2. Wait for the page to fully load (you should see the map with Chhattisgarh)
  3. Press F12 → Network tab → click Fetch/XHR
  4. Click on any "soilhealth4.dac.gov.in" request
  5. Click "Headers" tab
  6. Scroll down to "Request Headers"
  7. Find the "cookie" header — copy the ENTIRE value
  8. Paste it into the COOKIE variable below (between the quotes)

ALTERNATIVELY — use the Authorization header if there is one.

USAGE:
  1. Fill in COOKIE below
  2. python shc_cookie_scraper.py --states "CHHATTISGARH" --max-districts 5
"""

import requests, pandas as pd, time, random, re, logging, argparse, sys
from bs4 import BeautifulSoup
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("shc_scraper.log")]
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PASTE YOUR COOKIE HERE
# Get it from Chrome DevTools → Network → any soilhealth4.dac.gov.in request
# → Headers → Request Headers → cookie
# ══════════════════════════════════════════════════════════════════════════════
COOKIE = "PASTE_YOUR_COOKIE_HERE"

# ══════════════════════════════════════════════════════════════════════════════

API      = "https://soilhealth4.dac.gov.in/"
MAPS_URL = "https://soilhealth.dac.gov.in/legacy-maps/Cycle_II/{sname}/{scode}/{dname}/{dcode}"

GQL_STATES = {
    "operationName": "GetState",
    "variables": {},
    "query": "query GetState($getStateId: String, $code: String) { getState(id: $getStateId, code: $code) { _id name code } }"
}

GQL_DISTRICTS = {
    "operationName": "GetdistrictAndSubdistrictBystate",
    "query": "query GetdistrictAndSubdistrictBystate($state: String) { getdistrictAndSubdistrictBystate(state: $state) { _id name code state { _id name code } } }"
}

MIDPOINTS = {
    "N":  {"Low": 140,  "Medium": 420,  "High": 700},
    "P":  {"Low": 5,    "Medium": 17,   "High": 35},
    "K":  {"Low": 54,   "Medium": 194,  "High": 380},
    "pH": {"Acidic": 5.5, "Neutral": 7.0, "Alkaline": 8.2,
           "Low": 5.5,    "Medium": 7.0,  "High": 8.2},
    "OC": {"Low": 0.35, "Medium": 0.625, "High": 0.90},
}

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

# Hardcoded state list (from the JSON you already got)
STATES_HARDCODED = [
    {"_id": "63f9ce47519359b7438e76fa", "name": "ANDAMAN & NICOBAR",    "code": "35"},
    {"_id": "63f957b089d86ca9e2c00e14", "name": "ANDHRA PRADESH",       "code": "28"},
    {"_id": "63f86a1cd2e83876f03e7d32", "name": "ARUNACHAL PRADESH",    "code": "12"},
    {"_id": "63f87b48d2e83876f040050d", "name": "ASSAM",                "code": "18"},
    {"_id": "63f86570d2e83876f03e015b", "name": "BIHAR",                "code": "10"},
    {"_id": "63f8e35adbc0f7fe06708342", "name": "CHHATTISGARH",         "code": "22"},
    {"_id": "63f878d9c660ddb223461f8c", "name": "DADRA & NAGAR HAVELI AND DAMAN & DIU", "code": "38"},
    {"_id": "63f5ceb98cec41e6c95ce3a8", "name": "DELHI",               "code": "7"},
    {"_id": "63f9bcdc519359b7438cdeb3", "name": "GOA",                  "code": "30"},
    {"_id": "63f922a6dbc0f7fe06751641", "name": "GUJARAT",              "code": "24"},
    {"_id": "63f5c2cf98d5e0c03dba5507", "name": "HARYANA",              "code": "6"},
    {"_id": "63f21f6089e2887695740385", "name": "HIMACHAL PRADESH",     "code": "2"},
    {"_id": "63f20ca84fadd776ffd9d608", "name": "JAMMU & KASHMIR",      "code": "1"},
    {"_id": "63f893d236497c055ce2e09a", "name": "JHARKHAND",            "code": "20"},
    {"_id": "63f99fbd519359b7438a84ca", "name": "KARNATAKA",            "code": "29"},
    {"_id": "63f9bd39519359b7438ce777", "name": "KERALA",               "code": "32"},
    {"_id": "63f878a8c660ddb223461a90", "name": "LADAKH",               "code": "37"},
    {"_id": "63f8e729dbc0f7fe0670d050", "name": "MADHYA PRADESH",       "code": "23"},
    {"_id": "63f9322a89d86ca9e2bca5df", "name": "MAHARASHTRA",          "code": "27"},
    {"_id": "63f870acd2e83876f03f0b46", "name": "MANIPUR",              "code": "14"},
    {"_id": "63f87562d2e83876f03f7a2c", "name": "MEGHALAYA",            "code": "17"},
    {"_id": "63f873e5d2e83876f03f573a", "name": "MIZORAM",              "code": "15"},
    {"_id": "63f86f4dd2e83876f03eea29", "name": "NAGALAND",             "code": "13"},
    {"_id": "63f8bb7b36497c055ce5624b", "name": "ODISHA",               "code": "21"},
    {"_id": "63f9ce2c519359b7438e746d", "name": "PUDUCHERRY",           "code": "34"},
    {"_id": "63f2495789e288769575a424", "name": "PUNJAB",               "code": "3"},
    {"_id": "63f5cef58cec41e6c95ce84e", "name": "RAJASTHAN",            "code": "8"},
    {"_id": "63f869bfd2e83876f03e7396", "name": "SIKKIM",               "code": "11"},
    {"_id": "63f9be9f519359b7438d08bb", "name": "TAMIL NADU",           "code": "33"},
    {"_id": "63f871f5c660ddb223457dca", "name": "TELANGANA",            "code": "36"},
    {"_id": "63f874a0d2e83876f03f6866", "name": "TRIPURA",              "code": "16"},
    {"_id": "63f600f38cec41e6c9607e6b", "name": "UTTAR PRADESH",        "code": "9"},
    {"_id": "63f37f20491ca1c6048696ce", "name": "UTTARAKHAND",          "code": "5"},
    {"_id": "63f87a1836497c055ce09df5", "name": "WEST BENGAL",          "code": "19"},
]

# Chhattisgarh districts (from the JSON you already got)
CG_DISTRICTS_HARDCODED = [
    {"_id": "63f91dff76742aaa3ce749c0", "name": "BALOD",           "code": "646"},
    {"_id": "63f91c6676742aaa3ce72cd9", "name": "BALODA BAZAR",    "code": "644"},
    {"_id": "63f9104676742aaa3ce62e10", "name": "BASTAR",          "code": "374"},
    {"_id": "63f8e3bfdbc0f7fe06708b20", "name": "BEMETARA",        "code": "650"},
    {"_id": "63f91a2076742aaa3ce7034f", "name": "BIJAPUR",         "code": "636"},
    {"_id": "641dacf9523a9a8b4d9c8f8e", "name": "BALRAMPUR",       "code": "649"},
    {"_id": "641daed5523a9a8b4d9cac0d", "name": "BILASPUR",        "code": "375"},
    {"_id": "63f9115c76742aaa3ce647e7", "name": "DANTEWADA",       "code": "376"},
    {"_id": "63f9118c76742aaa3ce64c9b", "name": "DHAMTARI",        "code": "377"},
    {"_id": "63f9121976742aaa3ce659b0", "name": "DURG",            "code": "378"},
    {"_id": "63f91d2476742aaa3ce73bb6", "name": "GARIYABAND",      "code": "645"},
    {"_id": "63f8e470dbc0f7fe06709924", "name": "GAURELLA PENDRA MARWAHI", "code": "734"},
    {"_id": "63f9127d76742aaa3ce66267", "name": "JANJGIR-CHAMPA",  "code": "379"},
    {"_id": "63f912e876742aaa3ce66ada", "name": "JASHPUR",         "code": "380"},
    {"_id": "63f914b576742aaa3ce68ef3", "name": "KABIRDHAM",       "code": "382"},
    {"_id": "63f913a676742aaa3ce679b7", "name": "KANKER",          "code": "381"},
    {"_id": "63f8e4a6dbc0f7fe06709d84", "name": "KHAIRGARH CHHUIKHADAN GANDAI", "code": "759"},
    {"_id": "63f91bbc76742aaa3ce7218d", "name": "KONDAGAON",       "code": "643"},
    {"_id": "63f915ca76742aaa3ce6a2eb", "name": "KORBA",           "code": "383"},
    {"_id": "63f9168576742aaa3ce6b116", "name": "KOREA",           "code": "384"},
    {"_id": "63f916c576742aaa3ce6b6b6", "name": "MAHASAMUND",      "code": "385"},
    {"_id": "63f8e520dbc0f7fe0670a734", "name": "MANENDRAGARH CHIRIMIRI BHARATPUR", "code": "760"},
    {"_id": "63f8e57fdbc0f7fe0670ae93", "name": "MOHLA MANPUR AMBAGARH CHOUKI", "code": "761"},
    {"_id": "63f91ed476742aaa3ce7579f", "name": "MUNGELI",         "code": "647"},
    {"_id": "63f91aea76742aaa3ce71131", "name": "NARAYANPUR",      "code": "637"},
    {"_id": "63f917bf76742aaa3ce6cd8d", "name": "RAIGARH",         "code": "386"},
    {"_id": "63f9188d76742aaa3ce6e061", "name": "RAIPUR",          "code": "387"},
    {"_id": "63f918fa76742aaa3ce6ea21", "name": "RAJNANDGAON",     "code": "388"},
    {"_id": "63f8e5f8dbc0f7fe0670b863", "name": "SAKTI",           "code": "762"},
    {"_id": "63f8e66cdbc0f7fe0670c177", "name": "SARANGARH BILAIGARH", "code": "763"},
    {"_id": "63f91b5176742aaa3ce71993", "name": "SUKMA",           "code": "642"},
    {"_id": "63f9200276742aaa3ce7659d", "name": "SURAJPUR",        "code": "648"},
    {"_id": "63f9199776742aaa3ce6f7c5", "name": "SURGUJA",         "code": "389"},
]


def make_session(cookie=None):
    s = requests.Session()
    headers = {
        "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
        "Accept":       "application/json, */*",
        "Referer":      "https://soilhealth.dac.gov.in/",
        "Origin":       "https://soilhealth.dac.gov.in",
        "Content-Type": "application/json",
    }
    if cookie and cookie != "PASTE_YOUR_COOKIE_HERE":
        headers["Cookie"] = cookie
    s.headers.update(headers)
    return s


def gql(session, payload, retries=3):
    for attempt in range(1, retries + 1):
        try:
            r = session.post(API, json=payload, timeout=20)
            if r.status_code == 200:
                return r.json().get("data")
            log.warning(f"HTTP {r.status_code} attempt {attempt}: {r.text[:150]}")
            time.sleep(2 * attempt)
        except Exception as e:
            log.warning(f"Error attempt {attempt}: {e}")
            time.sleep(2 * attempt)
    return None


def get_districts_api(session, state_id, state_name):
    payload = {
        "operationName": GQL_DISTRICTS["operationName"],
        "variables":     {"state": state_id},
        "query":         GQL_DISTRICTS["query"],
    }
    data = gql(session, payload)
    if not data:
        return []
    return data.get("getdistrictAndSubdistrictBystate", [])


def fetch_map_page(session, sname, scode, dname, dcode):
    url = MAPS_URL.format(
        sname=sname.replace(" ", "%20"),
        scode=scode,
        dname=dname.replace(" ", "%20"),
        dcode=dcode
    )
    log.info(f"    GET {url}")
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200:
            return r.text
        log.warning(f"    HTTP {r.status_code}")
    except Exception as e:
        log.warning(f"    Error: {e}")
    return None


def parse_nutrients(html):
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    result = {}

    nutrient_keywords = {
        "N":  ["available nitrogen", "nitrogen"],
        "P":  ["available phosphorus", "phosphorous", "phosphorus"],
        "K":  ["available potassium", "potassium"],
        "pH": ["soil ph", "ph value", "reaction"],
        "OC": ["organic carbon"],
    }
    categories = ["Low", "Medium", "High", "Acidic", "Neutral", "Alkaline"]

    for nutrient, keywords in nutrient_keywords.items():
        start = None
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in keywords):
                start = i
                break
        if start is None:
            continue

        section = "\n".join(lines[start: start + 30])
        pct = {}
        for cat in categories:
            pat = rf"(?i)\b{cat}\b[^%]{{0,100}}?\((\d+\.?\d*)\s*%\)"
            m = re.search(pat, section)
            if m:
                try:
                    pct[cat] = float(m.group(1))
                except ValueError:
                    pass

        if len(pct) < 2:
            continue

        mp    = MIDPOINTS.get(nutrient, {})
        total = sum(pct.values())
        if total == 0:
            continue
        val = round(sum((v / total) * mp.get(k, 0)
                        for k, v in pct.items() if k in mp), 1)
        if val > 0:
            result[nutrient] = val
            log.info(f"      {nutrient}: {pct} -> {val}")

    return result


def save(rows, output):
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["district", "state"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    log.info(f"Saved {len(df)} rows -> {output}")


def scrape(target_states=None, max_districts=None,
           delay=2.0, output="data/shc_district_soil.csv",
           cookie=None):

    session = make_session(cookie)
    rows    = []
    count   = 0

    # Use hardcoded states (already have them from the JSON you got)
    states = STATES_HARDCODED
    if target_states:
        up     = [s.upper().strip() for s in target_states]
        states = [s for s in states if s["name"].upper() in up]
        log.info(f"Filtered to: {[s['name'] for s in states]}")

    for si, state in enumerate(states, 1):
        sid   = state["_id"]
        sname = state["name"]
        scode = state["code"]
        clim  = STATE_CLIMATE.get(sname, {"T": 25, "H": 70, "R": 100})
        fb    = STATE_NPK_FALLBACK.get(sname, {"N": 40, "P": 18, "K": 190, "pH": 7.2})

        log.info(f"\n[{si}/{len(states)}] {sname} (code={scode})")

        # Use hardcoded CG districts if available, else try API
        if sname == "CHHATTISGARH":
            districts = CG_DISTRICTS_HARDCODED
            log.info(f"  Using hardcoded {len(districts)} districts for CG")
        else:
            districts = get_districts_api(session, sid, sname)
            if not districts:
                log.warning(f"  No districts — skipping {sname}")
                continue

        time.sleep(delay)

        for di, dist in enumerate(districts, 1):
            if max_districts and count >= max_districts:
                log.info(f"Reached max {max_districts}. Stopping.")
                save(rows, output)
                return rows

            dname = dist["name"]
            dcode = dist["code"]
            log.info(f"  [{di}/{len(districts)}] {dname} (code={dcode})")

            html      = fetch_map_page(session, sname, scode, dname, dcode)
            nutrients = parse_nutrients(html)

            if len(nutrients) >= 2:
                source = "SHC_Portal"
                log.info(f"    OK: {nutrients}")
            else:
                nutrients = fb.copy()
                source    = "ICAR_State"
                log.info(f"    Fallback used for {dname}")

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
                "district_code":    dcode,
                "state_code":       scode,
            })
            count += 1
            time.sleep(delay + random.uniform(0.5, 1.5))

        save(rows, output)
        log.info(f"  State done. Total: {count}")
        time.sleep(delay * 2)

    save(rows, output)
    log.info(f"\nDone. {count} districts scraped.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--states",        default=None)
    ap.add_argument("--max-districts", type=int,   default=None)
    ap.add_argument("--delay",         type=float, default=2.0)
    ap.add_argument("--output",        default="data/shc_district_soil.csv")
    ap.add_argument("--cookie",        default=None,
                    help="Browser cookie string (optional, for API auth)")
    args = ap.parse_args()

    cookie = args.cookie or (COOKIE if COOKIE != "PASTE_YOUR_COOKIE_HERE" else None)
    target = [s.strip() for s in args.states.split(",")] if args.states else None

    print("=" * 60)
    print("  SHC Scraper — Hardcoded State/District IDs + Legacy Maps")
    print(f"  States:        {target or 'All'}")
    print(f"  Max districts: {args.max_districts or 'No limit'}")
    print(f"  Cookie:        {'Provided' if cookie else 'Not provided (API calls may fail)'}")
    print(f"  Output:        {args.output}")
    print("=" * 60)
    print()
    print("NOTE: The legacy-maps pages do NOT need a cookie.")
    print("      Only the district API calls need it.")
    print("      For Chhattisgarh, hardcoded district list is used.")
    print()

    rows = scrape(target, args.max_districts, args.delay, args.output, cookie)

    if rows:
        df = pd.read_csv(args.output)
        print(f"\n  {len(df)} districts saved to {args.output}")
        print(df.head(10).to_string(index=False))
    else:
        print("\nNo data collected. Check shc_scraper.log")