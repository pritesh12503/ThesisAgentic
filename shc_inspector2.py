"""
SHC Portal Inspector v2 — React-aware
=======================================
The portal is a React app. This script:
1. Waits for React to fully render (up to 30 seconds)
2. Intercepts ALL network API calls the browser makes
3. Prints every XHR/fetch request so we can see the real API endpoints

USAGE:
  python shc_inspector2.py

This will tell us:
  - The real API endpoints (e.g. /api/getStates)
  - The exact request format (POST vs GET, parameters)
  - The exact response structure (JSON fields)

Then we can call those APIs directly without Selenium.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

TARGET_URL = "https://soilhealth.dac.gov.in/publicreports/NutrientStatusReport"

def main():
    print("SHC Portal Inspector v2")
    print("Opening Chrome with network logging enabled...")
    print()

    # Enable performance logging to capture network requests
    options = Options()
    options.add_argument("--window-size=1400,900")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    # Enable network logging
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    try:
        print(f"Opening: {TARGET_URL}")
        driver.get(TARGET_URL)

        print("Waiting 15 seconds for React to fully render...")
        print("(watch the browser — the page should show dropdowns)")
        time.sleep(15)

        print(f"\nPage title: {driver.title}")
        print(f"Current URL: {driver.current_url}")

        # ── Try to find React-rendered dropdowns ───────────────────────────────
        print("\n--- Searching for dropdown elements (all types) ---")

        # React apps often use div/ul with role="listbox" or custom components
        search_strategies = [
            ("select",                   By.TAG_NAME,   "select"),
            ("role=combobox",            By.CSS_SELECTOR, "[role='combobox']"),
            ("role=listbox",             By.CSS_SELECTOR, "[role='listbox']"),
            ("role=option",              By.CSS_SELECTOR, "[role='option']"),
            ("class~=select",            By.CSS_SELECTOR, "[class*='select']"),
            ("class~=dropdown",          By.CSS_SELECTOR, "[class*='dropdown']"),
            ("class~=Select",            By.CSS_SELECTOR, "[class*='Select']"),
            ("class~=react-select",      By.CSS_SELECTOR, "[class*='react-select']"),
            ("class~=MuiSelect",         By.CSS_SELECTOR, "[class*='MuiSelect']"),
            ("class~=ant-select",        By.CSS_SELECTOR, "[class*='ant-select']"),
            ("placeholder~=State",       By.CSS_SELECTOR, "[placeholder*='State']"),
            ("placeholder~=District",    By.CSS_SELECTOR, "[placeholder*='District']"),
            ("aria-label~=state",        By.CSS_SELECTOR, "[aria-label*='tate']"),
        ]

        for label, by, selector in search_strategies:
            els = driver.find_elements(by, selector)
            if els:
                print(f"\n  FOUND [{label}]: {len(els)} elements")
                for el in els[:4]:
                    print(f"    tag={el.tag_name!r}  "
                          f"id={el.get_attribute('id')!r}  "
                          f"class={str(el.get_attribute('class') or '')[:50]!r}  "
                          f"text={el.text[:40]!r}")

        # ── Print ALL elements with IDs ────────────────────────────────────────
        print("\n--- All elements with IDs ---")
        all_with_id = driver.find_elements(By.CSS_SELECTOR, "[id]")
        print(f"  Total elements with id: {len(all_with_id)}")
        for el in all_with_id[:40]:
            eid  = el.get_attribute("id")
            etag = el.tag_name
            ecls = str(el.get_attribute("class") or "")[:40]
            etxt = el.text.strip()[:30] if el.text else ""
            print(f"    <{etag:10}> id={eid!r:35} class={ecls!r:42} text={etxt!r}")

        # ── Capture network calls ──────────────────────────────────────────────
        print("\n--- Network API calls captured ---")
        logs = driver.get_log("performance")
        api_calls = []
        for entry in logs:
            try:
                msg = json.loads(entry["message"])["message"]
                if msg["method"] == "Network.requestWillBeSent":
                    req = msg["params"]["request"]
                    url = req["url"]
                    method = req["method"]
                    # Only show API/XHR calls, not static assets
                    if any(x in url for x in [
                        "soilhealth", "api", "data", "report",
                        "state", "district", "nutrient"
                    ]) and not any(x in url for x in [
                        ".js", ".css", ".png", ".jpg", ".ico",
                        "fonts", "static", "chunk"
                    ]):
                        api_calls.append((method, url))
                        print(f"  {method:6} {url}")
            except Exception:
                pass

        if not api_calls:
            print("  No API calls captured yet.")
            print("  Trying to interact with the page to trigger API calls...")

            # Try clicking something to trigger state load
            try:
                # Look for any clickable element mentioning state
                clickable = driver.find_elements(
                    By.XPATH,
                    "//*[contains(text(),'State') or contains(text(),'Select')]"
                )
                if clickable:
                    print(f"  Clicking element: {clickable[0].text!r}")
                    clickable[0].click()
                    time.sleep(3)

                    # Re-capture logs
                    logs2 = driver.get_log("performance")
                    for entry in logs2:
                        try:
                            msg = json.loads(entry["message"])["message"]
                            if msg["method"] == "Network.requestWillBeSent":
                                req = msg["params"]["request"]
                                url = req["url"]
                                method = req["method"]
                                if "soilhealth" in url and ".js" not in url:
                                    print(f"  {method:6} {url}")
                        except Exception:
                            pass
            except Exception as e:
                print(f"  Click attempt failed: {e}")

        # ── Save full rendered HTML ────────────────────────────────────────────
        fname = "debug_rendered.html"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print(f"\nFull rendered HTML saved to: {fname}")
        print(f"(This is what React rendered — open in browser to see)")

        # ── Stay open for manual inspection ───────────────────────────────────
        print("\n" + "="*60)
        print("Browser staying open for 60 seconds.")
        print("RIGHT NOW — do this in the browser:")
        print("  1. Press F12 to open DevTools")
        print("  2. Click the 'Network' tab")
        print("  3. Click on the State dropdown and select any state")
        print("  4. Watch what API calls appear in the Network tab")
        print("  5. Copy-paste the URLs here")
        print("="*60)
        time.sleep(60)

    finally:
        driver.quit()
        print("\nInspector done.")


if __name__ == "__main__":
    main()
