"""
SHC Portal Inspector
=====================
Run this FIRST before the scraper.
It opens the portal in Chrome, waits for the page to fully load,
then prints ALL dropdown/select elements it finds on the page.

This tells us the EXACT dropdown IDs to use in the scraper.

USAGE:
  python shc_inspector.py

It will print something like:
  SELECT elements found:
    ID: "ddlState"     Name: "State"     Options: 28
    ID: "ddlDistrict"  Name: "District"  Options: 0

Then paste those IDs into shc_scraper.py
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

# ── Pages to try ───────────────────────────────────────────────────────────────
PAGES_TO_TRY = [
    "https://soilhealth.dac.gov.in/publicreports/NutrientStatusReport",
    "https://soilhealth.dac.gov.in/publicreports",
    "https://soilhealth.dac.gov.in/publicreports/stateDistrictWiseReport",
    "https://soilhealth.dac.gov.in/publicreports/NutrientStatusDistrictWise",
    "https://soilhealth.dac.gov.in",
]

def inspect_page(driver, url):
    print(f"\n{'='*60}")
    print(f"  URL: {url}")
    print(f"{'='*60}")

    driver.get(url)
    time.sleep(5)  # wait for JS to render

    print(f"  Page title: {driver.title}")
    print(f"  Current URL: {driver.current_url}")

    # ── Find all SELECT dropdowns ──────────────────────────────────────────────
    selects = driver.find_elements(By.TAG_NAME, "select")
    print(f"\n  SELECT dropdowns found: {len(selects)}")
    for s in selects:
        sid   = s.get_attribute("id")   or "(no id)"
        sname = s.get_attribute("name") or "(no name)"
        sclass= s.get_attribute("class") or "(no class)"
        opts  = s.find_elements(By.TAG_NAME, "option")
        opt_texts = [o.text.strip() for o in opts[:5]]  # first 5 options
        print(f"    ID={sid!r:25}  name={sname!r:20}  options={len(opts)}  first={opt_texts}")

    # ── Find all INPUT elements ────────────────────────────────────────────────
    inputs = driver.find_elements(By.TAG_NAME, "input")
    print(f"\n  INPUT elements found: {len(inputs)}")
    for inp in inputs:
        iid   = inp.get_attribute("id")   or "(no id)"
        iname = inp.get_attribute("name") or "(no name)"
        itype = inp.get_attribute("type") or "text"
        iph   = inp.get_attribute("placeholder") or ""
        if itype not in ("hidden", "submit"):
            print(f"    ID={iid!r:25}  name={iname!r:20}  type={itype}  placeholder={iph!r}")

    # ── Find all BUTTON elements ───────────────────────────────────────────────
    buttons = driver.find_elements(By.TAG_NAME, "button")
    print(f"\n  BUTTON elements found: {len(buttons)}")
    for btn in buttons:
        bid  = btn.get_attribute("id") or "(no id)"
        btxt = btn.text.strip() or "(no text)"
        print(f"    ID={bid!r:25}  text={btxt!r}")

    # ── Find all TABLE elements ────────────────────────────────────────────────
    tables = driver.find_elements(By.TAG_NAME, "table")
    print(f"\n  TABLE elements found: {len(tables)}")
    for i, t in enumerate(tables):
        rows = t.find_elements(By.TAG_NAME, "tr")
        print(f"    Table {i}: {len(rows)} rows")
        if rows:
            # Print first row headers
            first_row = rows[0].find_elements(By.TAG_NAME, "th")
            if not first_row:
                first_row = rows[0].find_elements(By.TAG_NAME, "td")
            headers = [h.text.strip() for h in first_row]
            print(f"      Headers: {headers}")

    # ── Find React/Angular rendered elements with data-* attributes ────────────
    print(f"\n  Elements with 'state' in ID or class:")
    try:
        state_els = driver.find_elements(
            By.XPATH,
            "//*[contains(translate(@id,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'state') or "
            "contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'state')]"
        )
        for el in state_els[:10]:
            tag   = el.tag_name
            eid   = el.get_attribute("id")   or "(no id)"
            ecls  = el.get_attribute("class") or "(no class)"
            etxt  = el.text.strip()[:50] if el.text else ""
            print(f"    <{tag}>  id={eid!r}  class={repr(ecls)[:40]}  text={etxt!r}")
    except Exception as e:
        print(f"    Error: {e}")

    # ── Save full page source ──────────────────────────────────────────────────
    fname = f"debug_{url.split('/')[-1] or 'home'}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print(f"\n  Full page source saved to: {fname}")

    return len(selects) > 0


def main():
    print("SHC Portal Inspector")
    print("Opening Chrome — do NOT close it")
    print()

    options = Options()
    options.add_argument("--window-size=1280,900")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    found = False
    try:
        for url in PAGES_TO_TRY:
            has_dropdowns = inspect_page(driver, url)
            if has_dropdowns:
                print(f"\n✓ Found dropdowns at: {url}")
                found = True
                # Stay on this page so you can inspect it visually
                print("\nBrowser staying open for 30 seconds so you can look at the page...")
                print("Take a screenshot or note what dropdowns you see.")
                time.sleep(30)
                break
            time.sleep(2)

        if not found:
            print("\n⚠ No SELECT dropdowns found on any page.")
            print("The portal may use custom React components instead of <select>.")
            print("\nCheck the saved debug_*.html files for the page structure.")
            print("Also check browser DevTools → Network tab for API calls.")
            time.sleep(15)

    finally:
        driver.quit()
        print("\nDone. Check the output above and the saved HTML files.")
        print("\nNEXT STEP:")
        print("  Send the output above (copy-paste) and I will fix the scraper.")


if __name__ == "__main__":
    main()
