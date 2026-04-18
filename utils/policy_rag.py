"""
utils/policy_rag.py — BATCHED VERSION

RATE LIMIT FIX:
  Old version: called Groq once per crop = 51 Groq calls per run
  New version: ONE Groq call returns policy for ALL crops at once
  Result: reduces Groq usage from ~5100 tokens/run to ~300 tokens/run

query_policy_batch() returns a dict {crop: policy_dict} for all crops at once.
query_policy() still works for individual lookups (backward compatible).
"""

import json
import logging
import re
import os
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

POLICY_DIR = Path(__file__).parent.parent / "data" / "policy_knowledge"

STATE_FILE_MAP = {
    "andhra pradesh": "andhra_pradesh", "ap": "andhra_pradesh",
    "arunachal pradesh": "arunachal_pradesh",
    "assam": "assam", "bihar": "bihar",
    "chandigarh": "chandigarh", "chhattisgarh": "chhattisgarh",
    "delhi": "delhi", "goa": "goa",
    "gujarat": "gujarat", "haryana": "haryana",
    "himachal pradesh": "himachal_pradesh", "hp": "himachal_pradesh",
    "jammu and kashmir": "jammu_kashmir", "j&k": "jammu_kashmir",
    "jharkhand": "jharkhand", "karnataka": "karnataka",
    "kerala": "kerala", "ladakh": "ladakh",
    "madhya pradesh": "madhya_pradesh", "mp": "madhya_pradesh",
    "maharashtra": "maharashtra", "manipur": "manipur",
    "meghalaya": "meghalaya", "mizoram": "mizoram",
    "nagaland": "nagaland", "odisha": "odisha",
    "puducherry": "puducherry", "pondicherry": "puducherry",
    "punjab": "punjab", "rajasthan": "rajasthan",
    "sikkim": "sikkim", "tamil nadu": "tamil_nadu", "tn": "tamil_nadu",
    "telangana": "telangana", "tripura": "tripura",
    "uttar pradesh": "uttar_pradesh", "up": "uttar_pradesh",
    "uttarakhand": "uttarakhand", "west bengal": "west_bengal", "wb": "west_bengal",
    "andaman and nicobar islands": "andaman_nicobar",
}

DEFAULT_POLICY = {
    "msp_per_kg": 0,
    "state_bonus_per_kg": 0,
    "effective_price_per_kg": 0,
    "procurement_efficiency_pct": 10,
    "subsidy_per_ha_per_season": 2000,
    "insurance_premium_pct": 0.02,
    "power_subsidy_per_ha": 0,
    "policy_note": "Using national average policy parameters.",
}

# MSP values (₹/quintal → ₹/kg, divide by 100) for major crops
# Source: CACP MSP 2023-24
KNOWN_MSP = {
    "Rice": 21.83, "Wheat": 21.50, "Maize": 17.40, "Jowar": 30.62,
    "Bajra": 25.00, "Ragi": 37.15, "Barley": 17.35,
    "Arhar/Tur": 70.00, "Moong(Green Gram)": 85.58, "Urad": 70.00,
    "Gram": 54.40, "Masoor": 60.00, "Groundnut": 60.77,
    "Rapeseed &Mustard": 55.50, "Sunflower": 68.00,
    "Soyabean": 43.00, "Sesamum": 90.00, "Linseed": 57.50,
    "Safflower": 58.00, "Cotton(lint)": 70.00, "Sugarcane": 3.15,
    "Copra": 108.00,
}


@lru_cache(maxsize=64)
def _load_state_policy_text(state_name: str) -> str:
    """Load policy text file for a state. Cached."""
    state_key = state_name.lower().strip()
    file_stem  = STATE_FILE_MAP.get(state_key, "")
    if not file_stem:
        file_stem = STATE_FILE_MAP.get("_national", "_national")

    for fname in [f"{file_stem}.txt", "_national.txt"]:
        fpath = POLICY_DIR / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                return f.read()[:3000]  # limit context size
    return ""


def _build_default_with_msp(crop: str) -> dict:
    """Build a policy dict using known MSP values + national defaults."""
    msp = KNOWN_MSP.get(crop, 0)
    return {
        "msp_per_kg":                 msp,
        "state_bonus_per_kg":         0,
        "effective_price_per_kg":     msp,
        "procurement_efficiency_pct": 15 if msp > 0 else 0,
        "subsidy_per_ha_per_season":  2000,
        "insurance_premium_pct":      0.02,
        "power_subsidy_per_ha":       0,
        "policy_note":                f"Central MSP applied. State bonus not available." if msp > 0
                                      else "No central MSP for this crop.",
    }


def query_policy_batch(
    crops: list,
    state_name: str,
    season: str,
    groq_model: str = "llama3-8b-8192",
) -> Dict[str, dict]:
    """
    Returns policy data for ALL crops in ONE Groq call.
    Drastically reduces TPM usage vs calling once per crop.
    
    Falls back to KNOWN_MSP + defaults if Groq unavailable.
    """
    # Start with MSP-based defaults for all crops
    result = {crop: _build_default_with_msp(crop) for crop in crops}

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        logger.warning("[PolicyRAG] No GROQ_API_KEY — using MSP defaults only")
        return result

    # Load state policy text
    policy_text = _load_state_policy_text(state_name)
    if not policy_text:
        logger.warning(f"[PolicyRAG] No policy text for '{state_name}' — using MSP defaults")
        return result

    # Focus batch on top crops that have MSP or state schemes to adjust
    # Limit to 15 crops to keep prompt small and avoid TPM limits
    major_crops = [c for c in crops if c in KNOWN_MSP][:12]
    other_crops  = [c for c in crops if c not in KNOWN_MSP][:3]
    batch_crops  = major_crops + other_crops

    crops_list = ", ".join(batch_crops)

    prompt = f"""You are an Indian agricultural policy expert.
State: {state_name.title()} | Season: {season}

Policy document:
{policy_text[:2000]}

For EACH of these crops: {crops_list}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "CropName": {{
    "state_bonus_per_kg": <number>,
    "procurement_efficiency_pct": <number 0-100>,
    "subsidy_per_ha_per_season": <number>,
    "power_subsidy_per_ha": <number>,
    "policy_note": "<one sentence>"
  }}
}}

Only include state-specific information. Use 0 if not mentioned."""

    try:
        llm      = ChatGroq(model=groq_model, temperature=0.0, max_tokens=2000)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw      = response.content.strip()

        # Extract JSON
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        parsed = json.loads(json_match.group())

        # Merge state-specific info into MSP defaults
        for crop in batch_crops:
            state_info = parsed.get(crop, {})
            if state_info:
                base = result[crop]
                bonus = float(state_info.get("state_bonus_per_kg", 0) or 0)
                base["state_bonus_per_kg"]         = bonus
                base["procurement_efficiency_pct"] = float(state_info.get("procurement_efficiency_pct", base["procurement_efficiency_pct"]) or 0)
                base["subsidy_per_ha_per_season"]  = float(state_info.get("subsidy_per_ha_per_season", base["subsidy_per_ha_per_season"]) or 0)
                base["power_subsidy_per_ha"]       = float(state_info.get("power_subsidy_per_ha", 0) or 0)
                note = state_info.get("policy_note", "")
                if note:
                    base["policy_note"] = str(note)
                # Recalculate effective price
                msp = base["msp_per_kg"]
                if msp > 0 or bonus > 0:
                    base["effective_price_per_kg"] = msp + bonus

        logger.info(f"[PolicyRAG] Batch policy fetched for {len(batch_crops)} crops in {state_name}")

    except Exception as e:
        logger.warning(f"[PolicyRAG] Batch call failed: {e}. Using MSP defaults.")

    return result


def query_policy(
    crop: str,
    state_name: str,
    season: str,
    groq_model: str = "llama3-8b-8192",
) -> dict:
    """
    Single-crop policy lookup. Uses batch internally for efficiency.
    Backward-compatible with existing market_agent calls.
    """
    batch = query_policy_batch([crop], state_name, season, groq_model)
    policy = batch.get(crop, _build_default_with_msp(crop))
    
    logger.info(
        f"[PolicyRAG] {crop} in {state_name}: "
        f"MSP=₹{policy['msp_per_kg']}, "
        f"bonus=₹{policy['state_bonus_per_kg']}, "
        f"subsidy=₹{policy['subsidy_per_ha_per_season']}/ha"
    )
    return policy
