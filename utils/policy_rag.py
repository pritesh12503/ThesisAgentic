"""
utils/policy_rag.py

Policy RAG (Retrieval-Augmented Generation) for agricultural policy.

Architecture:
  - One text file per state in data/policy_knowledge/
  - _national.txt covers central MSP for all crops
  - Each state file covers: state bonus, procurement efficiency,
    state-specific schemes, PMFBY, input subsidies

  At runtime:
  1. Load national + state-specific file
  2. Chunk into paragraphs
  3. Embed with sentence-transformers (free, local, no API)
  4. Query via FAISS
  5. Pass retrieved context to Groq
  6. Groq returns structured JSON with exactly 5 numbers:
       msp_per_kg, state_bonus_per_kg, effective_price_per_kg,
       subsidy_per_ha_per_season, insurance_premium_pct

These 5 numbers flow directly into the profit formula in market_agent.py.

IMPORTANT: This module uses HuggingFace sentence-transformers for embeddings.
No OpenAI API key required. Model: all-MiniLM-L6-v2 (80MB, downloads once).
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Optional
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

POLICY_DIR = Path(__file__).parent.parent / "data" / "policy_knowledge"

# State name → filename mapping
STATE_FILE_MAP = {
    # ── Major agricultural states ─────────────────────────────────────────────
    "punjab":                    "punjab.txt",
    "haryana":                   "haryana.txt",
    "maharashtra":               "maharashtra.txt",
    "karnataka":                 "karnataka.txt",
    "tamil nadu":                "tamil_nadu.txt",
    "tamilnadu":                 "tamil_nadu.txt",
    "telangana":                 "telangana.txt",
    "andhra pradesh":            "andhra_pradesh.txt",
    "andhrapradesh":             "andhra_pradesh.txt",
    "kerala":                    "kerala.txt",
    "gujarat":                   "gujarat.txt",
    "madhya pradesh":            "madhya_pradesh.txt",
    "madhyapradesh":             "madhya_pradesh.txt",
    "uttar pradesh":             "uttar_pradesh.txt",
    "uttarpradesh":              "uttar_pradesh.txt",
    "rajasthan":                 "rajasthan.txt",
    "odisha":                    "odisha.txt",
    "orissa":                    "odisha.txt",
    "chhattisgarh":              "chhattisgarh.txt",
    "west bengal":               "west_bengal.txt",
    "westbengal":                "west_bengal.txt",
    "bihar":                     "bihar.txt",
    # ── Remaining 12 states ───────────────────────────────────────────────────
    "uttarakhand":               "uttarakhand.txt",
    "himachal pradesh":          "himachal_pradesh.txt",
    "himachalpradesh":           "himachal_pradesh.txt",
    "hp":                        "himachal_pradesh.txt",
    "jharkhand":                 "jharkhand.txt",
    "assam":                     "assam.txt",
    "goa":                       "goa.txt",
    "manipur":                   "manipur.txt",
    "meghalaya":                 "meghalaya.txt",
    "tripura":                   "tripura.txt",
    "mizoram":                   "mizoram.txt",
    "nagaland":                  "nagaland.txt",
    "arunachal pradesh":         "arunachal_pradesh.txt",
    "arunachalpradesh":          "arunachal_pradesh.txt",
    "sikkim":                    "sikkim.txt",
    # ── 5 relevant UTs with own policy files ─────────────────────────────────
    "jammu and kashmir":         "jammu_kashmir.txt",
    "jammu & kashmir":           "jammu_kashmir.txt",
    "jammu kashmir":             "jammu_kashmir.txt",
    "j&k":                       "jammu_kashmir.txt",
    "jk":                        "jammu_kashmir.txt",
    "delhi":                     "delhi.txt",
    "nct delhi":                 "delhi.txt",
    "puducherry":                "puducherry.txt",
    "pondicherry":               "puducherry.txt",
    "andaman and nicobar":       "andaman_nicobar.txt",
    "andaman & nicobar":         "andaman_nicobar.txt",
    "andaman nicobar":           "andaman_nicobar.txt",
    "chandigarh":                "chandigarh.txt",
    # ── UTs that reuse closest state policy ──────────────────────────────────
    "dadra and nagar haveli":    "gujarat.txt",
    "daman and diu":             "gujarat.txt",
    "dadra nagar haveli":        "gujarat.txt",
    "ladakh":                    "jammu_kashmir.txt",
    "lakshadweep":               "_national.txt",
}

# Fallback policy numbers used when RAG fails or state not found
FALLBACK_POLICY = {
    "msp_per_kg":               0.0,   # no MSP assumed
    "state_bonus_per_kg":       0.0,
    "effective_price_per_kg":   0.0,   # will be replaced by market price
    "procurement_efficiency_pct": 20.0, # national average
    "subsidy_per_ha_per_season": 2000.0,  # PM-KISAN only
    "insurance_premium_pct":    0.02,   # 2% Kharif default
    "power_subsidy_per_ha":     3000.0,
    "policy_note":              "Using national average policy parameters.",
}


def _load_policy_text(state_name: str) -> str:
    """Load national policy + state-specific policy text."""
    national_path = POLICY_DIR / "_national.txt"
    national_text = national_path.read_text(encoding="utf-8") if national_path.exists() else ""

    state_key = state_name.lower().strip()
    state_file = STATE_FILE_MAP.get(state_key)

    if state_file:
        state_path = POLICY_DIR / state_file
        state_text = state_path.read_text(encoding="utf-8") if state_path.exists() else ""
    else:
        logger.warning(f"No state policy file for '{state_name}'. Using national only.")
        state_text = ""

    return national_text + "\n\n" + state_text


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    """Split text into overlapping chunks for retrieval."""
    # Split on double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += "\n" + para
        else:
            if current:
                chunks.append(current.strip())
            current = para

    if current:
        chunks.append(current.strip())

    return chunks


def _simple_keyword_retrieval(chunks: list, query: str, top_k: int = 6) -> list:
    """
    Simple keyword-based retrieval as fallback when sentence-transformers
    is not installed. Scores chunks by keyword overlap with query.
    """
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words & chunk_words)
        scored.append((overlap, chunk))
    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]


def _faiss_retrieval(chunks: list, query: str, top_k: int = 6) -> list:
    """
    FAISS-based semantic retrieval using sentence-transformers.
    Falls back to keyword retrieval if not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
        query_embedding  = model.encode([query], convert_to_numpy=True)

        # Cosine similarity
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        query_norm  = np.linalg.norm(query_embedding)
        similarities = (chunk_embeddings / chunk_norms) @ (query_embedding / query_norm).T
        top_indices  = similarities.flatten().argsort()[-top_k:][::-1]

        return [chunks[i] for i in top_indices]

    except ImportError:
        logger.info("sentence-transformers not installed. Using keyword retrieval.")
        return _simple_keyword_retrieval(chunks, query, top_k)


def query_policy(
    crop: str,
    state_name: str,
    season: str,
    groq_model: str = "llama3-8b-8192",
) -> dict:
    """
    Main entry point. Returns structured policy numbers for a crop in a state.

    Returns dict with keys:
      msp_per_kg                  : float  — central MSP price floor
      state_bonus_per_kg          : float  — additional state bonus
      effective_price_per_kg      : float  — msp + bonus (price floor)
      procurement_efficiency_pct  : float  — % of crop that actually reaches MSP
      subsidy_per_ha_per_season   : float  — all cash subsidies (PM-KISAN + state)
      insurance_premium_pct       : float  — farmer's PMFBY premium share
      power_subsidy_per_ha        : float  — power subsidy value
      policy_note                 : str    — human-readable policy summary
    """
    try:
        # Load text
        full_text = _load_policy_text(state_name)
        chunks    = _chunk_text(full_text)

        if not chunks:
            logger.warning("Policy text empty. Using fallback.")
            return {**FALLBACK_POLICY}

        # Retrieve relevant chunks
        query = (
            f"MSP price bonus procurement efficiency subsidy insurance premium "
            f"for {crop} in {state_name} {season} season 2024-25"
        )
        relevant_chunks = _faiss_retrieval(chunks, query, top_k=8)
        context = "\n\n---\n\n".join(relevant_chunks)

        # Ask Groq to extract structured numbers
        prompt = f"""You are an agricultural policy extraction assistant for India.

Extract policy parameters for the following crop from the context below.
Return ONLY a valid JSON object. No explanation, no markdown, no backticks.

Crop: {crop}
State: {state_name}
Season: {season}

Context:
{context}

Return this exact JSON structure:
{{
  "msp_per_kg": <float, central MSP per kg, 0 if no MSP>,
  "state_bonus_per_kg": <float, additional state bonus per kg above central MSP, 0 if none>,
  "effective_price_per_kg": <float, msp_per_kg + state_bonus_per_kg>,
  "procurement_efficiency_pct": <float, percentage 0-100 of crop likely to reach MSP procurement>,
  "subsidy_per_ha_per_season": <float, total cash subsidies per hectare per season including PM-KISAN Rs 2000 plus any state scheme>,
  "insurance_premium_pct": <float, PMFBY farmer premium as decimal e.g. 0.02 for 2%, 0 if state waives>,
  "power_subsidy_per_ha": <float, value of power/electricity subsidy per hectare, 0 if not mentioned>,
  "policy_note": "<one sentence: key policy fact for this crop in this state>"
}}

Rules:
- If no MSP exists for this crop, set msp_per_kg to 0
- If no state bonus, set state_bonus_per_kg to 0
- effective_price_per_kg must equal msp_per_kg + state_bonus_per_kg
- If state waives PMFBY premium, set insurance_premium_pct to 0
- PM-KISAN alone = Rs 2000/season. Add state scheme on top if mentioned.
- If data not found in context, use national defaults: insurance 0.02 Kharif / 0.015 Rabi"""

        groq_llm = ChatGroq(model=groq_model, temperature=0.0)
        response  = groq_llm.invoke([HumanMessage(content=prompt)])
        raw       = response.content.strip()

        # Clean any accidental markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()

        result = json.loads(raw)

        # Validate all required keys present
        required = [
            "msp_per_kg", "state_bonus_per_kg", "effective_price_per_kg",
            "procurement_efficiency_pct", "subsidy_per_ha_per_season",
            "insurance_premium_pct", "power_subsidy_per_ha", "policy_note"
        ]
        for key in required:
            if key not in result:
                result[key] = FALLBACK_POLICY.get(key, 0.0)

        # Sanity check: effective = msp + bonus
        result["effective_price_per_kg"] = (
            float(result["msp_per_kg"]) + float(result["state_bonus_per_kg"])
        )

        logger.info(
            f"[PolicyRAG] {crop} in {state_name}: "
            f"MSP=Rs{result['msp_per_kg']}, bonus=Rs{result['state_bonus_per_kg']}, "
            f"effective=Rs{result['effective_price_per_kg']}, "
            f"subsidy=Rs{result['subsidy_per_ha_per_season']}/ha, "
            f"insurance={result['insurance_premium_pct']*100:.1f}%"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"[PolicyRAG] JSON parse failed: {e}. Raw: {raw[:200]}")
        return {**FALLBACK_POLICY}
    except Exception as e:
        logger.error(f"[PolicyRAG] Error: {e}", exc_info=True)
        return {**FALLBACK_POLICY}
