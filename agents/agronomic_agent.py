"""
agents/agronomic_agent.py

LangGraph node: Agronomic Agent
- Filters soil+climate data by district
- Computes per-crop suitability scores via trained RF/GB model
- Computes SHAP feature importance values
- Calls Groq LLM to generate a natural-language agronomic reasoning summary
"""

import os
import logging
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from state import CropAdvisorState
from config import (
    AGRONOMIC_CSV, YIELD_CSV, GROQ_MODEL,
    AGRO_FEATURES, AGRO_FEATURE_LABELS, SUPPORTED_CROPS
)
from utils.data_utils import load_agronomic_data, load_yield_data, get_district_soil_profile
from utils.model_utils import (
    load_agronomic_model, train_agronomic_model,
    compute_agro_scores, compute_shap_values, normalize_scores
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "agronomic_model.pkl"


def _ensure_model_trained():
    """Train and save model if it doesn't exist yet."""
    if not MODEL_PATH.exists():
        logger.info("Model not found. Training agronomic model...")
        agro_df = load_agronomic_data(AGRONOMIC_CSV)
        yield_df = load_yield_data(YIELD_CSV)
        train_agronomic_model(agro_df, yield_df, MODEL_PATH, model_type="random_forest")
        logger.info("Model trained and saved.")


def _build_shap_summary_text(shap_vals: dict, crop: str) -> str:
    """Build human-readable SHAP summary for the recommended crop."""
    if crop not in shap_vals:
        return "SHAP values not available."
    crop_shap = shap_vals[crop]
    sorted_feats = sorted(crop_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for fname, val in sorted_feats[:5]:
        label = AGRO_FEATURE_LABELS.get(fname, fname)
        direction = "supports" if val > 0 else "reduces"
        lines.append(f"  • {label}: {direction} suitability (SHAP={val:+.3f})")
    return "\n".join(lines)


def agronomic_agent_node(state: CropAdvisorState) -> CropAdvisorState:
    """
    LangGraph node function for the Agronomic Agent.
    Reads: district, state_name
    Writes: agro_scores, shap_values, agro_top_crops, agro_reasoning
    """
    logger.info(f"[AgronomicAgent] Running for district: {state['district']}")

    try:
        # ── 1. Ensure model exists ────────────────────────────────────────────
        _ensure_model_trained()
        model, le, feature_names = load_agronomic_model(MODEL_PATH)

        # ── 2. Get soil+climate profile for district ──────────────────────────
        agro_df = load_agronomic_data(AGRONOMIC_CSV)
        soil_profile = get_district_soil_profile(agro_df, state["district"])
        logger.info(f"[AgronomicAgent] Soil profile: {soil_profile}")

        # ── 3. Compute suitability scores ─────────────────────────────────────
        raw_scores = compute_agro_scores(model, le, soil_profile, feature_names)
        # Only keep SUPPORTED_CROPS
        raw_scores = {k: v for k, v in raw_scores.items() if k in SUPPORTED_CROPS}
        agro_scores = normalize_scores(raw_scores)

        # ── 4. Compute SHAP values ────────────────────────────────────────────
        shap_vals = compute_shap_values(model, le, soil_profile, feature_names)
        shap_vals = {k: v for k, v in shap_vals.items() if k in SUPPORTED_CROPS}

        # ── 5. Rank crops ─────────────────────────────────────────────────────
        top_crops = sorted(agro_scores, key=agro_scores.get, reverse=True)

        # ── 6. Call Groq for agronomic reasoning narrative ────────────────────
        shap_text = _build_shap_summary_text(shap_vals, top_crops[0] if top_crops else "")
        profile_text = "\n".join(
            f"  {AGRO_FEATURE_LABELS.get(k, k)}: {v:.2f}" for k, v in soil_profile.items()
        )
        scores_text = "\n".join(
            f"  {crop}: {score:.3f}" for crop, score in
            sorted(agro_scores.items(), key=lambda x: x[1], reverse=True)
        )

        prompt = f"""You are an expert agronomist AI agent.

District: {state['district'].title()}, {state['state_name'].title()}
Season: {state['season']}

Soil and Climate Profile:
{profile_text}

Crop Suitability Scores (0=least suitable, 1=most suitable):
{scores_text}

Top SHAP Feature Drivers for '{top_crops[0] if top_crops else 'top crop'}':
{shap_text}

Write a concise 3-4 sentence agronomic reasoning summary explaining:
1. Why the top crop is environmentally suitable for this district
2. Which soil/climate features are most influential
3. Any agronomic risk or concern

Be specific with numbers from the soil profile. Keep it professional but accessible."""

        groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)
        response = groq_llm.invoke([HumanMessage(content=prompt)])
        agro_reasoning = response.content

        logger.info(f"[AgronomicAgent] Top crop: {top_crops[0] if top_crops else 'None'}, "
                    f"score: {agro_scores.get(top_crops[0], 0):.3f}")

        return {
            "agro_scores":    agro_scores,
            "shap_values":    shap_vals,
            "agro_top_crops": top_crops,
            "agro_reasoning": agro_reasoning,
        }

    except Exception as e:
        logger.error(f"[AgronomicAgent] Error: {e}", exc_info=True)
        return {
            "agro_scores":    {c: 0.0 for c in SUPPORTED_CROPS},
            "shap_values":    {},
            "agro_top_crops": SUPPORTED_CROPS,
            "agro_reasoning": f"Agronomic analysis failed: {str(e)}",
            "errors": [f"AgronomicAgent: {str(e)}"],
        }
