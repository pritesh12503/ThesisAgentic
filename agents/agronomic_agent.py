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
    compute_agro_scores, compute_shap_values, compute_lime_explanation, normalize_scores
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
        model, le, scaler, feature_names = load_agronomic_model(MODEL_PATH)

        # ── 2. Get soil+climate profile for district ──────────────────────────
        agro_df = load_agronomic_data(AGRONOMIC_CSV)
        soil_profile = get_district_soil_profile(agro_df, state["district"])
        logger.info(f"[AgronomicAgent] Soil profile: {soil_profile}")

        # ── 3. Compute suitability scores ─────────────────────────────────────
        raw_scores = compute_agro_scores(model, le, soil_profile, feature_names, scaler)

        # Model classes are lowercase; SUPPORTED_CROPS is title-case.
        crop_lower_map = {c.lower(): c for c in SUPPORTED_CROPS}

        model_scores: dict = {}
        for cls, score in raw_scores.items():
            canonical = crop_lower_map.get(cls.lower())
            if canonical:
                model_scores[canonical] = score

        # ── Heuristic scores for crops not in the model ───────────────────────
        # The agronomic model may only know a few crops if the Kaggle yield CSV
        # isn't present. Rather than zeroing out 49 crops, assign heuristic
        # scores based on known agronomic requirements vs the district profile.
        # Scores are rough but directionally correct and give the combined
        # scorer meaningful signal even before the full model is trained.
        ph    = soil_profile.get("pHsoil", 6.5)
        rain  = soil_profile.get("Rainfall_mm", 800)
        temp  = soil_profile.get("Temperature_C", 25)
        N     = soil_profile.get("Nitrogen Value", 40)
        humid = soil_profile.get("Humidity_%", 65)

        # Heuristic lookup: (ideal_pH, ideal_rain_mm, ideal_temp_C)
        # Score = 1 - normalised Euclidean distance from ideal
        CROP_IDEALS = {
            "Rice":              (6.5, 1200, 28), "Wheat":           (6.5,  500, 18),
            "Maize":             (6.5,  800, 26), "Jowar":           (6.8,  500, 28),
            "Bajra":             (7.0,  400, 30), "Ragi":            (6.0,  900, 25),
            "Barley":            (6.5,  400, 16), "Arhar/Tur":       (6.5,  700, 27),
            "Moong(Green Gram)": (6.5,  600, 28), "Urad":            (6.5,  700, 28),
            "Gram":              (6.5,  400, 20), "Masoor":          (6.5,  400, 18),
            "Cowpea":            (6.5,  700, 28), "Groundnut":       (6.5,  600, 28),
            "Rapeseed &Mustard": (6.5,  400, 18), "Sunflower":       (6.5,  600, 24),
            "Soyabean":          (6.5,  700, 26), "Sesamum":         (7.0,  500, 30),
            "Linseed":           (6.5,  400, 18), "Safflower":       (7.0,  400, 25),
            "Castor Seed":       (6.5,  500, 28), "Cotton(lint)":    (7.0,  700, 28),
            "Sugarcane":         (6.5, 1500, 28), "Tobacco":         (6.0,  600, 27),
            "Coconut":           (6.0, 1500, 30), "Arecanut":        (6.5, 2000, 28),
            "Coffee":            (6.0, 1800, 22), "Black pepper":    (5.5, 2500, 28),
            "Potato":            (5.5,  600, 18), "Onion":           (6.5,  500, 22),
            "Tomato":            (6.5,  700, 25), "Brinjal":         (6.5,  700, 27),
            "Cabbage":           (6.5,  600, 18), "Cauliflower":     (6.5,  600, 18),
            "Garlic":            (6.5,  500, 20), "Ginger":          (5.5, 1500, 28),
            "Turmeric":          (5.5, 1500, 28), "Tapioca":         (5.5, 1500, 28),
            "Sweet Potato":      (5.5, 1000, 27), "Bitter Gourd":    (6.5,  700, 28),
            "Bottle Gourd":      (6.5,  700, 28), "Pumpkin":         (6.5,  700, 28),
            "Banana":            (6.0, 1500, 28), "Mango":           (6.5,  800, 28),
            "Grapes":            (6.5,  700, 28), "Papaya":          (6.5,  800, 28),
            "Orange Fruit":      (6.0, 1000, 24), "Pineapple":       (5.5, 1500, 28),
            "Guava":             (6.5,  800, 28), "Lemon":           (6.0,  900, 26),
            "Apple":             (6.5,  900, 12),
        }

        def heuristic_score(crop: str) -> float:
            if crop not in CROP_IDEALS:
                return 0.3
            ideal_ph, ideal_rain, ideal_temp = CROP_IDEALS[crop]
            # Normalised distance components (tolerance ranges)
            d_ph   = abs(ph   - ideal_ph)   / 2.0
            d_rain = abs(rain - ideal_rain) / 1000.0
            d_temp = abs(temp - ideal_temp) / 15.0
            dist   = (d_ph + d_rain + d_temp) / 3.0
            return max(0.05, 1.0 - dist)  # floor at 0.05 — no crop is ever zero

        # Build final scores: use model probability where available,
        # heuristic otherwise. Scale heuristics to not exceed the
        # weakest model score so model-known crops stay preferred.
        if model_scores:
            min_model = min(model_scores.values())
            # heuristic scores are in [0,1]; scale so max heuristic = min_model * 0.95
            heuristic_raw = {c: heuristic_score(c)
                             for c in SUPPORTED_CROPS if c not in model_scores}
            max_h = max(heuristic_raw.values()) if heuristic_raw else 1.0
            scale = (min_model * 0.95) / max_h if max_h > 0 else 1.0
        else:
            heuristic_raw = {c: heuristic_score(c) for c in SUPPORTED_CROPS}
            scale = 1.0

        normalised: dict = {}
        for crop in SUPPORTED_CROPS:
            if crop in model_scores:
                normalised[crop] = model_scores[crop]
            else:
                normalised[crop] = heuristic_raw.get(crop, 0.01) * scale

        agro_scores = normalize_scores(normalised)

        # ── 4. Compute SHAP values ────────────────────────────────────────────
        shap_vals_raw = compute_shap_values(model, le, soil_profile, feature_names, scaler)
        # Apply same case normalisation as scores
        shap_vals = {}
        for cls, vals in shap_vals_raw.items():
            canonical = crop_lower_map.get(cls.lower())
            if canonical:
                shap_vals[canonical] = vals

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

        # ── 7. LIME validation (text only — no chart) ─────────────────────────
        # Compute on training data sample for LIME background distribution
        try:
            import numpy as _np
            train_bg = agro_df[AGRO_FEATURES].dropna().values
            # Use a sample for speed (LIME needs background, not full dataset)
            if len(train_bg) > 500:
                idx = _np.random.choice(len(train_bg), 500, replace=False)
                train_bg = train_bg[idx]
            rec_crop = top_crops[0] if top_crops else ""
            lime_text = compute_lime_explanation(
                model, le, soil_profile, feature_names,
                train_bg, rec_crop, scaler
            )
        except Exception as lime_err:
            lime_text = f"LIME skipped: {str(lime_err)}"
            logger.warning(f"[AgronomicAgent] LIME: {lime_err}")

        logger.info(f"[AgronomicAgent] Top crop: {top_crops[0] if top_crops else 'None'}, "
                    f"score: {agro_scores.get(top_crops[0], 0):.3f}")

        return {
            "agro_scores":    agro_scores,
            "shap_values":    shap_vals,
            "agro_top_crops": top_crops,
            "agro_reasoning": agro_reasoning,
            "lime_summary":   lime_text,
        }

    except Exception as e:
        logger.error(f"[AgronomicAgent] Error: {e}", exc_info=True)
        return {
            "agro_scores":    {c: 0.0 for c in SUPPORTED_CROPS},
            "shap_values":    {},
            "agro_top_crops": SUPPORTED_CROPS,
            "agro_reasoning": f"Agronomic analysis failed: {str(e)}",
            "lime_summary":   "",
            "errors": [f"AgronomicAgent: {str(e)}"],
        }
