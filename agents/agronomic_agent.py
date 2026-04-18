"""
agents/agronomic_agent.py

FIXES IN THIS VERSION:

Fix 1 — Season penalty:
  Wheat/Barley/Gram scoring high in Kharif season was wrong.
  Added SEASON_SUITABILITY dict. Crops unsuitable for selected season
  get agro_score *= 0.15 (85% penalty). Primary season crops get a boost.

Fix 2 — CROP_REC_TO_SUPPORTED mapping (all 22 base paper labels mapped):
  'chickpea'→'Gram', 'blackgram'→'Urad', 'cotton'→'Cotton(lint)', etc.

Fix 3 — Score combination:
  Model proba scaled to [0,1], heuristic in natural range, both normalized.
  No more near-zero values.
"""

import logging
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from state import CropAdvisorState
from config import (
    AGRONOMIC_CSV, YIELD_CSV, GROQ_MODEL,
    AGRO_FEATURES, AGRO_FEATURE_LABELS, SUPPORTED_CROPS
)
from utils.data_utils import load_agronomic_data, get_district_soil_profile
from utils.model_utils import (
    load_agronomic_model, train_agronomic_model,
    compute_agro_scores, compute_shap_values, normalize_scores
)

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).parent.parent / "models" / "agronomic_model.pkl"

# ── Crop_recommendation.csv labels → SUPPORTED_CROPS ─────────────────────────
CROP_REC_TO_SUPPORTED = {
    "rice":"Rice", "maize":"Maize", "banana":"Banana", "mango":"Mango",
    "grapes":"Grapes", "apple":"Apple", "papaya":"Papaya",
    "coconut":"Coconut", "coffee":"Coffee",
    "chickpea":"Gram", "kidneybeans":"Cowpea", "pigeonpeas":"Arhar/Tur",
    "mothbeans":"Moong(Green Gram)", "mungbean":"Moong(Green Gram)",
    "blackgram":"Urad", "lentil":"Masoor", "pomegranate":"Guava",
    "watermelon":"Pumpkin", "muskmelon":"Bottle Gourd",
    "orange":"Orange Fruit", "cotton":"Cotton(lint)", "jute":"Sugarcane",
}

# ── Season suitability ────────────────────────────────────────────────────────
# Crops completely wrong for a season get an 85% penalty on their agro score.
# Crops in their primary season get a 20% boost (capped at pre-normalize max).
SEASON_SUITABILITY = {
    "kharif": {
        "primary":    {"Rice","Maize","Jowar","Bajra","Ragi","Arhar/Tur",
                       "Moong(Green Gram)","Urad","Cowpea","Groundnut",
                       "Soyabean","Sesamum","Cotton(lint)","Sugarcane",
                       "Castor Seed","Ginger","Turmeric","Banana","Tomato",
                       "Brinjal","Bitter Gourd","Bottle Gourd","Pumpkin",
                       "Tapioca","Sweet Potato"},
        "unsuitable": {"Wheat","Barley","Gram","Masoor","Rapeseed &Mustard",
                       "Linseed","Safflower","Apple"},
    },
    "rabi": {
        "primary":    {"Wheat","Barley","Gram","Masoor","Rapeseed &Mustard",
                       "Linseed","Safflower","Potato","Onion","Garlic",
                       "Sunflower","Cabbage","Cauliflower"},
        "unsuitable": {"Rice","Cotton(lint)","Sugarcane"},
    },
    "zaid": {
        "primary":    {"Moong(Green Gram)","Urad","Sunflower","Maize",
                       "Tomato","Pumpkin","Watermelon","Cucumber"},
        "unsuitable": set(),
    },
}

SEASON_PENALTY  = 0.15   # unsuitable crop score multiplied by this
SEASON_BOOST    = 1.20   # primary season crop score multiplied by this


def _apply_season_adjustment(scores: dict, season: str) -> dict:
    """Apply season penalty/boost to agro scores."""
    season_key = season.lower().strip()
    rules = SEASON_SUITABILITY.get(season_key, {})
    unsuitable = rules.get("unsuitable", set())
    primary    = rules.get("primary",    set())

    adjusted = {}
    for crop, score in scores.items():
        if crop in unsuitable:
            adjusted[crop] = score * SEASON_PENALTY
        elif crop in primary:
            adjusted[crop] = min(score * SEASON_BOOST, score + 0.3)
        else:
            adjusted[crop] = score
    return adjusted


def _ensure_model_trained():
    if not MODEL_PATH.exists():
        logger.info("Model not found — training now...")
        from utils.data_utils import load_yield_data
        from config import YIELD_CSV
        agro_df  = load_agronomic_data(AGRONOMIC_CSV)
        yield_df = load_yield_data(YIELD_CSV)
        train_agronomic_model(agro_df, yield_df, MODEL_PATH,
                              model_type="random_forest", use_grid_search=False)


def _shap_text_summary(shap_vals: dict, crop: str) -> str:
    if crop not in shap_vals or not shap_vals[crop]:
        return ""
    sorted_feats = sorted(shap_vals[crop].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    lines = [f"SHAP explanation for {crop} (top 5 drivers):"]
    for fname, val in sorted_feats:
        label     = AGRO_FEATURE_LABELS.get(fname, fname)
        direction = "supports" if val > 0 else "reduces"
        lines.append(f"  • {label}: {direction} suitability ({val:+.4f})")
    return "\n".join(lines)


def _top_shap_text(shap_vals: dict, crop: str, n: int = 5) -> str:
    if crop not in shap_vals or not shap_vals[crop]:
        return "SHAP values not available."
    sorted_feats = sorted(shap_vals[crop].items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    lines = []
    for fname, val in sorted_feats:
        label     = AGRO_FEATURE_LABELS.get(fname, fname)
        direction = "supports" if val > 0 else "reduces"
        lines.append(f"  • {label}: {direction} suitability (SHAP={val:+.3f})")
    return "\n".join(lines)


def agronomic_agent_node(state: CropAdvisorState) -> CropAdvisorState:
    logger.info(f"[AgronomicAgent] {state['district']}, {state['state_name']}, {state['season']}")

    try:
        _ensure_model_trained()
        model, le, scaler, feature_names = load_agronomic_model(MODEL_PATH)

        agro_df      = load_agronomic_data(AGRONOMIC_CSV)
        soil_profile = get_district_soil_profile(
            agro_df, state["district"], state.get("state_name", "")
        )
        logger.info(f"[AgronomicAgent] Soil profile: {soil_profile}")

        # ── Raw model scores ──────────────────────────────────────────────────
        raw_scores = compute_agro_scores(model, le, soil_profile, feature_names, scaler)

        # ── Map to SUPPORTED_CROPS ────────────────────────────────────────────
        supported_lower = {c.lower(): c for c in SUPPORTED_CROPS}
        model_scores: dict = {}
        for cls, score in raw_scores.items():
            cls_lower = cls.lower().strip()
            canonical = CROP_REC_TO_SUPPORTED.get(cls_lower) or supported_lower.get(cls_lower)
            if canonical:
                if canonical not in model_scores or score > model_scores[canonical]:
                    model_scores[canonical] = score

        logger.info(f"[AgronomicAgent] Model mapped {len(model_scores)} crops")

        # ── Heuristic for remaining crops ─────────────────────────────────────
        ph    = soil_profile.get("pHsoil", 6.5)
        rain  = soil_profile.get("Rainfall_mm", 100)   # now monthly scale
        temp  = soil_profile.get("Temperature_C", 25)

        # Heuristic ideals also in monthly rainfall scale (annual/10)
        CROP_IDEALS = {
            "Rice":(6.5,200,28),"Wheat":(6.5,80,18),"Maize":(6.5,100,26),
            "Jowar":(6.8,60,28),"Bajra":(7.0,45,30),"Ragi":(6.0,100,25),
            "Barley":(6.5,55,16),"Arhar/Tur":(6.5,80,27),
            "Moong(Green Gram)":(6.5,70,28),"Urad":(6.5,75,28),
            "Gram":(6.5,50,20),"Masoor":(6.5,45,18),
            "Cowpea":(6.5,80,28),"Groundnut":(6.5,70,28),
            "Rapeseed &Mustard":(6.5,45,18),"Sunflower":(6.5,65,24),
            "Soyabean":(6.5,80,26),"Sesamum":(7.0,55,30),
            "Linseed":(6.5,45,18),"Safflower":(7.0,45,25),
            "Castor Seed":(6.5,55,28),"Cotton(lint)":(7.0,80,28),
            "Sugarcane":(6.5,180,28),"Tobacco":(6.0,65,27),
            "Coconut":(6.0,200,30),"Arecanut":(6.5,200,28),
            "Coffee":(6.0,180,22),"Black pepper":(5.5,250,28),
            "Potato":(5.5,65,18),"Onion":(6.5,55,22),
            "Tomato":(6.5,80,25),"Brinjal":(6.5,75,27),
            "Cabbage":(6.5,65,18),"Cauliflower":(6.5,65,18),
            "Garlic":(6.5,55,20),"Ginger":(5.5,150,28),
            "Turmeric":(5.5,150,28),"Tapioca":(5.5,150,28),
            "Sweet Potato":(5.5,110,27),"Bitter Gourd":(6.5,80,28),
            "Bottle Gourd":(6.5,80,28),"Pumpkin":(6.5,80,28),
            "Banana":(6.0,180,28),"Mango":(6.5,100,28),
            "Grapes":(6.5,75,18),"Papaya":(6.5,100,28),
            "Orange Fruit":(6.0,110,24),"Pineapple":(5.5,160,28),
            "Guava":(6.5,95,28),"Lemon":(6.0,100,26),"Apple":(6.5,100,12),
        }

        def heuristic(crop: str) -> float:
            if crop not in CROP_IDEALS:
                return 0.15
            ip, ir, it = CROP_IDEALS[crop]
            d = (abs(ph-ip)/2.5 + abs(rain-ir)/100 + abs(temp-it)/20) / 3
            return max(0.05, 1.0 - d)

        # Build combined scores
        all_scores: dict = {}
        if model_scores:
            max_model = max(model_scores.values())
            sf = 1.0 / max_model if max_model > 0 else 1.0
            for crop, s in model_scores.items():
                all_scores[crop] = s * sf
        for crop in SUPPORTED_CROPS:
            if crop not in all_scores:
                all_scores[crop] = heuristic(crop)

        # ── Season adjustment ─────────────────────────────────────────────────
        all_scores = _apply_season_adjustment(all_scores, state.get("season", ""))

        agro_scores = normalize_scores(all_scores)

        # ── SHAP ──────────────────────────────────────────────────────────────
        shap_vals_raw = compute_shap_values(model, le, soil_profile, feature_names, scaler)
        shap_vals = {}
        for cls, vals in shap_vals_raw.items():
            cls_lower = cls.lower().strip()
            canonical = CROP_REC_TO_SUPPORTED.get(cls_lower) or supported_lower.get(cls_lower)
            if canonical:
                shap_vals[canonical] = vals

        top_crops = sorted(agro_scores, key=agro_scores.get, reverse=True)
        rec_crop  = top_crops[0] if top_crops else ""
        logger.info(f"[AgronomicAgent] Top 5: {[(c, round(agro_scores[c],3)) for c in top_crops[:5]]}")

        shap_summary = _shap_text_summary(shap_vals, rec_crop)

        # ── Groq reasoning ────────────────────────────────────────────────────
        profile_text = "\n".join(
            f"  {AGRO_FEATURE_LABELS.get(k,k)}: {v:.2f}"
            for k, v in soil_profile.items()
        )
        scores_text = "\n".join(
            f"  {c}: {agro_scores[c]:.3f}"
            for c in top_crops[:10]
        )

        prompt = f"""You are an expert agronomist AI agent.

District: {state['district'].title()}, {state['state_name'].title()}
Season: {state['season']}

Soil and Climate Profile (rainfall in mm/month):
{profile_text}

Top Crop Suitability Scores (season-adjusted):
{scores_text}

Top SHAP drivers for '{rec_crop}':
{_top_shap_text(shap_vals, rec_crop)}

Write 3-4 sentences: why {rec_crop} suits this district+season, which features drive it, any risk."""

        groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)
        response = groq_llm.invoke([HumanMessage(content=prompt)])

        return {
            "agro_scores":    agro_scores,
            "shap_values":    shap_vals,
            "agro_top_crops": top_crops,
            "agro_reasoning": response.content,
            "lime_summary":   shap_summary,
        }

    except Exception as e:
        logger.error(f"[AgronomicAgent] Error: {e}", exc_info=True)
        return {
            "agro_scores":    {c: 0.0 for c in SUPPORTED_CROPS},
            "shap_values":    {},
            "agro_top_crops": SUPPORTED_CROPS,
            "agro_reasoning": f"Agronomic analysis failed: {str(e)}",
            "lime_summary":   "",
            "errors":         [f"AgronomicAgent: {str(e)}"],
        }
