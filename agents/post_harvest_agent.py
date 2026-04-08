"""
agents/post_harvest_agent.py

LangGraph node: Post-Harvest Advisor
Independent node — runs in parallel with explainability after orchestrator.
Has NO dependency on agro_scores, economic_scores, SHAP, or any other agent output
except: recommended_crop, district, state_name, season, price_predictions.

Calls Groq once at the end to narrate the structured advisory dict into
farmer-friendly plain language.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from state import CropAdvisorState
from config import GROQ_MODEL
from utils.post_harvest import (
    build_post_harvest_advisory,
    get_district_coordinates,
)

logger = logging.getLogger(__name__)


def post_harvest_node(state: CropAdvisorState) -> CropAdvisorState:
    """
    LangGraph node for post-harvest advisory.

    Reads:  recommended_crop, district, state_name, season, price_predictions
    Writes: post_harvest_action, post_harvest_sell_month, post_harvest_channel,
            post_harvest_storage, post_harvest_net_gain, weather_signal,
            weather_urgency, post_harvest_advisory
    """
    logger.info("[PostHarvestAgent] Running post-harvest analysis.")
    errors = list(state.get("errors") or [])

    crop     = state.get("recommended_crop", "")
    district = state.get("district", "")
    state_nm = state.get("state_name", "")
    season   = state.get("season", "")

    # Get current predicted price for this crop from market agent output
    price_preds = state.get("price_predictions") or {}
    current_price = price_preds.get(crop, 20.0)   # fallback Rs 20/kg

    try:
        # Get district coordinates for weather API
        coords = get_district_coordinates(district, state_nm)
        lat, lon = (coords[0], coords[1]) if coords else (None, None)
        if not coords:
            logger.warning(f"[PostHarvestAgent] No coordinates for {district}, {state_nm}. "
                           f"Weather forecast will be skipped.")

        # Run all sub-modules
        advisory = build_post_harvest_advisory(
            crop=crop,
            district=district,
            state=state_nm,
            season=season,
            current_price_per_kg=current_price,
            lat=lat,
            lon=lon,
        )

        # Build Groq prompt from structured advisory
        weather_block = (
            f"Weather alert ({advisory['weather_urgency']} urgency): {advisory['weather_headline']}\n"
            f"{advisory['weather_detail']}"
            if advisory["weather_signal"] != "NORMAL"
            else "No significant weather events forecast in the next 7 days."
        )

        storage_block = (
            f"Storage: {advisory['storage_type']}\n"
            f"Shelf life: {advisory['shelf_months']} months\n"
            f"Storage cost: Rs {advisory['storage_cost']:.0f}/quintal\n"
            f"Price gain if wait: Rs {advisory['net_gain_quintal']:.0f}/quintal net"
            if advisory["storage_viable"]
            else f"Storage: {advisory['storage_type']} — waiting not viable financially"
        )

        prompt = f"""You are a post-harvest advisory expert for Indian farmers.
The farmer has decided to grow {crop} in {district.title()}, {state_nm.title()}.

FINAL DECISION: {advisory['final_action'].replace('_', ' ')}

SEASONAL ANALYSIS:
  Current price: Rs {current_price:.2f}/kg
  Best selling month historically: {advisory['best_month']}
  Expected price at peak: Rs {advisory['projected_price']:.2f}/kg (+{advisory['peak_uplift_pct']}%)
  Months to wait: {advisory['months_to_wait']}

{storage_block}

WEATHER SITUATION:
{weather_block}

SELLING CHANNEL:
  Recommended: {advisory['channel']}
  Note: {advisory['channel_note']}

Write a concise, practical post-harvest advisory (4–5 sentences) that:
1. Tells the farmer clearly: sell now OR wait until [month]
2. Explains the financial reasoning in simple terms (use the Rs numbers)
3. Mentions any weather alert if urgency is HIGH or MEDIUM
4. Tells them where to sell locally
5. Gives one key storage tip if waiting is recommended

Write directly to the farmer. Use simple language. Be specific with numbers.
Do NOT mention scores, agents, or technical system details."""

        groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)
        response = groq_llm.invoke([HumanMessage(content=prompt)])
        advisory_text = response.content

        logger.info(
            f"[PostHarvestAgent] Crop: {crop} | Action: {advisory['final_action']} | "
            f"Weather: {advisory['weather_signal']} ({advisory['weather_urgency']})"
        )

        return {
            **state,
            "post_harvest_action":     advisory["final_action"],
            "post_harvest_sell_month": advisory["sell_month"],
            "post_harvest_channel":    advisory["channel"],
            "post_harvest_storage":    advisory["storage_type"],
            "post_harvest_net_gain":   advisory["net_gain_quintal"],
            "weather_signal":          advisory["weather_signal"],
            "weather_urgency":         advisory["weather_urgency"],
            "post_harvest_advisory":   advisory_text,
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"[PostHarvestAgent] Error: {e}", exc_info=True)
        errors.append(f"PostHarvestAgent error: {str(e)}")
        return {
            **state,
            "post_harvest_action":     "SELL_IMMEDIATELY",
            "post_harvest_sell_month": "as soon as possible",
            "post_harvest_channel":    "local APMC mandi",
            "post_harvest_storage":    "standard dry storage",
            "post_harvest_net_gain":   0.0,
            "weather_signal":          "NORMAL",
            "weather_urgency":         "LOW",
            "post_harvest_advisory":   f"Post-harvest analysis failed: {str(e)}. "
                                       f"Please consult your local agricultural extension officer.",
            "errors": errors,
        }
