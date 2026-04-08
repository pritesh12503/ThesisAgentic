"""
agents/market_agent.py

LangGraph node: Market Agent (Economic Intelligence)

UPDATED: Now integrates Policy RAG into the profit formula.

Policy-Adjusted Profit Formula:
  effective_price = max(predicted_market_price, policy.effective_price_per_kg)
                    × procurement_weight + predicted_market_price × (1-procurement_weight)
  gross_revenue   = yield_kg_ha × effective_price
  net_profit      = gross_revenue
                    - input_cost
                    + subsidy_per_ha_per_season
                    + power_subsidy_per_ha
                    - (insurance_premium_pct × sum_insured_per_ha)

This profit is then normalized to Score_economic [0,1] across all crops.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict

from sklearn.linear_model import LinearRegression

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from state import CropAdvisorState
from config import (
    YIELD_CSV, PRICE_CSV, GROQ_MODEL,
    SUPPORTED_CROPS, CROP_TO_COMMODITY, INPUT_COSTS
)
from utils.data_utils import (
    load_yield_data, load_price_data,
    get_price_stats, normalize_district
)
from utils.model_utils import normalize_scores
from utils.policy_rag import query_policy

logger = logging.getLogger(__name__)

# Approximate sum insured per ha for PMFBY premium calculation (Rs/ha)
# Premium cost = insurance_premium_pct × sum_insured
PMFBY_SUM_INSURED = {
    "Rice": 30000, "Wheat": 28000, "Maize": 22000, "Jowar": 16000,
    "Bajra": 14000, "Ragi": 15000, "Barley": 18000, "Arhar/Tur": 20000,
    "Moong(Green Gram)": 16000, "Urad": 16000, "Gram": 20000,
    "Masoor": 18000, "Cowpea": 14000, "Groundnut": 30000,
    "Rapeseed &Mustard": 20000, "Sunflower": 22000, "Soyabean": 20000,
    "Sesamum": 18000, "Linseed": 17000, "Safflower": 18000,
    "Castor Seed": 18000, "Cotton(lint)": 35000, "Sugarcane": 40000,
    "Tobacco": 30000, "Coconut": 25000, "Arecanut": 28000,
    "Coffee": 30000, "Black pepper": 35000,
    "Potato": 45000, "Onion": 40000, "Tomato": 35000, "Brinjal": 28000,
    "Cabbage": 30000, "Cauliflower": 32000, "Garlic": 40000,
    "Ginger": 50000, "Turmeric": 40000, "Tapioca": 22000,
    "Sweet Potato": 20000, "Bitter Gourd": 28000, "Bottle Gourd": 25000,
    "Pumpkin": 20000, "Banana": 35000, "Mango": 25000, "Grapes": 60000,
    "Papaya": 28000, "Orange Fruit": 28000, "Pineapple": 30000,
    "Guava": 22000, "Lemon": 22000, "Apple": 50000,
}


def _predict_yield_for_crop(yield_df: pd.DataFrame, district: str, crop: str) -> float:
    """Predict yield (kg/ha) using linear trend regression. Falls back to national mean."""
    district  = normalize_district(district)
    dist_crop = yield_df[
        (yield_df["District_Name"] == district) & (yield_df["Crop"] == crop)
    ]
    data = dist_crop.sort_values("Crop_Year") if len(dist_crop) >= 5 \
           else yield_df[yield_df["Crop"] == crop].sort_values("Crop_Year")

    if data.empty:
        return 0.0
    if len(data) >= 5:
        X   = data["Crop_Year"].values.reshape(-1, 1)
        y   = data["Yield_kg_per_ha"].values
        reg = LinearRegression().fit(X, y)
        return max(float(reg.predict([[data["Crop_Year"].max() + 1]])[0]), 0.0)
    return float(data["Yield_kg_per_ha"].mean())


def _predict_market_price(price_df: pd.DataFrame, crop: str) -> float:
    """Return mean modal price (Rs/kg) from price dataset."""
    commodities = CROP_TO_COMMODITY.get(crop, [])
    if not commodities:
        return 0.0
    filtered = price_df[price_df["Commodity"].isin(commodities)]
    if filtered.empty:
        return 0.0
    return float(filtered["Modal Price"].mean()) / 100.0


def _compute_effective_price(
    market_price: float,
    policy: dict,
) -> float:
    """
    Compute the price a farmer realistically receives per kg.

    Logic:
      If effective_price_per_kg (MSP + bonus) > market_price:
        A fraction (procurement_efficiency_pct%) of farmers get the effective price.
        The rest sell at market price.
        Blended price = effective × pct + market × (1 - pct)
      Else:
        Market price is already above MSP — MSP is not binding.
        Effective price = market price.
    """
    effective_floor = float(policy.get("effective_price_per_kg", 0.0))
    proc_eff        = float(policy.get("procurement_efficiency_pct", 20.0)) / 100.0

    if effective_floor > market_price and effective_floor > 0:
        # Weighted average: some farmers get MSP, rest get market
        blended = effective_floor * proc_eff + market_price * (1.0 - proc_eff)
        return round(blended, 3)
    else:
        # Market already above MSP — irrelevant as floor
        return market_price


def _policy_adjusted_profit(
    yield_kg_ha: float,
    effective_price: float,
    crop: str,
    policy: dict,
) -> float:
    """
    Net profit per hectare after applying all policy adjustments.

    net_profit = (yield × effective_price)
               - input_cost
               + pm_kisan_and_state_subsidy
               + power_subsidy
               - pmfby_insurance_premium_cost
    """
    input_cost   = float(INPUT_COSTS.get(crop, 20000))
    gross        = yield_kg_ha * effective_price

    # Subsidies (cash income)
    subsidy      = float(policy.get("subsidy_per_ha_per_season", 2000.0))
    power_sub    = float(policy.get("power_subsidy_per_ha", 3000.0))

    # Insurance cost (deducted)
    ins_pct      = float(policy.get("insurance_premium_pct", 0.02))
    sum_insured  = float(PMFBY_SUM_INSURED.get(crop, 20000))
    ins_cost     = ins_pct * sum_insured

    net = gross - input_cost + subsidy + power_sub - ins_cost
    return max(net, 0.0)


def market_agent_node(state: CropAdvisorState) -> CropAdvisorState:
    """
    LangGraph node: Market Agent.
    Reads:  district, state_name, season
    Writes: economic_scores, yield_predictions, price_predictions,
            profit_estimates, policy_details, market_reasoning
    """
    logger.info(f"[MarketAgent] district={state['district']}, state={state['state_name']}")
    errors = list(state.get("errors") or [])

    try:
        yield_df = load_yield_data(YIELD_CSV)
        price_df = load_price_data(PRICE_CSV)

        yield_preds:    Dict[str, float] = {}
        price_preds:    Dict[str, float] = {}
        eff_prices:     Dict[str, float] = {}
        profit_ests:    Dict[str, float] = {}
        policy_details: Dict[str, dict]  = {}

        for crop in SUPPORTED_CROPS:
            # Step 1: yield prediction
            y_pred = _predict_yield_for_crop(yield_df, state["district"], crop)
            yield_preds[crop] = y_pred

            # Step 2: market price prediction
            p_market = _predict_market_price(price_df, crop)
            price_preds[crop] = p_market

            # Step 3: policy RAG — get MSP, bonus, subsidies, insurance
            policy = query_policy(
                crop       = crop,
                state_name = state["state_name"],
                season     = state["season"],
                groq_model = GROQ_MODEL,
            )
            policy_details[crop] = policy

            # Step 4: effective price (policy-adjusted)
            eff_price = _compute_effective_price(p_market, policy)
            eff_prices[crop] = eff_price

            # Step 5: policy-adjusted net profit
            profit = _policy_adjusted_profit(y_pred, eff_price, crop, policy)
            profit_ests[crop] = profit

            logger.info(
                f"[MarketAgent] {crop}: yield={y_pred:.0f} kg/ha, "
                f"market=Rs{p_market:.2f}/kg, eff=Rs{eff_price:.2f}/kg, "
                f"profit=Rs{profit:.0f}/ha"
            )

        # Normalize to Score_economic [0,1]
        economic_scores = normalize_scores(profit_ests)

        # Build Groq prompt — show top 8 crops for narrative
        top_crops = sorted(economic_scores, key=economic_scores.get, reverse=True)[:8]
        crops_table = "\n".join(
            f"  {c:25s}: yield={yield_preds[c]:.0f} kg/ha, "
            f"mkt=Rs{price_preds[c]:.2f}/kg, eff=Rs{eff_prices[c]:.2f}/kg "
            f"(MSP+bonus+subsidy applied), profit=Rs{profit_ests[c]:,.0f}/ha, "
            f"score={economic_scores[c]:.3f}"
            for c in top_crops
        )

        prompt = f"""You are an agricultural economics AI agent for Indian farmers.

District: {state['district'].title()}, {state['state_name'].title()}
Season: {state['season']}

Top crops by policy-adjusted profit (includes MSP floor, state bonus, subsidies, insurance):
{crops_table}

Write a 3-4 sentence market intelligence summary:
1. Most economically attractive crop and why (policy context matters)
2. Whether MSP is above or below market price for top crop
3. Any notable state-specific policy advantage in {state['state_name'].title()}

Be specific with rupee figures. Keep it farmer-focused."""

        groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)
        response = groq_llm.invoke([HumanMessage(content=prompt)])

        return {
            **state,
            "economic_scores":    economic_scores,
            "yield_predictions":  yield_preds,
            "price_predictions":  eff_prices,   # store effective price (policy-adjusted)
            "profit_estimates":   profit_ests,
            "policy_details":     policy_details,
            "market_reasoning":   response.content,
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"[MarketAgent] Error: {e}", exc_info=True)
        errors.append(f"MarketAgent error: {str(e)}")
        return {
            **state,
            "economic_scores":   {c: 0.0 for c in SUPPORTED_CROPS},
            "yield_predictions": {c: 0.0 for c in SUPPORTED_CROPS},
            "price_predictions": {c: 0.0 for c in SUPPORTED_CROPS},
            "profit_estimates":  {c: 0.0 for c in SUPPORTED_CROPS},
            "policy_details":    {},
            "market_reasoning":  f"Market analysis failed: {str(e)}",
            "errors": errors,
        }
