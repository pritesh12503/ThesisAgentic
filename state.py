"""
state.py
Shared LangGraph state object. Every agent node reads from and writes to this.
"""

from typing import TypedDict, Optional, List, Dict, Any


class CropAdvisorState(TypedDict):
    # ── User inputs ──────────────────────────────────────────────────────────
    district: str
    state_name: str
    season: str                        # "Kharif" | "Rabi" | "Zaid"

    # ── Agronomic agent outputs ───────────────────────────────────────────────
    agro_scores: Optional[Dict[str, float]]          # {crop: score_agro}
    shap_values: Optional[Dict[str, Dict[str, float]]]  # {crop: {feature: shap}}
    agro_top_crops: Optional[List[str]]              # ranked crop list
    agro_reasoning: Optional[str]                    # LLM narrative from Groq

    # ── Market agent outputs ──────────────────────────────────────────────────
    economic_scores: Optional[Dict[str, float]]      # {crop: score_economic}
    yield_predictions: Optional[Dict[str, float]]    # {crop: predicted_yield_kg_ha}
    price_predictions: Optional[Dict[str, float]]    # {crop: effective_price_per_kg (policy-adjusted)}
    profit_estimates: Optional[Dict[str, float]]     # {crop: policy_adjusted_profit_INR_ha}
    policy_details: Optional[Dict[str, dict]]        # {crop: {msp, bonus, subsidy, insurance...}}
    market_reasoning: Optional[str]                  # LLM narrative from Groq

    # ── Orchestrator outputs ──────────────────────────────────────────────────
    final_scores: Optional[Dict[str, float]]         # {crop: combined score}
    recommended_crop: Optional[str]                  # Crop*
    top_3_crops: Optional[List[str]]

    # ── Explainability outputs ────────────────────────────────────────────────
    shap_summary: Optional[str]                      # top features text
    sell_timing: Optional[str]                       # post-harvest strategy
    policy_note: Optional[str]                       # MSP / policy text
    final_explanation: Optional[str]                 # full human-readable explanation

    # ── Post-harvest advisor outputs (independent node) ──────────────────────
    post_harvest_action:     Optional[str]   # SELL_IMMEDIATELY | WAIT | SELL_THIS_WEEK
    post_harvest_sell_month: Optional[str]   # "February" | "immediately"
    post_harvest_channel:    Optional[str]   # "e-NAM Thanjavur APMC" | "FCI centre" etc.
    post_harvest_storage:    Optional[str]   # storage type + duration advice
    post_harvest_net_gain:   Optional[float] # Rs/quintal net gain from waiting (0 = sell now)
    weather_signal:          Optional[str]   # SURGE_OPPORTUNITY | STORAGE_RISK | PERISHABLE_CRASH | TRANSPORT_DISRUPTION | NORMAL
    weather_urgency:         Optional[str]   # HIGH | MEDIUM | LOW
    post_harvest_advisory:   Optional[str]   # final Groq narration text

    # ── Metadata ─────────────────────────────────────────────────────────────
    errors: Optional[List[str]]
    w1: float                                        # weight for agronomic score
    w2: float                                        # weight for economic score
