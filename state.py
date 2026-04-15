"""
state.py
Shared LangGraph state object. Every agent node reads from and writes to this.

IMPORTANT — errors field uses Annotated reducer:
  Both parallel fan-out pairs write to 'errors':
    Fan-out 1: agronomic_agent + market_agent (parallel)
    Fan-out 2: explainability + post_harvest_advisor (parallel)
  Without a reducer, LangGraph throws InvalidUpdateError.
  With operator.add as reducer, parallel writes are merged by concatenation.
"""

import operator
from typing import TypedDict, Optional, List, Dict, Annotated


def _merge_errors(a: Optional[List[str]], b: Optional[List[str]]) -> List[str]:
    """Merge two error lists from parallel nodes. Never crashes on None."""
    return (a or []) + (b or [])


class CropAdvisorState(TypedDict):
    # ── User inputs ──────────────────────────────────────────────────────────
    district:    str
    state_name:  str
    season:      str          # "Kharif" | "Rabi" | "Zaid"
    w1:          float        # weight for agronomic score
    w2:          float        # weight for economic score

    # ── Agronomic agent outputs ───────────────────────────────────────────────
    agro_scores:    Optional[Dict[str, float]]           # {crop: score_agro}
    shap_values:    Optional[Dict[str, Dict[str, float]]] # {crop: {feature: shap}}
    agro_top_crops: Optional[List[str]]                  # ranked crop list
    agro_reasoning: Optional[str]                        # LLM narrative
    lime_summary:   Optional[str]                        # LIME text validation of SHAP

    # ── Market agent outputs ──────────────────────────────────────────────────
    economic_scores:   Optional[Dict[str, float]]  # {crop: score_economic}
    yield_predictions: Optional[Dict[str, float]]  # {crop: yield_kg_ha}
    price_predictions: Optional[Dict[str, float]]  # {crop: effective_price_per_kg}
    profit_estimates:  Optional[Dict[str, float]]  # {crop: profit_INR_ha}
    policy_details:    Optional[Dict[str, dict]]   # {crop: {msp, bonus, subsidy...}}
    market_reasoning:  Optional[str]

    # ── Orchestrator outputs ──────────────────────────────────────────────────
    final_scores:     Optional[Dict[str, float]]
    recommended_crop: Optional[str]
    top_3_crops:      Optional[List[str]]

    # ── Explainability outputs ────────────────────────────────────────────────
    shap_summary:      Optional[str]
    sell_timing:       Optional[str]
    policy_note:       Optional[str]
    final_explanation: Optional[str]

    # ── Post-harvest advisor outputs ──────────────────────────────────────────
    post_harvest_action:     Optional[str]
    post_harvest_sell_month: Optional[str]
    post_harvest_channel:    Optional[str]
    post_harvest_storage:    Optional[str]
    post_harvest_net_gain:   Optional[float]
    weather_signal:          Optional[str]
    weather_urgency:         Optional[str]
    post_harvest_advisory:   Optional[str]

    # ── Metadata — Annotated so parallel nodes can both write safely ──────────
    errors: Annotated[List[str], _merge_errors]
