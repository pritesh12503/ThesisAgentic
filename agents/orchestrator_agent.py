"""
agents/orchestrator_agent.py

LangGraph node: Orchestrator Agent
Combines Score_agro and Score_economic using weighted sum:
    Crop* = argmax_c [ w1 * Score_agro(c) + w2 * Score_economic(c) ]
"""

import logging
from typing import Dict

from state import CropAdvisorState
from config import SUPPORTED_CROPS, DEFAULT_W1, DEFAULT_W2

logger = logging.getLogger(__name__)


def orchestrator_node(state: CropAdvisorState) -> CropAdvisorState:
    """
    LangGraph node function for the Orchestrator.
    Reads: agro_scores, economic_scores, w1, w2
    Writes: final_scores, recommended_crop, top_3_crops
    """
    logger.info("[Orchestrator] Combining agronomic and economic scores.")

    agro_scores     = state.get("agro_scores") or {}
    economic_scores = state.get("economic_scores") or {}
    w1 = state.get("w1", DEFAULT_W1)
    w2 = state.get("w2", DEFAULT_W2)

    # Normalize weights to sum to 1
    total = w1 + w2
    if total == 0:
        w1, w2 = 0.5, 0.5
    else:
        w1, w2 = w1 / total, w2 / total

    final_scores: Dict[str, float] = {}
    for crop in SUPPORTED_CROPS:
        s_agro = agro_scores.get(crop, 0.0)
        s_econ = economic_scores.get(crop, 0.0)
        final_scores[crop] = round(w1 * s_agro + w2 * s_econ, 4)

    ranked = sorted(final_scores, key=final_scores.get, reverse=True)
    recommended_crop = ranked[0] if ranked else None
    top_3 = ranked[:3]

    logger.info(
        f"[Orchestrator] w1={w1:.2f}, w2={w2:.2f} | "
        f"Recommended: {recommended_crop} (score={final_scores.get(recommended_crop, 0):.4f})"
    )

    return {
        **state,
        "final_scores":     final_scores,
        "recommended_crop": recommended_crop,
        "top_3_crops":      top_3,
    }
