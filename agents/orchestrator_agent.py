"""
agents/orchestrator_agent.py

LangGraph node: Orchestrator Agent

UPDATED: Cross-agent consistency validation (Option B)

Core formula:
    final_score = w1 × Score_agro + w2 × Score_econ_adjusted

Where Score_econ_adjusted applies a penalty when agronomic score is very low.
This prevents crops with unrealistic national-fallback yields (e.g. Grapes in
Chhattisgarh) from dominating the recommendation despite being agronomically
unsuitable for the district.

Penalty logic:
    if agro_score < AGRO_THRESHOLD:
        penalty = agro_score / AGRO_THRESHOLD    # scales 0→0, threshold→1
        econ_adjusted = econ_score × penalty
    else:
        econ_adjusted = econ_score               # no change

Threshold = 0.15 (conservative — only genuinely unsuitable crops penalised).
"""

import logging
from typing import Dict

from state import CropAdvisorState
from config import SUPPORTED_CROPS, DEFAULT_W1, DEFAULT_W2

logger = logging.getLogger(__name__)

# Crops with agro_score below this are penalised in economic scoring.
# 0.15 is conservative — only clear mismatches (Apple in tropics, Coconut in
# Rajasthan, Grapes in Chhattisgarh) fall below this.
AGRO_VIABILITY_THRESHOLD = 0.15


def _apply_agro_penalty(agro_score: float, econ_score: float) -> float:
    """
    Reduce economic score proportionally when agronomic suitability is very low.
    This prevents high-value crops with national-average yields from ranking above
    locally-suitable crops.
    """
    if agro_score < AGRO_VIABILITY_THRESHOLD:
        penalty = agro_score / AGRO_VIABILITY_THRESHOLD  # [0, 1)
        return econ_score * penalty
    return econ_score


def orchestrator_node(state: CropAdvisorState) -> CropAdvisorState:
    """
    LangGraph node: Orchestrator.
    Reads:  agro_scores, economic_scores, w1, w2
    Writes: final_scores, recommended_crop, top_3_crops
    """
    logger.info("[Orchestrator] Combining agronomic and economic scores.")

    agro_scores     = state.get("agro_scores")     or {}
    economic_scores = state.get("economic_scores") or {}
    w1 = state.get("w1", DEFAULT_W1)
    w2 = state.get("w2", DEFAULT_W2)

    # Normalise weights
    total = w1 + w2
    if total == 0:
        w1, w2 = 0.5, 0.5
    else:
        w1, w2 = w1 / total, w2 / total

    final_scores: Dict[str, float] = {}
    penalty_applied: Dict[str, float] = {}   # for transparency logging

    for crop in SUPPORTED_CROPS:
        s_agro = agro_scores.get(crop, 0.0)
        s_econ = economic_scores.get(crop, 0.0)

        # Cross-agent consistency: penalise econ score if crop is
        # agronomically unsuitable for this district
        s_econ_adj = _apply_agro_penalty(s_agro, s_econ)

        if s_econ_adj < s_econ:
            penalty_applied[crop] = round(s_econ - s_econ_adj, 4)

        final_scores[crop] = round(w1 * s_agro + w2 * s_econ_adj, 4)

    if penalty_applied:
        logger.info(
            f"[Orchestrator] Agro-penalty applied to {len(penalty_applied)} crops: "
            + ", ".join(f"{c}(-{v:.3f})" for c, v in
                        sorted(penalty_applied.items(), key=lambda x: x[1], reverse=True)[:5])
        )

    ranked           = sorted(final_scores, key=final_scores.get, reverse=True)
    recommended_crop = ranked[0] if ranked else None
    top_3            = ranked[:3]

    logger.info(
        f"[Orchestrator] w1={w1:.2f}, w2={w2:.2f} | "
        f"Recommended: {recommended_crop} "
        f"(score={final_scores.get(recommended_crop, 0):.4f})"
    )

    return {
        "final_scores":     final_scores,
        "recommended_crop": recommended_crop,
        "top_3_crops":      top_3,
    }
