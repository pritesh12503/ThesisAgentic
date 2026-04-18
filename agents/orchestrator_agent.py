"""
agents/orchestrator_agent.py

LangGraph node: Orchestrator Agent — Option B cross-agent penalty (FIXED)

The penalty threshold was 0.15, but heuristic-based agro scores after
min-max normalisation inflate values — unsuitable crops like Grapes in
Chhattisgarh were getting 0.27, above the threshold, so no penalty fired.

FIX: Threshold raised to 0.35 to correctly catch heuristic-inflated scores.
After proper model retraining (from Colab), true probability scores will be
0.02–0.05 for genuinely unsuitable crops, well below any threshold.

Penalty formula (unchanged):
    if agro_score < THRESHOLD:
        penalty     = agro_score / THRESHOLD   # 0 → 0,  threshold → 1
        econ_adj    = econ_score × penalty
    else:
        econ_adj    = econ_score               # unaffected

Example with threshold=0.35:
    Grapes/Chhattisgarh: agro=0.27 < 0.35 → penalty=0.27/0.35=0.77
        econ_adj = 1.0 × 0.77 = 0.77
        combined = 0.5×0.27 + 0.5×0.77 = 0.52   (was 0.64 — now lower)

    Rice/Punjab:         agro=0.85 ≥ 0.35 → no penalty
        combined = 0.5×0.85 + 0.5×0.70 = 0.775  (unchanged)
"""

import logging
from typing import Dict

from state import CropAdvisorState
from config import SUPPORTED_CROPS, DEFAULT_W1, DEFAULT_W2

logger = logging.getLogger(__name__)

# CALIBRATED threshold for heuristic-based scores.
# Heuristic scores after min-max span 0–1 but unsuitable crops cluster ~0.1–0.35.
# Truly suitable crops (in model training data) cluster ~0.5–1.0.
# 0.35 correctly separates the two groups with heuristic scores.
# After full model retraining: this threshold can be lowered to 0.15.
AGRO_VIABILITY_THRESHOLD = 0.35


def _apply_agro_penalty(agro_score: float, econ_score: float) -> float:
    """
    Reduce economic score proportionally when agronomic suitability is low.
    Prevents high-value crops with national-average yields from beating
    locally suitable crops in the final ranking.
    """
    if agro_score < AGRO_VIABILITY_THRESHOLD:
        penalty = agro_score / AGRO_VIABILITY_THRESHOLD  # linear, [0, 1)
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

    total = w1 + w2
    if total == 0:
        w1, w2 = 0.5, 0.5
    else:
        w1, w2 = w1 / total, w2 / total

    final_scores: Dict[str, float] = {}
    penalised: Dict[str, tuple] = {}   # crop → (original_econ, adjusted_econ)

    for crop in SUPPORTED_CROPS:
        s_agro = agro_scores.get(crop, 0.0)
        s_econ = economic_scores.get(crop, 0.0)

        s_econ_adj = _apply_agro_penalty(s_agro, s_econ)

        if s_econ_adj < s_econ:
            penalised[crop] = (round(s_econ, 4), round(s_econ_adj, 4))

        final_scores[crop] = round(w1 * s_agro + w2 * s_econ_adj, 4)

    # Log top penalised crops for transparency
    if penalised:
        top_penalised = sorted(penalised.items(),
                               key=lambda x: x[1][0] - x[1][1], reverse=True)[:5]
        logger.info(
            f"[Orchestrator] Agro-penalty applied to {len(penalised)} crops. "
            "Top penalised: " +
            ", ".join(f"{c}(econ:{v[0]}→{v[1]})" for c, v in top_penalised)
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
