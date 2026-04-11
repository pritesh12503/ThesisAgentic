"""
agents/explainability_agent.py

LangGraph node: Explainability Agent
- Summarizes SHAP feature importance for the recommended crop
- Explains WHY this crop was chosen (agronomic + economic reasoning)
- Calls Groq for a holistic natural-language explanation

NOTE: Post-harvest advisory (selling time, storage, weather, channel)
      is handled by the separate post_harvest_agent node.
      This node focuses purely on crop recommendation explainability.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from state import CropAdvisorState
from config import GROQ_MODEL, AGRO_FEATURE_LABELS, POLICY_KNOWLEDGE

logger = logging.getLogger(__name__)


def _top_shap_features(shap_vals: dict, crop: str, n: int = 5) -> str:
    """Format top SHAP features for a crop as bullet points."""
    if crop not in shap_vals or not shap_vals[crop]:
        return "SHAP feature breakdown not available."
    crop_shap = shap_vals[crop]
    sorted_feats = sorted(crop_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    lines = []
    for fname, val in sorted_feats:
        label = AGRO_FEATURE_LABELS.get(fname, fname)
        direction = "positive" if val > 0 else "negative"
        lines.append(f"  • {label}: {direction} influence (SHAP = {val:+.4f})")
    return "\n".join(lines)


def explainability_node(state: CropAdvisorState) -> CropAdvisorState:
    """
    LangGraph node: Explainability Agent.
    Reads:  recommended_crop, top_3_crops, agro_scores, economic_scores,
            final_scores, shap_values, yield_predictions, price_predictions,
            profit_estimates, agro_reasoning, market_reasoning
    Writes: shap_summary, policy_note, final_explanation
    """
    logger.info("[ExplainabilityAgent] Generating crop recommendation explanation.")

    crop        = state.get("recommended_crop", "")
    top_3       = state.get("top_3_crops", [])
    agro_scores = state.get("agro_scores")  or {}
    econ_scores = state.get("economic_scores") or {}
    final_scores= state.get("final_scores") or {}
    shap_vals   = state.get("shap_values")  or {}
    profit_ests = state.get("profit_estimates") or {}

    try:
        shap_summary = _top_shap_features(shap_vals, crop)
        policy_note  = POLICY_KNOWLEDGE

        scores_summary = "\n".join(
            f"  {c:25s}: agro={agro_scores.get(c,0):.3f}, "
            f"econ={econ_scores.get(c,0):.3f}, "
            f"combined={final_scores.get(c,0):.3f}, "
            f"profit=Rs{profit_ests.get(c,0):,.0f}/ha"
            for c in top_3
        )

        prompt = f"""You are an explainable AI crop advisory system for Indian farmers.

District: {state.get('district','').title()}, {state.get('state_name','').title()}
Season: {state.get('season','')}

RECOMMENDED CROP: {crop.upper()}
Top 3 crops by combined score: {', '.join(c for c in top_3)}

Score breakdown (agro = soil/climate suitability, econ = profit potential):
{scores_summary}

Agronomic agent reasoning:
{state.get('agro_reasoning','Not available')}

Market agent reasoning:
{state.get('market_reasoning','Not available')}

Top SHAP feature drivers for {crop}:
{shap_summary}

Write a clear explanation (4–5 sentences) covering:
1. Why {crop} was recommended over the alternatives
2. The key soil/climate factors that make it suitable (use SHAP features)
3. The economic case (yield, price, profit estimate)
4. Any significant risk or caveat the farmer should know

Write directly to the farmer. Be specific with numbers. Do not mention post-harvest advice here."""

        groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)
        response = groq_llm.invoke([HumanMessage(content=prompt)])
        final_explanation = response.content

        logger.info(f"[ExplainabilityAgent] Explanation generated for crop: {crop}")

        return {
            "shap_summary":     shap_summary,
            "sell_timing":      "",          # now handled by post_harvest_agent
            "policy_note":      policy_note,
            "final_explanation":final_explanation,
        }

    except Exception as e:
        logger.error(f"[ExplainabilityAgent] Error: {e}", exc_info=True)
        return {
            "shap_summary":      "",
            "sell_timing":       "",
            "policy_note":       POLICY_KNOWLEDGE,
            "final_explanation": f"Recommended crop: {crop}. "
                                 f"Agronomic score: {agro_scores.get(crop,0):.3f}. "
                                 f"Economic score: {econ_scores.get(crop,0):.3f}. "
                                 f"Estimated profit: Rs{profit_ests.get(crop,0):,.0f}/ha.",
            "errors": [f"ExplainabilityAgent: {str(e)}"],
        }
