"""
graph.py
Assembles the LangGraph StateGraph for the Agentic Crop Advisor.

Graph structure:
    START
      │
      ▼
  user_input
      │
   ┌──┴──┐   (parallel fan-out)
   ▼     ▼
agro   market
   └──┬──┘   (fan-in at orchestrator)
      ▼
 orchestrator
      │
      ▼
 explainability
      │
      ▼
     END
"""

import logging
from langgraph.graph import StateGraph, START, END

from state import CropAdvisorState
from agents.agronomic_agent import agronomic_agent_node
from agents.market_agent import market_agent_node
from agents.orchestrator_agent import orchestrator_node
from agents.explainability_agent import explainability_node
from agents.post_harvest_agent import post_harvest_node

logger = logging.getLogger(__name__)


def user_input_node(state: CropAdvisorState) -> CropAdvisorState:
    """
    Pass-through node that validates and logs the user input.
    All initialization (district, season, weights) is expected to be
    set before the graph is invoked.
    """
    logger.info(
        f"[UserInput] district={state['district']}, "
        f"state={state['state_name']}, season={state['season']}, "
        f"w1={state['w1']}, w2={state['w2']}"
    )
    return state


def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""

    builder = StateGraph(CropAdvisorState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("user_input",          user_input_node)
    builder.add_node("agronomic_agent",     agronomic_agent_node)
    builder.add_node("market_agent",        market_agent_node)
    builder.add_node("orchestrator",        orchestrator_node)
    builder.add_node("explainability",      explainability_node)
    builder.add_node("post_harvest_advisor",post_harvest_node)   # NEW

    # ── Define edges ──────────────────────────────────────────────────────────
    builder.add_edge(START, "user_input")

    # Fan-out: user_input → both agents in parallel
    builder.add_edge("user_input", "agronomic_agent")
    builder.add_edge("user_input", "market_agent")

    # Fan-in: both agents → orchestrator
    builder.add_edge("agronomic_agent", "orchestrator")
    builder.add_edge("market_agent",    "orchestrator")

    # After orchestrator: explainability + post_harvest run in PARALLEL
    builder.add_edge("orchestrator", "explainability")
    builder.add_edge("orchestrator", "post_harvest_advisor")   # NEW parallel edge

    # Both feed to END
    builder.add_edge("explainability",       END)
    builder.add_edge("post_harvest_advisor", END)   # NEW

    graph = builder.compile()
    logger.info("[Graph] LangGraph compiled successfully.")
    return graph


def run_crop_advisor(
    district: str,
    state_name: str,
    season: str = "Kharif",
    w1: float = 0.5,
    w2: float = 0.5,
) -> CropAdvisorState:
    """
    High-level entry point. Returns the final state after the full graph run.
    """
    initial_state: CropAdvisorState = {
        "district":               district,
        "state_name":             state_name,
        "season":                 season,
        "w1":                     w1,
        "w2":                     w2,
        "agro_scores":            None,
        "shap_values":            None,
        "agro_top_crops":         None,
        "agro_reasoning":         None,
        "lime_summary":           None,
        "economic_scores":        None,
        "yield_predictions":      None,
        "price_predictions":      None,
        "profit_estimates":       None,
        "policy_details":         None,
        "market_reasoning":       None,
        "final_scores":           None,
        "recommended_crop":       None,
        "top_3_crops":            None,
        "shap_summary":           None,
        "sell_timing":            None,
        "policy_note":            None,
        "final_explanation":      None,
        "post_harvest_action":     None,
        "post_harvest_sell_month": None,
        "post_harvest_channel":    None,
        "post_harvest_storage":    None,
        "post_harvest_net_gain":   None,
        "weather_signal":          None,
        "weather_urgency":         None,
        "post_harvest_advisory":   None,
        "errors": [],   # must be list, not None — Annotated reducer expects list
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    result = run_crop_advisor(
        district="buldhana",
        state_name="maharashtra",
        season="Kharif",
        w1=0.5,
        w2=0.5,
    )

    print("\n" + "="*60)
    print("CROP ADVISORY RESULT")
    print("="*60)
    print(f"Recommended Crop : {result['recommended_crop']}")
    print(f"Top 3 Crops      : {result['top_3_crops']}")
    print(f"Final Scores     : {result['final_scores']}")
    print()
    print("Final Explanation:")
    print(result["final_explanation"])
    print()
    print("Sell Timing:")
    print(result["sell_timing"])
    if result.get("errors"):
        print("\nErrors:", result["errors"])
