"""
app.py
Streamlit UI for the Agentic Crop Advisor.
Run with: streamlit run app.py
"""

import os
import sys
import logging
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Make sure project root is in path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import SUPPORTED_CROPS, UI_SEASONS, DEFAULT_W1, DEFAULT_W2, AGRO_FEATURE_LABELS
from utils.data_utils import load_agronomic_data, load_yield_data
from config import AGRONOMIC_CSV, YIELD_CSV

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriAdvisor AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2d6a4f;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f7f4;
        border-left: 4px solid #2d6a4f;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
        border: 2px solid #2d6a4f;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .crop-badge {
        font-size: 2rem;
        font-weight: 800;
        color: #1b4332;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .score-chip {
        display: inline-block;
        background: #2d6a4f;
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 2px;
    }
    .agent-section {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    .agent-title {
        font-weight: 600;
        color: #1b4332;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load district lists ────────────────────────────────────────────────────────
@st.cache_data
def get_district_list():
    try:
        agro_df = load_agronomic_data(AGRONOMIC_CSV)
        yield_df = load_yield_data(YIELD_CSV)
        agro_d = set(agro_df["District"].str.title().unique())
        yield_d = set(yield_df["Dist Name"].str.title().unique())
        all_d = sorted(agro_d | yield_d)
        return all_d
    except Exception:
        return ["Buldhana", "Ludhiana", "Allahabad", "Bangalore", "Mysuru"]


@st.cache_data
def get_state_list():
    try:
        agro_df = load_agronomic_data(AGRONOMIC_CSV)
        return sorted(agro_df["State"].str.title().unique().tolist())
    except Exception:
        return ["Maharashtra", "Karnataka", "Gujarat", "Tamil Nadu"]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/seedling.png", width=60)
    st.title("AgriAdvisor AI")
    st.caption("Agentic AI for Crop Planning")
    st.divider()

    st.subheader("🔑 API Configuration")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com"
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("API key set ✓")
    elif os.environ.get("GROQ_API_KEY"):
        st.success("API key loaded from environment ✓")
    else:
        st.warning("Enter Groq API key to enable LLM reasoning")

    st.divider()
    st.subheader("📍 Location & Season")

    districts = get_district_list()
    states    = get_state_list()

    selected_district = st.selectbox(
        "District",
        options=districts,
        index=districts.index("Buldhana") if "Buldhana" in districts else 0,
        help="Select your farming district"
    )

    selected_state = st.selectbox(
        "State",
        options=states,
        index=0,
        help="Select your state"
    )

    selected_season = st.selectbox(
        "Season",
        options=UI_SEASONS,
        index=0,
        help="Kharif = Jun–Oct, Rabi = Nov–Apr, Zaid = Mar–Jun"
    )

    st.divider()
    st.subheader("⚖️ Score Weights")
    st.caption("Balance agronomic vs economic priority")

    w1 = st.slider(
        "Agronomic weight (soil/climate)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Higher = prioritize crop suitability for your soil"
    )
    w2 = round(1.0 - w1, 2)
    st.info(f"Agronomic: **{w1:.0%}** | Economic: **{w2:.0%}**")

    st.divider()
    run_button = st.button(
        "🌱 Get Crop Recommendation",
        type="primary",
        use_container_width=True,
    )


# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌾 AgriAdvisor AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Explainable Agentic AI Framework · Agronomic + Market Intelligence</div>',
    unsafe_allow_html=True
)

if not run_button:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <b>🌿 Agronomic Agent</b><br>
            <small>Analyzes N, P, K, pH, climate using Random Forest + SHAP explainability</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <b>📈 Market Agent</b><br>
            <small>Predicts crop yield & price, estimates profit per hectare</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <b>🎯 Orchestrator</b><br>
            <small>Combines both scores with tunable weights for final recommendation</small>
        </div>
        """, unsafe_allow_html=True)

    st.info("👈 Configure your location and weights in the sidebar, then click **Get Crop Recommendation**")

else:
    # ── Validate API key ───────────────────────────────────────────────────────
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
        st.stop()

    # ── Run the LangGraph pipeline ────────────────────────────────────────────
    with st.spinner("🤖 Running Agentic AI pipeline... (this takes 20-40 seconds)"):
        progress = st.progress(0, text="Initializing agents...")

        try:
            from graph import run_crop_advisor

            progress.progress(20, text="Training/loading agronomic model...")

            result = run_crop_advisor(
                district=selected_district.lower(),
                state_name=selected_state.lower(),
                season=selected_season,
                w1=w1,
                w2=w2,
            )
            progress.progress(100, text="Done!")

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            st.stop()

    # ── Display errors if any ──────────────────────────────────────────────────
    if result.get("errors"):
        for err in result["errors"]:
            st.warning(f"⚠️ {err}")

    # ── Recommendation banner ──────────────────────────────────────────────────
    crop = result.get("recommended_crop", "N/A")
    top3 = result.get("top_3_crops", [])
    final_scores = result.get("final_scores", {})

    CROP_EMOJI = {"rice": "🌾", "maize": "🌽", "chickpea": "🫘", "cotton": "🌿"}
    emoji = CROP_EMOJI.get(crop.lower(), "🌱")

    st.markdown(f"""
    <div class="recommendation-box">
        <div style="font-size:0.85rem;color:#555;margin-bottom:6px">RECOMMENDED CROP</div>
        <div class="crop-badge">{emoji} {crop.upper()}</div>
        <div style="margin-top:8px;font-size:0.9rem;color:#1b4332">
            Combined Score: <b>{final_scores.get(crop, 0):.3f}</b> &nbsp;|&nbsp;
            District: <b>{selected_district}</b> &nbsp;|&nbsp;
            Season: <b>{selected_season}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top 3 comparison ───────────────────────────────────────────────────────
    st.subheader("📊 Crop Score Comparison")
    tab1, tab2 = st.tabs(["Visual Chart", "Data Table"])

    with tab1:
        crops_to_show = top3 if top3 else list(final_scores.keys())
        agro_vals  = [result.get("agro_scores", {}).get(c, 0) for c in crops_to_show]
        econ_vals  = [result.get("economic_scores", {}).get(c, 0) for c in crops_to_show]
        final_vals = [final_scores.get(c, 0) for c in crops_to_show]

        x = np.arange(len(crops_to_show))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width, agro_vals,  width, label=f"Agronomic (w={w1:.2f})", color="#52b788", alpha=0.85)
        ax.bar(x,         econ_vals,  width, label=f"Economic  (w={w2:.2f})", color="#f4a261", alpha=0.85)
        ax.bar(x + width, final_vals, width, label="Combined Score",           color="#2d6a4f", alpha=0.95)

        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in crops_to_show], fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (0–1)")
        ax.set_title("Agent Score Comparison", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        profit_ests = result.get("profit_estimates", {})
        yield_preds = result.get("yield_predictions", {})
        price_preds = result.get("price_predictions", {})

        table_data = []
        for c in SUPPORTED_CROPS:
            table_data.append({
                "Crop": c.title(),
                "Agro Score": f"{result.get('agro_scores', {}).get(c, 0):.3f}",
                "Econ Score": f"{result.get('economic_scores', {}).get(c, 0):.3f}",
                "Combined": f"{final_scores.get(c, 0):.3f}",
                "Yield (kg/ha)": f"{yield_preds.get(c, 0):,.0f}",
                "Price (₹/kg)": f"{price_preds.get(c, 0):.2f}",
                "Est. Profit (₹/ha)": f"₹{profit_ests.get(c, 0):,.0f}",
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── SHAP Feature Importance ────────────────────────────────────────────────
    st.subheader("🔍 SHAP Feature Importance")
    shap_vals = result.get("shap_values", {})

    if shap_vals and crop in shap_vals:
        crop_shap = shap_vals[crop]
        sorted_feats = sorted(crop_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:7]
        feat_names = [AGRO_FEATURE_LABELS.get(f, f) for f, _ in sorted_feats]
        feat_vals  = [v for _, v in sorted_feats]

        colors = ["#52b788" if v >= 0 else "#e63946" for v in feat_vals]

        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        bars = ax2.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1], alpha=0.85)
        ax2.axvline(0, color="#333", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("SHAP Value (impact on model output)", fontsize=10)
        ax2.set_title(f"Feature drivers for {crop.title()} recommendation", fontsize=12, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        pos_patch = mpatches.Patch(color="#52b788", alpha=0.85, label="Positive influence")
        neg_patch = mpatches.Patch(color="#e63946", alpha=0.85, label="Negative influence")
        ax2.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()
    else:
        st.info("SHAP values not available for this run.")

    # ── Agent Reasoning ────────────────────────────────────────────────────────
    st.subheader("🤖 Agent Reasoning (Groq LLM)")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="agent-title">🌿 Agronomic Agent</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="agent-section">{result.get("agro_reasoning", "Not available")}</div>',
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown('<div class="agent-title">📈 Market Agent</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="agent-section">{result.get("market_reasoning", "Not available")}</div>',
            unsafe_allow_html=True
        )

    # ── Final Holistic Explanation ─────────────────────────────────────────────
    st.subheader("📋 Final Advisory Recommendation")
    st.markdown(
        f'<div class="agent-section">{result.get("final_explanation", "Not available")}</div>',
        unsafe_allow_html=True
    )

    # ── Post-Harvest Strategy ──────────────────────────────────────────────────
    st.subheader("📦 Post-Harvest Strategy")

    ph_advisory  = result.get("post_harvest_advisory", "")
    ph_action    = result.get("post_harvest_action", "")
    ph_month     = result.get("post_harvest_sell_month", "")
    ph_channel   = result.get("post_harvest_channel", "")
    ph_storage   = result.get("post_harvest_storage", "")
    ph_net_gain  = result.get("post_harvest_net_gain", 0.0) or 0.0
    w_signal     = result.get("weather_signal", "NORMAL")
    w_urgency    = result.get("weather_urgency", "LOW")

    if ph_advisory:
        # Weather alert banner — only show if not NORMAL
        if w_signal != "NORMAL":
            urgency_color = {"HIGH": "#d32f2f", "MEDIUM": "#f57c00"}.get(w_urgency, "#388e3c")
            urgency_icon  = {"HIGH": "🚨", "MEDIUM": "⚠️"}.get(w_urgency, "ℹ️")
            st.markdown(
                f'<div style="background:{urgency_color};color:white;padding:10px 16px;'
                f'border-radius:8px;margin-bottom:12px;font-weight:600;">'
                f'{urgency_icon} Weather Alert ({w_urgency}): {w_signal.replace("_"," ")}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Action + sell month pill
        action_color = {"SELL_IMMEDIATELY": "#d32f2f", "WAIT": "#388e3c",
                        "SELL_THIS_WEEK": "#f57c00", "SELL_SOON": "#f57c00"}.get(ph_action, "#555")
        action_label = ph_action.replace("_", " ")
        sell_label   = f"Sell by: {ph_month}" if ph_month and ph_month != "immediately" \
                       else "Sell: immediately"
        st.markdown(
            f'<div style="display:flex;gap:10px;margin-bottom:12px;">'
            f'<span style="background:{action_color};color:white;padding:4px 14px;'
            f'border-radius:20px;font-weight:700;">{action_label}</span>'
            f'<span style="background:#e0e0e0;color:#333;padding:4px 14px;'
            f'border-radius:20px;">📅 {sell_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Main advisory text
        st.markdown(
            f'<div class="agent-section">{ph_advisory}</div>',
            unsafe_allow_html=True,
        )

        # Three detail cards in columns
        col_ph1, col_ph2, col_ph3 = st.columns(3)

        with col_ph1:
            st.markdown("**📍 Selling Channel**")
            if ph_channel:
                st.info(ph_channel)

        with col_ph2:
            st.markdown("**🏪 Storage**")
            if ph_storage:
                st.info(ph_storage)

        with col_ph3:
            st.markdown("**💰 Net Gain if Wait**")
            if ph_net_gain and ph_net_gain > 0:
                st.success(f"Rs {ph_net_gain:,.0f} / quintal")
            else:
                st.warning("Sell now — storage not viable")

    else:
        st.info("Post-harvest advisory not available for this run.")

    # Government Policy expander (kept from original)
    with st.expander("📋 Government Policy & MSP Details", expanded=False):
        st.text(result.get("policy_note", ""))

    # ── Raw State Debug (optional) ─────────────────────────────────────────────
    with st.expander("🔧 Debug: Full Agent State"):
        import json
        debug_state = {k: v for k, v in result.items() if k != "policy_note"}
        st.json(debug_state)

    st.divider()
    st.caption("AgriAdvisor AI · Thesis Project · Agentic AI Framework for Crop Planning · Powered by LangGraph + Groq")
