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
def get_districts_by_state() -> dict:
    """Load {state -> [districts]} mapping from district_coordinates.csv."""
    try:
        import pandas as pd
        from pathlib import Path
        coords_path = Path(__file__).parent / "data" / "district_coordinates.csv"
        df = pd.read_csv(coords_path).dropna(subset=["district", "state"])
        result = {}
        for state, grp in df.groupby("state"):
            result[state.strip()] = sorted(grp["district"].str.strip().unique().tolist())
        return result
    except Exception:
        return {"Maharashtra": ["Buldhana", "Nagpur", "Pune"],
                "Karnataka":   ["Bangalore", "Mysuru"],
                "Gujarat":     ["Ahmedabad", "Surat"]}


@st.cache_data
def get_state_list() -> list:
    """All 28 states + UTs."""
    return sorted([
        "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh",
        "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Delhi",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
        "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala",
        "Ladakh", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha",
        "Puducherry", "Punjab", "Rajasthan", "Sikkim",
        "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
        "Uttarakhand", "West Bengal",
    ])


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

    districts_by_state = get_districts_by_state()
    states             = get_state_list()

    selected_state = st.selectbox(
        "State",
        options=states,
        index=states.index("Maharashtra") if "Maharashtra" in states else 0,
        help="Select your state"
    )

    # Districts filtered to selected state only
    state_districts = districts_by_state.get(selected_state, [])
    if not state_districts:
        # Fallback: show all districts if state not found in coords
        all_districts = sorted({d for dlist in districts_by_state.values() for d in dlist})
        state_districts = all_districts

    default_district = "Buldhana" if "Buldhana" in state_districts else state_districts[0] if state_districts else "Buldhana"

    selected_district = st.selectbox(
        "District",
        options=state_districts,
        index=state_districts.index(default_district) if default_district in state_districts else 0,
        help="Select your farming district"
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
        profit_ests   = result.get("profit_estimates", {})
        yield_preds   = result.get("yield_predictions", {})
        price_preds   = result.get("price_predictions", {})    # effective price (policy-adjusted)
        policy_dtls   = result.get("policy_details", {}) or {}

        table_data = []
        for c in SUPPORTED_CROPS:
            pol = policy_dtls.get(c, {})
            mkt_price  = pol.get("msp_per_kg", 0) if pol else 0
            eff_price  = price_preds.get(c, 0)
            msp        = pol.get("effective_price_per_kg", 0) if pol else 0
            subsidy    = pol.get("subsidy_per_ha_per_season", 0) if pol else 0
            table_data.append({
                "Crop":              c.title(),
                "Agro Score":        f"{result.get('agro_scores', {}).get(c, 0):.3f}",
                "Econ Score":        f"{result.get('economic_scores', {}).get(c, 0):.3f}",
                "Combined":          f"{final_scores.get(c, 0):.3f}",
                "Yield (kg/ha)":     f"{yield_preds.get(c, 0):,.0f}",
                "Market Price (₹/kg)": f"{eff_price:.2f}",
                "MSP Floor (₹/kg)":  f"{msp:.2f}" if msp > 0 else "—",
                "Subsidy (₹/ha)":    f"₹{subsidy:,.0f}" if subsidy > 0 else "—",
                "Est. Profit (₹/ha)": f"₹{profit_ests.get(c, 0):,.0f}",
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── SHAP Feature Importance ────────────────────────────────────────────────
    st.subheader("🔍 SHAP Feature Importance")
    shap_vals = result.get("shap_values", {})\

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

        # LIME validation — compact text, no extra chart
        lime_summary = result.get("lime_summary", "")
        if lime_summary:
            with st.expander("🧪 LIME Validation (cross-check of SHAP)", expanded=False):
                st.caption(
                    "LIME (Local Interpretable Model-agnostic Explanations) independently "
                    "explains this single prediction. Consistent direction with SHAP = reliable explanation."
                )
                st.code(lime_summary, language=None)
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

    # ── Policy Impact Transparency ────────────────────────────────────────────
    st.subheader("🏛️ Policy Impact Transparency")
    st.caption("Shows exactly which government policies were applied and their financial effect")

    pol_details = result.get("policy_details", {}) or {}
    rec_pol     = pol_details.get(crop, {}) or {}
    profit_ests = result.get("profit_estimates", {}) or {}
    yield_preds = result.get("yield_predictions", {}) or {}
    price_preds = result.get("price_predictions", {}) or {}

    if rec_pol:
        eff_price   = float(rec_pol.get("effective_price_per_kg", 0))
        msp         = float(rec_pol.get("msp_per_kg", 0))
        bonus       = float(rec_pol.get("state_bonus_per_kg", 0))
        proc_pct    = float(rec_pol.get("procurement_efficiency_pct", 0))
        subsidy     = float(rec_pol.get("subsidy_per_ha_per_season", 0))
        ins_pct     = float(rec_pol.get("insurance_premium_pct", 0.02))
        power_sub   = float(rec_pol.get("power_subsidy_per_ha", 0))
        policy_note_txt = str(rec_pol.get("policy_note", ""))

        yield_val   = float(yield_preds.get(crop, 0))
        market_p    = float(price_preds.get(crop, 0))
        total_profit = float(profit_ests.get(crop, 0))

        # Decompose profit
        from config import INPUT_COSTS
        input_cost  = float(INPUT_COSTS.get(crop, 20000))
        base_rev    = yield_val * market_p
        msp_uplift  = max(0, yield_val * (eff_price - market_p)) if eff_price > market_p else 0
        ins_cost    = ins_pct * 30000  # approximate sum insured

        # Without policy
        profit_without_policy = max(0, base_rev - input_cost)
        profit_with_policy    = total_profit

        # Three columns: price, subsidies, comparison
        cp1, cp2, cp3 = st.columns(3)

        with cp1:
            st.markdown("**💰 Price Floor Applied**")
            if msp > 0 or bonus > 0:
                st.markdown(f"""
<div style='background:#e8f5e9;border-radius:8px;padding:12px;'>
<div style='font-size:0.82rem;color:#555'>Central MSP</div>
<div style='font-size:1.2rem;font-weight:700;color:#1b5e20'>₹{msp:.2f}/kg</div>
<div style='font-size:0.82rem;color:#555;margin-top:6px'>State Bonus</div>
<div style='font-size:1.1rem;font-weight:600;color:#2e7d32'>+₹{bonus:.2f}/kg</div>
<div style='font-size:0.82rem;color:#555;margin-top:6px'>Effective Floor</div>
<div style='font-size:1.2rem;font-weight:700;color:#1b5e20'>₹{msp+bonus:.2f}/kg</div>
<div style='font-size:0.78rem;color:#777;margin-top:4px'>Procurement: {proc_pct:.0f}% of produce</div>
</div>""", unsafe_allow_html=True)
            else:
                st.info("No MSP for this crop — market price only")

        with cp2:
            st.markdown("**🎁 Subsidies & Insurance**")
            st.markdown(f"""
<div style='background:#e3f2fd;border-radius:8px;padding:12px;'>
<div style='font-size:0.82rem;color:#555'>PM-KISAN + State Scheme</div>
<div style='font-size:1.2rem;font-weight:700;color:#0d47a1'>+₹{subsidy:,.0f}/ha</div>
<div style='font-size:0.82rem;color:#555;margin-top:6px'>Power Subsidy</div>
<div style='font-size:1.1rem;font-weight:600;color:#1565c0'>+₹{power_sub:,.0f}/ha</div>
<div style='font-size:0.82rem;color:#555;margin-top:6px'>PMFBY Premium (deducted)</div>
<div style='font-size:1.1rem;font-weight:600;color:#c62828'>-₹{ins_cost:,.0f}/ha</div>
</div>""", unsafe_allow_html=True)

        with cp3:
            st.markdown("**📊 Policy Impact**")
            delta = profit_with_policy - profit_without_policy
            delta_pct = (delta / profit_without_policy * 100) if profit_without_policy > 0 else 0
            delta_color = "#2e7d32" if delta >= 0 else "#c62828"
            arrow = "▲" if delta >= 0 else "▼"
            st.markdown(f"""
<div style='background:#fff8e1;border-radius:8px;padding:12px;'>
<div style='font-size:0.82rem;color:#555'>Without policy</div>
<div style='font-size:1.1rem;font-weight:600;color:#555'>₹{profit_without_policy:,.0f}/ha</div>
<div style='font-size:0.82rem;color:#555;margin-top:6px'>With policy</div>
<div style='font-size:1.2rem;font-weight:700;color:#1b5e20'>₹{profit_with_policy:,.0f}/ha</div>
<div style='font-size:1.0rem;font-weight:700;color:{delta_color};margin-top:6px'>
{arrow} ₹{abs(delta):,.0f}/ha ({delta_pct:+.1f}%)</div>
<div style='font-size:0.75rem;color:#777;margin-top:4px'>Policy contribution</div>
</div>""", unsafe_allow_html=True)

        if policy_note_txt:
            st.caption(f"📌 {policy_note_txt}")

        # Policy profit breakdown bar chart
        st.markdown("**Profit Decomposition**")
        fig_pol, ax_pol = plt.subplots(figsize=(8, 2.2))
        components = []
        values     = []
        colors_pol = []

        if base_rev > 0:
            components.append("Base Revenue\n(yield × market price)")
            values.append(base_rev)
            colors_pol.append("#52b788")
        if msp_uplift > 0:
            components.append(f"MSP Uplift\n(+{proc_pct:.0f}% at floor)")
            values.append(msp_uplift)
            colors_pol.append("#2d6a4f")
        if subsidy > 0:
            components.append("Subsidies\n(PM-KISAN + state)")
            values.append(subsidy)
            colors_pol.append("#74c69d")
        if power_sub > 0:
            components.append("Power Subsidy")
            values.append(power_sub)
            colors_pol.append("#95d5b2")
        components.append("Input Costs\n(deducted)")
        values.append(-input_cost)
        colors_pol.append("#e63946")
        if ins_cost > 0:
            components.append("Insurance\n(deducted)")
            values.append(-ins_cost)
            colors_pol.append("#ff6b6b")

        bar_colors = [c if v >= 0 else c for c, v in zip(colors_pol, values)]
        bars_pol = ax_pol.barh(components[::-1], values[::-1],
                               color=bar_colors[::-1], alpha=0.85, height=0.5)
        ax_pol.axvline(0, color="#333", linewidth=0.8)
        ax_pol.set_xlabel("₹/ha")
        ax_pol.set_title(f"Profit Breakdown for {crop.title()}", fontsize=11, fontweight="bold")
        ax_pol.spines["top"].set_visible(False)
        ax_pol.spines["right"].set_visible(False)
        ax_pol.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
        fig_pol.tight_layout()
        st.pyplot(fig_pol)
        plt.close()

        # Policy comparison for top 3 crops
        with st.expander("📋 Policy Details — Top 3 Crops Compared", expanded=False):
            top3_crops = result.get("top_3_crops", []) or []
            pol_rows = []
            for tc in top3_crops:
                tp = pol_details.get(tc, {}) or {}
                pol_rows.append({
                    "Crop":                tc.title(),
                    "MSP (₹/kg)":         f"₹{tp.get('msp_per_kg',0):.2f}",
                    "State Bonus (₹/kg)":  f"₹{tp.get('state_bonus_per_kg',0):.2f}",
                    "Procurement %":       f"{tp.get('procurement_efficiency_pct',0):.0f}%",
                    "Subsidy (₹/ha)":      f"₹{tp.get('subsidy_per_ha_per_season',0):,.0f}",
                    "PMFBY Premium":       f"{tp.get('insurance_premium_pct',0)*100:.1f}%",
                    "Power Subsidy (₹/ha)": f"₹{tp.get('power_subsidy_per_ha',0):,.0f}",
                    "Policy Note":         str(tp.get('policy_note', '—'))[:80],
                })
            if pol_rows:
                st.dataframe(pd.DataFrame(pol_rows), hide_index=True, use_container_width=True)

    else:
        st.info("Policy details not available for this run.")

    # ── Raw State Debug ────────────────────────────────────────────────────────
    with st.expander("🔧 Debug: Full Agent State"):
        import json
        debug_state = {k: v for k, v in result.items() if k != "policy_note"}
        st.json(debug_state)

    st.divider()
    st.caption("AgriAdvisor AI · Thesis Project · Agentic AI Framework for Crop Planning · Powered by LangGraph + Groq")
