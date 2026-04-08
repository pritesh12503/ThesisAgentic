# 🌾 AgriAdvisor AI
### An Explainable Agentic AI Framework Integrating Agronomic and Market Intelligence for Crop Planning

> **Thesis Project** | Agentic AI | LangGraph + Groq + SHAP + Streamlit

---

## 📐 Architecture Overview

```
DATA SOURCES
├── thesis_agronomic_dataset_clean.csv     → Agronomic Agent
├── Custom_Crops_yield_Historical_Dataset.csv → Market Agent
└── Price_Agriculture_commodities_Week.csv → Market Agent

LANGGRAPH PIPELINE
START
  └─▶ user_input_node
        ├─▶ agronomic_agent_node  (parallel)
        │     • Filters soil+climate by district
        │     • Random Forest → Score_agro per crop
        │     • SHAP feature importance
        │     • Groq LLM → agronomic reasoning text
        │
        └─▶ market_agent_node     (parallel)
              • Yield prediction via trend regression
              • Price lookup from price dataset
              • Profit = yield × price − input_cost
              • Score_economic = normalized profit
              • Groq LLM → market reasoning text
                    │
                    ▼ (fan-in)
              orchestrator_node
                • Crop* = argmax_c [w1·Score_agro + w2·Score_economic]
                    │
                    ▼
              explainability_node
                • SHAP summary
                • Selling-time strategy
                • MSP / policy note
                • Groq LLM → final advisory text
                    │
                    ▼
                   END
```

---

## 📁 Project Structure

```
crop_advisor/
│
├── app.py                        # Streamlit UI (main entry point)
├── graph.py                      # LangGraph assembly + run_crop_advisor()
├── state.py                      # Shared TypedDict state schema
├── config.py                     # All constants, paths, crop mappings
├── requirements.txt
├── .env.example                  # Rename to .env and add your Groq key
│
├── agents/
│   ├── agronomic_agent.py        # Agronomic Agent node
│   ├── market_agent.py           # Market Agent node
│   ├── orchestrator_agent.py     # Orchestrator node
│   └── explainability_agent.py   # Explainability + Post-Harvest node
│
├── utils/
│   ├── data_utils.py             # CSV loading, district normalization, filtering
│   └── model_utils.py            # Train/save/load RF model, SHAP computation
│
├── models/                       # Auto-created on first run
│   └── agronomic_model.pkl       # Saved trained model (auto-generated)
│
└── data/                         # ← PUT YOUR CSV FILES HERE
    ├── thesis_agronomic_dataset_clean.csv
    ├── Custom_Crops_yield_Historical_Dataset.csv
    └── Price_Agriculture_commodities_Week.csv
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone / download the project
```bash
# If using git
git clone <your-repo-url>
cd crop_advisor

# Or just download and unzip the folder
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your datasets
Copy your three CSV files into the `data/` folder:
```
data/thesis_agronomic_dataset_clean.csv
data/Custom_Crops_yield_Historical_Dataset.csv
data/Price_Agriculture_commodities_Week.csv
```

### Step 5 — Get a Groq API key (free)
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free) → Create API Key
3. Copy the key (starts with `gsk_...`)

Either:
- Paste it into the Streamlit sidebar when the app runs, **OR**
- Create a `.env` file:
```bash
cp .env.example .env
# Edit .env and paste your key
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 6 — Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

> **Note:** On the first run, the agronomic model will be trained automatically.
> This takes ~30 seconds. After that it's saved to `models/agronomic_model.pkl`
> and loads instantly on subsequent runs.

---

## 🧪 Test Without UI (CLI)

You can test the full pipeline from the command line:

```bash
python graph.py
```

Or in a Python script:
```python
from graph import run_crop_advisor

result = run_crop_advisor(
    district="buldhana",
    state_name="maharashtra",
    season="Kharif",
    w1=0.5,    # 50% weight to agronomic score
    w2=0.5,    # 50% weight to economic score
)

print("Recommended crop:", result["recommended_crop"])
print("Final scores:", result["final_scores"])
print("Explanation:", result["final_explanation"])
```

---

## 🔬 Key Design Decisions

### Why filtering instead of merging?
Each dataset has different granularity (soil = static, climate = annual, price = weekly).
Merging would force artificial joins and introduce NaN cascades.
Instead, each agent filters its own dataset by district, processes independently,
and only the **scores** (not raw data) are combined at the orchestrator level.

### Why LangGraph instead of plain LangChain?
LangGraph provides:
- **Stateful graph execution** — shared state object persists across all nodes
- **Parallel fan-out** — agronomic and market agents run simultaneously
- **Fan-in synchronization** — orchestrator waits for both before running
- **Checkpointing** — graph state can be saved/resumed (useful for debugging)
- The graph structure **is** the architecture — visually self-documenting

### Why Groq instead of OpenAI?
- Free tier with generous rate limits
- LPU inference = extremely fast responses (100+ tokens/sec)
- LangChain-native via `langchain-groq`
- `llama3-8b-8192` is sufficient for reasoning summaries

### Why SHAP for explainability?
- Model-agnostic (works with Random Forest and Gradient Boosting)
- Additive feature attribution = each feature gets a quantified contribution
- TreeExplainer is fast for tree-based models
- Directly answers "why was this crop recommended?" at feature level

---

## 📊 Datasets Used

| Dataset | Rows | Key Columns | Used By |
|---------|------|-------------|---------|
| `thesis_agronomic_dataset_clean.csv` | 119,784 | N, P, K, pH, Temp, Humidity, Rainfall | Agronomic Agent |
| `Custom_Crops_yield_Historical_Dataset.csv` | 50,765 | Crop, Yield_kg_per_ha, District, Year | Market Agent |
| `Price_Agriculture_commodities_Week.csv` | 23,093 | Commodity, Modal Price, District | Market Agent |

**Supported crops:** Rice, Maize, Chickpea, Cotton

---

## 🎓 Thesis Evaluation Metrics

To evaluate your framework for the thesis, run comparisons on:

1. **Accuracy** — Compare recommended crops against ground truth (best-yield crop per district)
2. **Explainability quality** — User study or SHAP consistency score
3. **Baseline comparison** — Single-agent (agro only) vs dual-agent (agro + market)
4. **Weight sensitivity** — How recommendations change as w1/w2 varies
5. **District coverage** — % of districts with data in all three datasets

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: langgraph` | Run `pip install -r requirements.txt` |
| `AuthenticationError: Groq` | Check your GROQ_API_KEY in `.env` or sidebar |
| `No data for district X` | The agent falls back to national average — this is expected |
| Model training error | Check that CSV files are in the `data/` folder with correct names |
| Streamlit not found | `pip install streamlit` |

---

## 📝 Citation

If you use this framework in your thesis, cite as:
```
[Your Name] (2025). An Explainable Agentic AI Framework Integrating
Agronomic and Market Intelligence for Crop Planning.
[Your University], Department of [Your Department].
```

---

*Built with LangGraph · LangChain · Groq (Llama 3) · scikit-learn · SHAP · Streamlit*
