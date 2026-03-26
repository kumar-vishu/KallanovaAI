"""
Kellanova NZ Retail Intelligence — Streamlit Dashboard
Run:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from api.data_context import DataContext
from dashboard.utils import fmt_currency, kpi_row
from ai.review_generator import check_ollama_available

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kellanova NZ Retail Intelligence",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_RED = "#D52B1E"

st.markdown("""
<style>
  /* ── Base layout — minimal padding ───────────── */
  .block-container {
    padding-top:0.25rem !important;
    padding-bottom:0.5rem !important;
    max-width:1440px;
  }
  /* Crush default Streamlit vertical gaps */
  div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    gap:0.25rem !important;
  }
  .element-container { margin-bottom:0.15rem !important; }

  /* ── Headings — tight margins ─────────────────── */
  h1,h2,h3,h4,h5 {color:#D52B1E; font-weight:700; margin:0 0 2px 0 !important;}
  h1 {font-size:1.2rem !important;}
  h2 {font-size:1.0rem !important;}
  h3 {font-size:0.9rem !important;}
  h4,h5 {font-size:0.82rem !important;}
  p  {margin:0 0 2px 0 !important;}

  /* ── KPI Cards — slim ─────────────────────────── */
  .kpi-card {
    background:#fff; border-left:4px solid #D52B1E;
    border-radius:6px; padding:8px 12px; margin:2px 0;
    box-shadow:0 1px 5px rgba(0,0,0,0.07);
  }
  .kpi-label {color:#888;font-size:10px;text-transform:uppercase;
              letter-spacing:.7px;margin:0 0 2px;}
  .kpi-value {color:#1A1F3D;font-size:20px;font-weight:700;margin:0;}
  .kpi-sub   {color:#aaa;font-size:10px;margin:1px 0 0;}

  /* ── Native st.metric — compact ─────────────── */
  div[data-testid="metric-container"] {
    background:#fff; border-radius:6px;
    padding:6px 10px !important;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);
  }
  div[data-testid="metric-container"] label {font-size:10px !important;}
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size:18px !important;
  }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size:11px !important;
  }

  /* ── Tabs ─────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {gap:2px; border-bottom:2px solid #f0f0f0;}
  .stTabs [data-baseweb="tab"] {
    padding:6px 14px; font-weight:600; font-size:13px;
    border-radius:6px 6px 0 0; color:#555;
  }
  .stTabs [aria-selected="true"] {
    color:#D52B1E !important;
    border-bottom:3px solid #D52B1E !important;
  }
  /* Remove top gap inside tab panels */
  .stTabs [data-baseweb="tab-panel"] { padding-top:0.4rem !important; }

  /* ── Alert boxes — slim ───────────────────────── */
  div[data-testid="stAlert"] { padding:6px 12px !important; margin:3px 0 !important; }

  /* ── Expanders ────────────────────────────────── */
  details { margin:2px 0 !important; }
  summary { padding:4px 8px !important; font-size:13px; }

  /* ── Status Badges ───────────────────────────── */
  .badge        {display:inline-block;padding:2px 8px;border-radius:10px;
                 font-size:11px;font-weight:600;}
  .badge-critical{background:#D52B1E;color:#fff;}
  .badge-high    {background:#F57C00;color:#fff;}
  .badge-medium  {background:#F9A825;color:#333;}
  .badge-low     {background:#388E3C;color:#fff;}
  .badge-green   {background:#388E3C;color:#fff;}
  .badge-orange  {background:#F57C00;color:#fff;}
  .badge-red     {background:#C62828;color:#fff;}
  .badge-grey    {background:#9E9E9E;color:#fff;}
  .badge-blue    {background:#1565C0;color:#fff;}

  /* ── Sidebar dark theme ─────────────────────── */
  [data-testid="stSidebar"] {background:#1A1F3D !important;}
  [data-testid="stSidebar"] * {color:#dce3f0;}
  [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:#fff !important;}
  [data-testid="stSidebar"] hr {border-color:#3a3f5c;}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stRadio label {color:#cdd5e0 !important;}

  /* ── Markdown text — fix spacing & font consistency ── */
  /* Prevent any inherited letter/word-spacing from breaking AI-generated prose */
  .stMarkdown, .stMarkdown p, .stMarkdown li,
  .stMarkdown span, .stMarkdown strong, .stMarkdown em {
    letter-spacing: normal !important;
    word-spacing:   normal !important;
    white-space:    normal !important;
    word-break:     normal !important;
  }
  /* Make inline `code` use the body font so numbers don't render in monospace */
  .stMarkdown code, p code, li code {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background:  rgba(213,43,30,0.10) !important;
    color:       #D52B1E !important;
    border-radius: 3px !important;
    padding:     1px 5px !important;
    font-size:   0.95em !important;
    font-weight: 600 !important;
  }
  /* Bold numbers / currency inside AI markdown should look rich, not code-like */
  .stMarkdown strong {
    color:       #1A1F3D;
    font-weight: 700 !important;
  }

  /* ── Hide Streamlit chrome ───────────────────── */
  footer {visibility:hidden;}
  #MainMenu {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading data …")
def load_ctx():
    return DataContext.get()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar(ctx: DataContext):
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Kellanova_logo.svg/320px-Kellanova_logo.svg.png", width=200)
        st.markdown("## 🌐 Navigation")
        view = st.radio("Dashboard Level", ["🗺 Territory", "👤 Sales Rep", "🏪 Store"], index=0)

        st.markdown("---")
        if "Territory" in view:
            options = ctx.territories[["territory_id","territory_name"]].apply(
                lambda r: f"{r.territory_id} – {r.territory_name}", axis=1
            ).tolist()
            sel = st.selectbox("Select Territory", options)
            territory_id = sel.split(" – ")[0]
            return view, territory_id, None, None

        elif "Rep" in view:
            options = ctx.sales_reps[["rep_id","rep_name","territory_id"]].apply(
                lambda r: f"{r.rep_id} – {r.rep_name} ({r.territory_id})", axis=1
            ).tolist()
            sel = st.selectbox("Select Rep", options)
            rep_id = sel.split(" – ")[0]
            return view, None, rep_id, None

        else:  # Store
            store_opts = ctx.stores[["store_id","store_name","city","chain"]].apply(
                lambda r: f"{r.store_id} – {r.store_name} ({r.city})", axis=1
            ).tolist()
            sel = st.selectbox("Select Store", store_opts)
            store_id = int(sel.split(" – ")[0])
            return view, None, None, store_id

    return view, None, None, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ctx = load_ctx()
    view, territory_id, rep_id, store_id = sidebar(ctx)

    st.markdown(
        "<p style='font-size:13px;color:#888;margin:0 0 4px'>🌟 Kellanova NZ Retail Intelligence</p>",
        unsafe_allow_html=True,
    )
    ollama = check_ollama_available()
    if not ollama["available"]:
        st.warning(
            f"⚠ Ollama not reachable at `{ollama['host']}` — AI reviews disabled. "
            "Check the server is running (`ollama serve`) and the host/port is accessible."
        )
    elif not ollama["model_ready"]:
        st.warning(
            f"⚠ Ollama is running at `{ollama['host']}` but model is not pulled. "
            f"Run on that server: `{ollama['pull_cmd']}`"
        )

    ai_ready = ollama["available"] and ollama["model_ready"]

    if "Territory" in view:
        from dashboard.territory_view import render_territory
        render_territory(ctx, territory_id, ai_ready)
    elif "Rep" in view:
        from dashboard.rep_view import render_rep
        render_rep(ctx, rep_id, ai_ready)
    else:
        from dashboard.store_view import render_store
        render_store(ctx, store_id, ai_ready)


if __name__ == "__main__":
    main()

