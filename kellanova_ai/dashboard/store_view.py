"""
Store Dashboard — tabbed deep-dive view.
Tabs: Overview | POS Sales | Audit & Compliance | Promotions | Inventory | AI Review
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from streamlit_folium import st_folium

from api.data_context import DataContext
from analytics.root_cause import store_root_cause_summary
from analytics.event_engine import get_store_event_opportunities
from vector.case_library import retrieve_similar_cases
from dashboard.map_utils import make_store_map
from dashboard.utils import fmt_currency
from ai.review_generator import generate_store_review

SEVERITY_EMOJI  = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
SEVERITY_BADGE  = {"Critical": "badge-critical", "High": "badge-high",
                   "Medium": "badge-medium",    "Low": "badge-low"}


def _kpi_card(label: str, value: str, sub: str = "", color: str = "#D52B1E"):
    sub_html = f"<p class='kpi-sub'>{sub}</p>" if sub else ""
    st.markdown(
        f"<div class='kpi-card' style='border-left-color:{color}'>"
        f"<p class='kpi-label'>{label}</p>"
        f"<p class='kpi-value'>{value}</p>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def render_store(ctx: DataContext, store_id: int, ollama_ok: bool):
    from dashboard.store_tabs import (
        render_pos_tab, render_audit_tab, render_promo_tab,
        render_inventory_tab, render_distribution_tab,
    )

    store_row = ctx.stores[ctx.stores["store_id"] == store_id]
    if store_row.empty:
        st.error(f"Store {store_id} not found")
        return
    store = store_row.iloc[0]

    score_row  = ctx.store_scores[ctx.store_scores["store_id"] == store_id]
    score      = score_row.iloc[0] if not score_row.empty else pd.Series(dtype=object)
    hid_row    = ctx.hidden[ctx.hidden["store_id"] == store_id]
    hidden     = hid_row.iloc[0] if not hid_row.empty else pd.Series(dtype=object)
    rep_row    = ctx.sales_reps[ctx.sales_reps["rep_id"] == store.rep_id]
    rep_name   = rep_row.iloc[0]["rep_name"] if not rep_row.empty else str(store.rep_id)

    # ── Store header — single compact strip ──────────────────────────────────
    compliance_pct = float(score.get("compliance_score", 0)) * 100 if not score.empty else 0
    rank = int(score.get("rank", 0)) if not score.empty else 0
    comp_color = "#D52B1E" if compliance_pct < 70 else "#F57C00" if compliance_pct < 85 else "#388E3C"
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"background:#f8f8f8;border-left:4px solid #D52B1E;border-radius:6px;"
        f"padding:6px 14px;margin-bottom:6px'>"
        f"<span style='font-size:15px;font-weight:700;color:#1A1F3D'>🏪 {store.store_name}</span>"
        f"<span style='font-size:12px;color:#555'><b>{store.chain}</b> &nbsp;·&nbsp; {store.city}"
        f" &nbsp;·&nbsp; Rep: <b>{rep_name}</b></span>"
        f"<span style='font-size:13px;font-weight:700;color:{comp_color}'>"
        f"{compliance_pct:.0f}% compliance &nbsp; Rank #{rank}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    latest_weeks = sorted(ctx.pos_sales["week"].unique())[-4:]
    recent = ctx.pos_sales[
        (ctx.pos_sales["store_id"] == store_id) & (ctx.pos_sales["week"].isin(latest_weeks))
    ]
    actual_rev   = recent["revenue"].sum() / 4
    expected_rev = (recent["expected_units"] * recent["price"]).sum() / 4
    opp_val      = max(0, expected_rev - actual_rev)
    hid_val      = float(hidden.get("hidden_opportunity", 0)) if not hidden.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: _kpi_card("Avg Weekly Sales",    fmt_currency(actual_rev),   "Last 4 weeks",       "#D52B1E")
    with k2: _kpi_card("Expected Weekly",     fmt_currency(expected_rev), "Planogram target",   "#1565C0")
    with k3: _kpi_card("Opportunity Gap",     fmt_currency(opp_val),      "vs expected",        "#F57C00" if opp_val > 100 else "#388E3C")
    with k4: _kpi_card("Hidden Opportunity",  fmt_currency(hid_val),      "AI-detected uplift", "#6A1B9A")

    # ── Pre-compute shared data ───────────────────────────────────────────────
    ev_opps   = get_store_event_opportunities(store_id, ctx.event_store_map)
    rca       = store_root_cause_summary(store_id, ctx.opportunities)
    top_issue = rca[0]["issue_type"] if rca else "shelf_compliance"
    cases     = retrieve_similar_cases(f"{top_issue} Kellanova NZ", issue_type=top_issue, top_k=3)

    # ── Tabs — AI Review first ────────────────────────────────────────────────
    tab_ai, tab_ov, tab_pos, tab_audit, tab_promo, tab_inv, tab_dist = st.tabs([
        "🤖 AI Review", "📊 Overview", "💰 POS Sales", "✅ Audit",
        "🎯 Promos", "📦 Inventory", "🚀 Dist. Gaps",
    ])

    with tab_ai:
        _ai_review_tab(store, actual_rev, expected_rev, opp_val, hid_val,
                       rca, ev_opps, cases, hidden, ollama_ok, ctx=ctx)

    with tab_ov:
        _overview_tab(ctx, store_id, store, rca, ev_opps, cases)

    with tab_pos:
        render_pos_tab(ctx, store_id)

    with tab_audit:
        render_audit_tab(ctx, store_id)

    with tab_promo:
        render_promo_tab(ctx, store_id)

    with tab_inv:
        render_inventory_tab(ctx, store_id)

    with tab_dist:
        render_distribution_tab(ctx, store_id)


# ─────────────────────────────────────────────────────────────────────────────
# Overview Tab  (3-column compact layout)
# ─────────────────────────────────────────────────────────────────────────────
def _overview_tab(ctx, store_id, store, rca, ev_opps, cases):
    col_l, col_m, col_r = st.columns([5, 4, 3])

    # ── Left: trend chart + execution alerts ─────────────────────────────────
    with col_l:
        wk = ctx.pos_sales[ctx.pos_sales["store_id"] == store_id].groupby("week").agg(
            actual=("revenue", "sum"),
            expected=("expected_units", lambda x: (x * ctx.pos_sales.loc[x.index, "price"]).sum()),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wk["week"], y=wk["actual"], name="Actual",
                                 line=dict(color="#D52B1E", width=2.5),
                                 fill="tozeroy", fillcolor="rgba(213,43,30,0.06)"))
        fig.add_trace(go.Scatter(x=wk["week"], y=wk["expected"], name="Expected",
                                 line=dict(color="#1565C0", width=2, dash="dash")))
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=8, b=0),
                          legend=dict(orientation="h", y=1.14, font=dict(size=11)),
                          plot_bgcolor="#fafafa", paper_bgcolor="white",
                          yaxis=dict(tickprefix="$", gridcolor="#eee"))
        st.plotly_chart(fig, use_container_width=True)

        ex = ctx.execution[ctx.execution["store_id"] == store_id]
        oos    = ex[ex["stock_status"] == "out_of_stock"]
        lowf   = ex[ex["compliance_rate"] < 0.6]
        nodisp = ex[~ex["display_present"].astype(bool)]
        if len(oos):    st.error(f"🔴 **{len(oos)} SKUs Out of Stock** — restock immediately")
        if len(lowf):   st.warning(f"🟠 **{len(lowf)} SKUs** below planogram facings")
        if len(nodisp): st.warning(f"🟡 **{len(nodisp)} SKUs** missing required display")
        if not len(oos) and not len(lowf) and not len(nodisp):
            st.success("✅ No critical execution issues")

        if rca:
            for i, action in enumerate(rca[0]["actions"][:3], 1):
                st.checkbox(f"{i}. {action}", key=f"action_{store_id}_{i}")

    # ── Middle: map + events ──────────────────────────────────────────────────
    with col_m:
        store_map = make_store_map(store, ev_opps)
        st_folium(store_map, width="100%", height=230, returned_objects=[])

        if not ev_opps.empty:
            for _, ev in ev_opps.head(2).iterrows():
                st.info(
                    f"**{ev.event_name}** · {ev.event_date}  \n"
                    f"📍 {ev.distance_km:.1f} km · {ev.expected_attendance:,} att.  \n"
                    f"Est. uplift: **{fmt_currency(ev.est_revenue_uplift)}**"
                )

    # ── Right: RCA expanders + cases ─────────────────────────────────────────
    with col_r:
        for r in rca[:3]:
            sev   = r.get("severity", "Medium")
            emoji = SEVERITY_EMOJI.get(sev, "⚪")
            badge = SEVERITY_BADGE.get(sev, "badge-grey")
            with st.expander(
                f"{emoji} {r['issue_type'].replace('_',' ').title()} — {fmt_currency(r['total_opp'])}"
            ):
                st.markdown(
                    f"<span class='badge {badge}'>{sev}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(r["explanation"])
                for a in r["actions"][:2]:
                    st.markdown(f"- {a}")

        if cases:
            for c in cases[:2]:
                with st.expander(
                    f"📂 {c.get('case_id','')} ({c.get('similarity_score',0):.2f})"
                ):
                    st.caption(c.get("description", ""))
                    st.markdown(f"**Action:** {c.get('action_taken','')}")
                    st.markdown(f"**Result:** {c.get('result','')}")


# ─────────────────────────────────────────────────────────────────────────────
# AI Review Tab
# ─────────────────────────────────────────────────────────────────────────────
def _ai_review_tab(store, actual_rev, expected_rev, opp_val, hid_val,
                   rca, ev_opps, cases, hidden, ollama_ok, ctx=None):
    st.caption("Powered by Ollama — all processing runs locally, no data leaves the network.")
    if not ollama_ok:
        st.warning("Start Ollama to enable AI reviews: `ollama serve`")
        return
    if st.button("✨ Generate AI Store Review", type="primary"):
        with st.spinner("Generating insights …"):
            store_id = int(store.store_id)

            # ── Multi-period revenue figures ──────────────────────────────────
            all_weeks = sorted(ctx.pos_sales["week"].unique()) if ctx is not None else []
            l4w_weeks  = all_weeks[-4:]  if len(all_weeks) >= 4  else all_weeks
            l13w_weeks = all_weeks[-13:] if len(all_weeks) >= 13 else all_weeks

            def _period_rev(weeks):
                s = ctx.pos_sales[
                    (ctx.pos_sales["store_id"] == store_id) &
                    (ctx.pos_sales["week"].isin(weeks))
                ]
                return float(s["revenue"].sum()), float((s["expected_units"] * s["price"]).sum())

            l4w_act,  l4w_exp  = _period_rev(l4w_weeks)  if ctx is not None else (actual_rev * 4, expected_rev * 4)
            l13w_act, l13w_exp = _period_rev(l13w_weeks) if ctx is not None else (actual_rev * 13, expected_rev * 13)
            ytd_act,  ytd_exp  = _period_rev(all_weeks)  if ctx is not None else (actual_rev * 52, expected_rev * 52)

            # ── Category breakdown (L4W) ──────────────────────────────────────
            cat_perf = []
            if ctx is not None:
                prods = ctx.products[["sku_id", "category"]].copy()
                l4w_sales = ctx.pos_sales[
                    (ctx.pos_sales["store_id"] == store_id) &
                    (ctx.pos_sales["week"].isin(l4w_weeks))
                ].merge(prods, on="sku_id", how="left")
                cat_agg = l4w_sales.groupby("category").agg(
                    actual=("revenue", "sum"),
                    expected=("expected_units", lambda x: (x * l4w_sales.loc[x.index, "price"]).sum()),
                ).reset_index()
                cat_agg["gap"] = cat_agg["actual"] - cat_agg["expected"]
                cat_agg["pct"] = cat_agg.apply(
                    lambda r: (r["gap"] / r["expected"] * 100) if r["expected"] else 0, axis=1
                )
                cat_agg = cat_agg.sort_values("gap")
                cat_perf = cat_agg.to_dict("records")

            # ── Distribution gaps ─────────────────────────────────────────────
            dist_gaps = []
            if ctx is not None:
                from dashboard.store_tabs import _compute_distribution_gaps
                gaps_df = _compute_distribution_gaps(ctx, store_id)
                if not gaps_df.empty:
                    dist_gaps = gaps_df[[
                        "product_name", "stock_status", "peer_avg_units", "weekly_opp"
                    ]].to_dict("records")

            review = generate_store_review({
                "store_name":            store.store_name,
                "chain":                 store.chain,
                "city":                  store.city,
                "actual_revenue":        actual_rev,
                "expected_revenue":      expected_rev,
                "opportunity_value":     opp_val,
                "hidden_opportunity":    hid_val,
                "l4w_actual":            l4w_act,
                "l4w_expected":          l4w_exp,
                "l13w_actual":           l13w_act,
                "l13w_expected":         l13w_exp,
                "ytd_actual":            ytd_act,
                "ytd_expected":          ytd_exp,
                "category_performance":  cat_perf,
                "distribution_gaps":     dist_gaps,
                "issues":        "; ".join([r["issue_type"] for r in rca[:3]]),
                "root_causes":   "; ".join([r["explanation"][:80] for r in rca[:2]]),
                "nearby_events": ", ".join([ev.event_name for _, ev in ev_opps.iterrows()][:2]) or "None",
                "similar_cases": "; ".join([c.get("action_taken", "")[:60] for c in cases[:2]]),
            })
            st.markdown(review)

