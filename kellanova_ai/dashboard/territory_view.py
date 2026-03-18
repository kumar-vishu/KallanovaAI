"""
Territory Dashboard View
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from streamlit_folium import st_folium

from api.data_context import DataContext
from analytics.scoring import opportunity_breakdown
from analytics.event_engine import get_territory_event_opportunities, event_opportunity_summary
from dashboard.map_utils import make_territory_map
from dashboard.utils import fmt_currency
from ai.review_generator import generate_territory_review


def render_territory(ctx: DataContext, territory_id: str, ollama_ok: bool):
    terr = ctx.territory_scores[ctx.territory_scores["territory_id"] == territory_id].iloc[0]
    st.markdown(f"## 🗺 Territory: {terr.territory_name}")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Stores",        int(terr.total_stores))
    c2.metric("Stores w/ Opps",      int(terr.stores_with_opportunity))
    c3.metric("Opportunity Value",   fmt_currency(float(terr.total_opportunity_value)))
    c4.metric("Promo Compliance",    f"{float(terr.promo_compliance_pct):.1f}%")
    c5.metric("Event Opp Stores",    int(terr.event_opp_stores))

    st.markdown("---")
    col_map, col_right = st.columns([3, 2])

    with col_map:
        st.markdown("### 📍 Store Opportunity Map")
        m = make_territory_map(ctx.stores, ctx.store_scores, ctx.events, territory_id)
        st_folium(m, width=700, height=420, returned_objects=[])

    with col_right:
        # Opportunity breakdown chart
        st.markdown("### 📊 Opportunity Breakdown")
        opp_bd = opportunity_breakdown(
            ctx.opportunities, level="territory", level_id=territory_id,
            store_scores=ctx.store_scores
        )
        if not opp_bd.empty:
            fig = px.pie(
                opp_bd, names="issue_type", values="total_value",
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4,
            )
            fig.update_layout(margin=dict(l=0,r=0,t=10,b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_stores, col_reps = st.columns(2)

    with col_stores:
        st.markdown("### 🏆 Top Opportunity Stores")
        terr_stores = ctx.store_scores[ctx.store_scores["territory_id"] == territory_id]
        top5 = terr_stores.nlargest(5, "final_score")[
            ["store_name","chain","city","total_opportunity_value","top_issue","final_score"]
        ].rename(columns={
            "store_name":"Store","chain":"Chain","city":"City",
            "total_opportunity_value":"Opp Value ($)","top_issue":"Top Issue","final_score":"Score"
        })
        top5["Opp Value ($)"] = top5["Opp Value ($)"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top5, use_container_width=True, hide_index=True)

    with col_reps:
        st.markdown("### 👥 Rep Performance")
        terr_reps = ctx.rep_scores[ctx.rep_scores["territory_id"] == territory_id][
            ["rep_name","stores_managed","total_opportunity_value","high_priority_stores"]
        ].rename(columns={
            "rep_name":"Rep","stores_managed":"Stores",
            "total_opportunity_value":"Opp Value ($)","high_priority_stores":"High Priority"
        })
        terr_reps["Opp Value ($)"] = terr_reps["Opp Value ($)"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(terr_reps, use_container_width=True, hide_index=True)

    # ── Events ────────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Event Opportunities")
    ev_opps = get_territory_event_opportunities(territory_id, ctx.event_store_map)
    if not ev_opps.empty:
        ev_summary = ev_opps.groupby(["event_name","event_type","event_date","expected_attendance"]).agg(
            nearby_stores=("store_id","nunique"),
            total_uplift=("est_revenue_uplift","sum"),
        ).reset_index().sort_values("expected_attendance", ascending=False)
        ev_summary["expected_attendance"] = ev_summary["expected_attendance"].apply(lambda x: f"{x:,}")
        ev_summary["total_uplift"] = ev_summary["total_uplift"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(ev_summary.rename(columns={
            "event_name":"Event","event_type":"Type","event_date":"Date",
            "expected_attendance":"Attendance","nearby_stores":"Nearby Stores","total_uplift":"Est. Uplift"
        }), use_container_width=True, hide_index=True)
    else:
        st.info("No events within range of this territory.")

    # ── AI Review ─────────────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Territory Review")
    if ollama_ok:
        if st.button("Generate AI Territory Review", type="primary"):
            with st.spinner("Generating AI review …"):
                opp_bd = opportunity_breakdown(ctx.opportunities, level="territory",
                                               level_id=territory_id, store_scores=ctx.store_scores)
                top_issues = ", ".join(opp_bd["issue_type"].tolist()[:3])
                top_stores_str = ", ".join(
                    terr_stores.nlargest(3,"final_score")
                    .apply(lambda r: f"{r.store_name} (${r.total_opportunity_value:,.0f})", axis=1)
                    .tolist()
                )
                review = generate_territory_review({
                    "territory_name":          terr.territory_name,
                    "total_stores":            int(terr.total_stores),
                    "stores_with_opportunity": int(terr.stores_with_opportunity),
                    "total_opportunity_value": float(terr.total_opportunity_value),
                    "total_hidden_opp":        float(terr.total_hidden_opp),
                    "promo_compliance_pct":    float(terr.promo_compliance_pct),
                    "event_opp_stores":        int(terr.event_opp_stores),
                    "top_issues":              top_issues,
                    "top_stores":              top_stores_str,
                })
                st.markdown(review)
    else:
        st.info("Start Ollama to enable AI reviews: `ollama serve`")

