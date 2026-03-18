"""
Territory dashboard API routes.
"""
from fastapi import APIRouter, HTTPException
from api.data_context import DataContext
from analytics.scoring import opportunity_breakdown
from analytics.event_engine import get_territory_event_opportunities
from ai.review_generator import generate_territory_review

router = APIRouter(prefix="/territory-dashboard", tags=["Territory"])


@router.get("/{territory_id}")
def territory_dashboard(territory_id: str, include_ai_review: bool = False):
    ctx = DataContext.get()

    terr = ctx.territory_scores[ctx.territory_scores["territory_id"] == territory_id]
    if terr.empty:
        raise HTTPException(404, f"Territory {territory_id} not found")
    t = terr.iloc[0].to_dict()

    # Top opportunity stores
    terr_stores = ctx.store_scores[ctx.store_scores["territory_id"] == territory_id]
    top_stores  = terr_stores.nlargest(5, "final_score")[
        ["store_id","store_name","chain","city","final_score","total_opportunity_value","top_issue"]
    ].to_dict("records")

    # Rep performance
    terr_reps = ctx.rep_scores[ctx.rep_scores["territory_id"] == territory_id][
        ["rep_id","rep_name","stores_managed","total_opportunity_value","high_priority_stores"]
    ].to_dict("records")

    # Opportunity breakdown
    opp_breakdown = opportunity_breakdown(
        ctx.opportunities, level="territory", level_id=territory_id,
        store_scores=ctx.store_scores
    ).to_dict("records")

    # Event opportunities
    ev_opps = get_territory_event_opportunities(territory_id, ctx.event_store_map)
    events  = ev_opps.groupby(["event_name","event_type","event_date","expected_attendance"]).agg(
        nearby_stores=("store_id","nunique"), total_uplift=("est_revenue_uplift","sum")
    ).reset_index().to_dict("records")

    payload = {
        "territory_id":             territory_id,
        "territory_name":           t.get("territory_name"),
        "snapshot": {
            "total_stores":            int(t.get("total_stores",0)),
            "stores_with_opportunity": int(t.get("stores_with_opportunity",0)),
            "total_opportunity_value": round(float(t.get("total_opportunity_value",0)),2),
            "total_hidden_opportunity":round(float(t.get("total_hidden_opp",0)),2),
            "promo_compliance_pct":    round(float(t.get("promo_compliance_pct",0)),1),
            "event_opp_stores":        int(t.get("event_opp_stores",0)),
        },
        "opportunity_breakdown":   opp_breakdown,
        "top_opportunity_stores":  top_stores,
        "rep_performance":         terr_reps,
        "event_opportunities":     events,
    }

    if include_ai_review:
        top_issues_str = ", ".join([o["issue_type"] for o in opp_breakdown[:3]])
        top_stores_str = ", ".join([f"{s['store_name']} (${s['total_opportunity_value']:,.0f})" for s in top_stores[:3]])
        payload["ai_review"] = generate_territory_review({
            **payload["snapshot"],
            "territory_name": t.get("territory_name"),
            "top_issues":     top_issues_str,
            "top_stores":     top_stores_str,
        })

    return payload

