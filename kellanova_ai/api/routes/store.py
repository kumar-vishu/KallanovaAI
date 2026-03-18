"""
Store dashboard API routes.
"""
from fastapi import APIRouter, HTTPException
from api.data_context import DataContext
from analytics.root_cause import store_root_cause_summary
from analytics.event_engine import get_store_event_opportunities
from vector.case_library import retrieve_similar_cases, format_cases_for_display
from ai.review_generator import generate_store_review

router = APIRouter(prefix="/store-dashboard", tags=["Store"])


@router.get("/{store_id}")
def store_dashboard(store_id: int, include_ai_review: bool = False):
    ctx = DataContext.get()

    store_row = ctx.stores[ctx.stores["store_id"] == store_id]
    if store_row.empty:
        raise HTTPException(404, f"Store {store_id} not found")
    store = store_row.iloc[0].to_dict()

    # Score row
    score_row = ctx.store_scores[ctx.store_scores["store_id"] == store_id]
    score = score_row.iloc[0].to_dict() if not score_row.empty else {}

    # POS latest 4 weeks
    latest_weeks = sorted(ctx.pos_sales["week"].unique())[-4:]
    recent = ctx.pos_sales[
        (ctx.pos_sales["store_id"] == store_id) & (ctx.pos_sales["week"].isin(latest_weeks))
    ]
    actual_revenue   = round(float(recent["revenue"].sum()), 2)
    expected_revenue = round(float((recent["expected_units"] * recent["price"]).sum()), 2)

    # Execution
    exec_data = ctx.execution[ctx.execution["store_id"] == store_id]
    oos_skus = exec_data[exec_data["stock_status"]=="out_of_stock"]["sku_id"].tolist()
    low_facing = exec_data[exec_data["compliance_rate"] < 0.6]["sku_id"].tolist()
    no_display = exec_data[~exec_data["display_present"].astype(bool)]["sku_id"].tolist()

    # Root cause
    rca = store_root_cause_summary(store_id, ctx.opportunities)

    # Hidden opportunity
    hid_row = ctx.hidden[ctx.hidden["store_id"] == store_id]
    hidden_opp = hid_row.iloc[0].to_dict() if not hid_row.empty else {}

    # Event opportunities
    ev_opps = get_store_event_opportunities(store_id, ctx.event_store_map)
    events  = ev_opps[["event_name","event_type","event_date","distance_km",
                        "recommended_action","est_revenue_uplift"]].to_dict("records")

    # Similar cases (top issue)
    top_issue = rca[0]["issue_type"] if rca else "shelf_compliance"
    query     = f"{top_issue} store execution issue Kellanova NZ"
    similar   = retrieve_similar_cases(query, issue_type=top_issue, top_k=3)
    cases_df  = format_cases_for_display(similar)

    payload = {
        "store_id":   store_id,
        "store_name": store["store_name"],
        "chain":      store["chain"],
        "city":       store["city"],
        "rep_id":     store["rep_id"],
        "performance": {
            "actual_weekly_revenue":   actual_revenue / 4,
            "expected_weekly_revenue": expected_revenue / 4,
            "opportunity_value":       round((expected_revenue - actual_revenue) / 4, 2),
            "hidden_opportunity":      round(float(hidden_opp.get("hidden_opportunity", 0)), 2),
            "predicted_weekly_sales":  round(float(hidden_opp.get("predicted_sales", 0)), 2),
        },
        "detected_issues": {
            "out_of_stock_skus":   oos_skus[:5],
            "low_facing_skus":     low_facing[:5],
            "no_display_skus":     no_display[:5],
        },
        "root_cause_analysis": rca,
        "event_opportunities": events,
        "similar_cases":       cases_df.to_dict("records"),
        "action_plan":         rca[0]["actions"] if rca else [],
    }

    if include_ai_review:
        issues_str = "; ".join([r["issue_type"] for r in rca[:3]])
        rca_str    = "; ".join([r["explanation"][:80] for r in rca[:2]])
        events_str = ", ".join([e["event_name"] for e in events[:2]]) or "None"
        cases_str  = "; ".join([f"{c.get('action_taken','')[:60]}" for c in similar[:2]])
        payload["ai_review"] = generate_store_review({
            **payload["performance"],
            "store_name":    store["store_name"],
            "chain":         store["chain"],
            "city":          store["city"],
            "actual_revenue":   payload["performance"]["actual_weekly_revenue"],
            "expected_revenue": payload["performance"]["expected_weekly_revenue"],
            "issues":        issues_str,
            "root_causes":   rca_str,
            "nearby_events": events_str,
            "similar_cases": cases_str,
        })

    return payload

