"""
FastAPI application entry point.
Run:  uvicorn api.main:app --reload --port 8000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import territory, rep, store
from api.data_context import DataContext
from analytics.visit_planner import optimise_all_reps
from analytics.event_engine import event_opportunity_summary
from ai.review_generator import check_ollama_available

app = FastAPI(
    title="Kellanova NZ Retail Intelligence API",
    description="AI-powered retail execution intelligence platform for CPG field sales.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(territory.router)
app.include_router(rep.router)
app.include_router(store.router)


@app.on_event("startup")
async def startup():
    """Pre-load data context on startup."""
    print("\n Kellanova NZ Retail Intelligence Platform starting …")
    DataContext.get()
    print(" Ready!\n")


# ── Additional endpoints ───────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Kellanova NZ Retail Intelligence",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    ollama = check_ollama_available()
    return {
        "api":    "ok",
        "ollama": ollama,
    }


@app.get("/visit-plan/{rep_id}", tags=["Visit Planning"])
def visit_plan(rep_id: str):
    ctx   = DataContext.get()
    route = optimise_all_reps(ctx.store_scores, ctx.sales_reps, ctx.events)
    if rep_id not in route:
        from fastapi import HTTPException
        raise HTTPException(404, f"Rep {rep_id} not found")
    return {"rep_id": rep_id, "visit_route": route[rep_id].to_dict("records")}


@app.get("/event-opportunities", tags=["Events"])
def event_opportunities(territory_id: str | None = None):
    ctx     = DataContext.get()
    ev_map  = ctx.event_store_map
    if territory_id:
        ev_map = ev_map[ev_map["territory_id"] == territory_id]
    summary = event_opportunity_summary(ev_map)
    return {"events": summary.to_dict("records")}


@app.get("/hidden-opportunities", tags=["Analytics"])
def hidden_opportunities(territory_id: str | None = None, min_opportunity: float = 100.0):
    ctx = DataContext.get()
    df  = ctx.hidden[ctx.hidden["hidden_opportunity"] >= min_opportunity].copy()
    if territory_id:
        store_ids = ctx.stores[ctx.stores["territory_id"] == territory_id]["store_id"].tolist()
        df = df[df["store_id"].isin(store_ids)]
    df = df.merge(ctx.stores[["store_id","store_name","chain","city","rep_id","territory_id"]], on="store_id", how="left")
    return {
        "count":  len(df),
        "stores": df.sort_values("hidden_opportunity", ascending=False).to_dict("records"),
    }


@app.get("/territories", tags=["Reference"])
def list_territories():
    ctx = DataContext.get()
    return ctx.territories.to_dict("records")


@app.get("/reps", tags=["Reference"])
def list_reps():
    ctx = DataContext.get()
    return ctx.sales_reps.to_dict("records")


@app.get("/stores", tags=["Reference"])
def list_stores(territory_id: str | None = None, rep_id: str | None = None):
    ctx = DataContext.get()
    df  = ctx.stores.copy()
    if territory_id:
        df = df[df["territory_id"] == territory_id]
    if rep_id:
        df = df[df["rep_id"] == rep_id]
    return df.to_dict("records")

