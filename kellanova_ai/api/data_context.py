"""
Singleton data context — loads all CSVs once at startup,
trains models, builds FAISS index, and caches everything in memory.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.settings import DATA_DIR
from analytics.scoring       import score_stores, score_reps, score_territories
from analytics.event_engine  import get_event_store_matches
from analytics.hidden_opportunity import train_hidden_opportunity_model
from analytics.promotion_lift     import train_promo_lift_model
from analytics.root_cause         import enrich_opportunities_with_rca
from vector.case_library          import build_case_index, load_case_index


class DataContext:
    """Holds all DataFrames and trained models in memory."""
    _instance: "DataContext | None" = None

    def __init__(self):
        self._loaded = False

    @classmethod
    def get(cls) -> "DataContext":
        if cls._instance is None:
            cls._instance = DataContext()
            cls._instance._load()
        return cls._instance

    def _load(self):
        print("  Loading data context …")
        self.territories  = self._csv("territories")
        self.sales_reps   = self._csv("sales_reps")
        self.stores       = self._csv("stores")
        self.products     = self._csv("products")
        self.demographics = self._csv("demographics")
        self.pos_sales    = self._csv("pos_sales")
        self.execution    = self._csv("retail_execution")
        self.promotions   = self._csv("promotions")
        self.events       = self._csv("local_events")
        self.features     = self._csv("store_features")
        self.opportunities= enrich_opportunities_with_rca(self._csv("store_opportunities"))
        self.hidden       = self._csv("hidden_opportunities")
        self.visit_plan   = self._csv("visit_plan")
        self.cases        = self._csv("case_library")

        # Derived scoring
        self.store_scores = score_stores(self.opportunities, self.hidden, self.stores, self.events)
        self.rep_scores   = score_reps(self.store_scores, self.sales_reps, self.visit_plan)
        self.territory_scores = score_territories(
            self.store_scores, self.rep_scores, self.territories,
            self.opportunities, self.events
        )
        self.event_store_map = get_event_store_matches(self.events, self.stores)

        # Models
        self.promo_model, self.promo_enc = self._load_promo_model()
        self.hidden_model_result         = self._load_hidden_model()

        # Vector index
        if not load_case_index():
            build_case_index(self.cases)

        self._loaded = True
        print("  Data context ready ✓")

    def _csv(self, name: str) -> pd.DataFrame:
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Data file missing: {path}\n"
                f"Run: python -m synthetic.generate_all"
            )
        return pd.read_csv(path)

    def _load_promo_model(self):
        from analytics.promotion_lift import load_promo_lift_model, train_promo_lift_model
        model, enc = load_promo_lift_model()
        if model is None:
            result = train_promo_lift_model(
                self.pos_sales, self.promotions, self.execution,
                self.demographics, self.stores
            )
            return result["model"], result["encoder"]
        return model, enc

    def _load_hidden_model(self):
        from analytics.hidden_opportunity import train_hidden_opportunity_model
        from config.settings import MODELS_DIR
        if (MODELS_DIR / "hidden_opp_model.pkl").exists():
            return None
        return train_hidden_opportunity_model(self.pos_sales, self.stores, self.demographics)

