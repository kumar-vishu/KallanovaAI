"""
Hidden Opportunity Detection Model
Trains a model to predict expected sales from store attributes + demographics,
then flags stores where actual < predicted (hidden demand).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from config.settings import MODELS_DIR, RANDOM_SEED

MODEL_PATH = MODELS_DIR / "hidden_opp_model.pkl"
ENC_PATH   = MODELS_DIR / "hidden_opp_enc.pkl"


def _build_store_features(
    pos_sales:    pd.DataFrame,
    stores:       pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    store_avg = (
        pos_sales.groupby("store_id")["revenue"]
        .mean().rename("avg_weekly_revenue").reset_index()
    )
    df = store_avg.merge(stores, on="store_id")
    demo_idx = demographics.set_index("region")
    df["median_income"]       = df["region"].map(demo_idx["median_income"])
    df["population_density"]  = df["region"].map(demo_idx["population_density"])
    df["household_size"]      = df["region"].map(demo_idx["household_size"])
    df["ethnicity_index"]     = df["region"].map(demo_idx["ethnicity_index"])
    return df.dropna(subset=["avg_weekly_revenue"])


def train_hidden_opportunity_model(
    pos_sales:    pd.DataFrame,
    stores:       pd.DataFrame,
    demographics: pd.DataFrame,
) -> dict:
    df    = _build_store_features(pos_sales, stores, demographics)
    le_ch = LabelEncoder()
    le_fm = LabelEncoder()
    le_rg = LabelEncoder()
    df["chain_enc"]  = le_ch.fit_transform(df["chain"].astype(str))
    df["format_enc"] = le_fm.fit_transform(df["store_format"].astype(str))
    df["region_enc"] = le_rg.fit_transform(df["region"].astype(str))

    FEATURES = [
        "base_demand_multiplier","median_income","population_density",
        "household_size","ethnicity_index","chain_enc","format_enc","region_enc"
    ]
    X = df[FEATURES]
    y = df["avg_weekly_revenue"]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, random_state=RANDOM_SEED
    )
    model.fit(X, y)

    cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
    print(f"  Hidden opp model trained — CV MAE = ${cv_mae:.2f}")

    joblib.dump(model,  MODEL_PATH)
    joblib.dump((le_ch, le_fm, le_rg), ENC_PATH)

    # Predict and compute hidden opportunity
    df["predicted_revenue"]  = model.predict(X).clip(min=0)
    df["hidden_opportunity"] = (df["predicted_revenue"] - df["avg_weekly_revenue"]).clip(lower=0)
    df["model_version"]      = "v1.0-gbm"

    scores = df[["store_id","predicted_revenue","avg_weekly_revenue","hidden_opportunity","model_version"]].copy()
    scores.columns = ["store_id","predicted_sales","actual_sales","hidden_opportunity","model_version"]
    return {"model": model, "encoders": (le_ch, le_fm, le_rg), "scores": scores.round(2), "cv_mae": cv_mae}


def score_hidden_opportunities(
    pos_sales:    pd.DataFrame,
    stores:       pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """Load model and score all stores. Falls back to rule-based if no model saved."""
    if MODEL_PATH.exists() and ENC_PATH.exists():
        model = joblib.load(MODEL_PATH)
        le_ch, le_fm, le_rg = joblib.load(ENC_PATH)
    else:
        result = train_hidden_opportunity_model(pos_sales, stores, demographics)
        return result["scores"]

    df = _build_store_features(pos_sales, stores, demographics)
    df["chain_enc"]  = le_ch.transform(df["chain"].astype(str).map(
        lambda x: x if x in le_ch.classes_ else le_ch.classes_[0]))
    df["format_enc"] = le_fm.transform(df["store_format"].astype(str).map(
        lambda x: x if x in le_fm.classes_ else le_fm.classes_[0]))
    df["region_enc"] = le_rg.transform(df["region"].astype(str).map(
        lambda x: x if x in le_rg.classes_ else le_rg.classes_[0]))

    FEATURES = [
        "base_demand_multiplier","median_income","population_density",
        "household_size","ethnicity_index","chain_enc","format_enc","region_enc"
    ]
    df["predicted_sales"]    = model.predict(df[FEATURES]).clip(min=0)
    df["actual_sales"]       = df["avg_weekly_revenue"]
    df["hidden_opportunity"] = (df["predicted_sales"] - df["actual_sales"]).clip(lower=0)
    df["model_version"]      = "v1.0-gbm"
    return df[["store_id","predicted_sales","actual_sales","hidden_opportunity","model_version"]].round(2)

