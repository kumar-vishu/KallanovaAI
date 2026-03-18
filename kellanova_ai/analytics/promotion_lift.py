"""
Promotion Lift Model
Estimates the incremental sales effect of promotions using gradient boosting.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from config.settings import MODELS_DIR, RANDOM_SEED


MODEL_PATH = MODELS_DIR / "promo_lift_model.pkl"
ENC_PATH   = MODELS_DIR / "promo_lift_enc.pkl"


def _build_features(
    pos_sales:    pd.DataFrame,
    promotions:   pd.DataFrame,
    execution:    pd.DataFrame,
    demographics: pd.DataFrame,
    stores:       pd.DataFrame,
) -> pd.DataFrame:
    """Join all inputs into a flat feature frame for promotion lift modelling."""
    df = pos_sales.copy()

    # Promo price
    promo_lookup = promotions.groupby(["store_id","sku_id"])["promo_price"].mean()
    df["promo_price"] = df.set_index(["store_id","sku_id"]).index.map(
        lambda x: promo_lookup.get(x, np.nan)
    ).values
    df["discount_pct"] = ((df["price"] - df["promo_price"].fillna(df["price"])) / df["price"]).clip(0, 0.5)
    df["promo_flag"]   = df["promo_active"].astype(int)

    # Display flag
    disp_lookup = execution.set_index(["store_id","sku_id"])["display_present"]
    df["display_flag"] = df.set_index(["store_id","sku_id"]).index.map(
        lambda x: int(disp_lookup.get(x, False))
    ).values

    # Demographics
    store_region = stores.set_index("store_id")["region"]
    df["region"] = df["store_id"].map(store_region)
    demo_idx = demographics.set_index("region")
    df["median_income"]      = df["region"].map(demo_idx["median_income"])
    df["population_density"] = df["region"].map(demo_idx["population_density"])
    df["household_size"]     = df["region"].map(demo_idx["household_size"])

    # Store format
    df["store_format"] = df["store_id"].map(stores.set_index("store_id")["store_format"])

    return df.dropna(subset=["units_sold","promo_flag","discount_pct"])


def train_promo_lift_model(
    pos_sales: pd.DataFrame,
    promotions: pd.DataFrame,
    execution:  pd.DataFrame,
    demographics: pd.DataFrame,
    stores:     pd.DataFrame,
) -> dict:
    df = _build_features(pos_sales, promotions, execution, demographics, stores)

    le_fmt = LabelEncoder()
    df["store_format_enc"] = le_fmt.fit_transform(df["store_format"].astype(str))

    FEATURES = ["price","promo_flag","discount_pct","display_flag",
                "median_income","population_density","household_size","store_format_enc"]
    TARGET   = "units_sold"

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 3),
        "r2":  round(r2_score(y_test, y_pred), 3),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le_fmt, ENC_PATH)
    print(f"  Promo lift model trained — MAE={metrics['mae']}  R²={metrics['r2']}")
    return {"model": model, "encoder": le_fmt, "metrics": metrics}


def predict_promo_lift(
    model,
    le_fmt,
    price: float,
    promo_flag: int,
    discount_pct: float,
    display_flag: int,
    median_income: float,
    population_density: float,
    household_size: float,
    store_format: str,
) -> float:
    fmt_enc = le_fmt.transform([store_format])[0] if store_format in le_fmt.classes_ else 0
    X = pd.DataFrame([{
        "price": price, "promo_flag": promo_flag, "discount_pct": discount_pct,
        "display_flag": display_flag, "median_income": median_income,
        "population_density": population_density, "household_size": household_size,
        "store_format_enc": fmt_enc,
    }])
    return round(float(model.predict(X)[0]), 2)


def load_promo_lift_model():
    if MODEL_PATH.exists() and ENC_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(ENC_PATH)
    return None, None

