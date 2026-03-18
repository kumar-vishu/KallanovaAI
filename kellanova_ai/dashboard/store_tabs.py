"""
Store deep-dive tab renderers.
Each function receives the shared DataContext and the selected store_id.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from dashboard.utils import fmt_currency

# ── Shared helpers ────────────────────────────────────────────────────────────

STOCK_BADGE  = {"in_stock": ("badge-green", "✅ In Stock"),
                "low_stock": ("badge-orange", "⚠ Low Stock"),
                "out_of_stock": ("badge-red", "🔴 Out of Stock")}
DISP_BADGE   = {True: ("badge-green", "✅ Present"), False: ("badge-red", "❌ Missing")}

def _stock_html(status: str) -> str:
    cls, label = STOCK_BADGE.get(str(status), ("badge-grey", status))
    return f"<span class='badge {cls}'>{label}</span>"

def _disp_html(present) -> str:
    key = bool(present)
    cls, label = DISP_BADGE.get(key, ("badge-grey", str(present)))
    return f"<span class='badge {cls}'>{label}</span>"

def _section(title: str):
    st.markdown(f"### {title}")


# ── POS Sales Tab ─────────────────────────────────────────────────────────────

def render_pos_tab(ctx, store_id: int):

    sales  = ctx.pos_sales[ctx.pos_sales["store_id"] == store_id].copy()
    if sales.empty:
        st.info("No sales data for this store.")
        return

    # Last 4 weeks
    last4 = sorted(sales["week"].unique())[-4:]
    recent = sales[sales["week"].isin(last4)]

    prods  = ctx.products[["sku_id", "product_name", "brand", "category"]].copy()
    df     = (recent.groupby("sku_id")
              .agg(units_sold=("units_sold", "sum"),
                   revenue=("revenue", "sum"),
                   expected_rev=("expected_units", lambda x:
                       (x * sales.loc[x.index, "price"]).sum()),
                   promo_weeks=("promo_active", "sum"))
              .reset_index()
              .merge(prods, on="sku_id", how="left"))

    df["variance_pct"] = ((df["revenue"] - df["expected_rev"]) / df["expected_rev"].replace(0, 1) * 100).round(1)
    df = df.sort_values("revenue", ascending=False)

    # Summary KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue (4wk)", fmt_currency(df["revenue"].sum()))
    k2.metric("vs Expected",         fmt_currency(df["expected_rev"].sum()),
              delta=f"{((df['revenue'].sum()/df['expected_rev'].sum()-1)*100):.1f}%")
    k3.metric("Total Units Sold",    f"{df['units_sold'].sum():,}")
    k4.metric("SKUs on Promo",       f"{(df['promo_weeks'] > 0).sum()}")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        # Bar chart — top 10 SKUs
        top10 = df.head(10)
        names = top10["product_name"].str.slice(0, 22)
        fig = go.Figure()
        fig.add_bar(x=names, y=top10["revenue"],     name="Actual",   marker_color="#D52B1E")
        fig.add_bar(x=names, y=top10["expected_rev"], name="Expected", marker_color="#1565C0", opacity=0.55)
        fig.update_layout(barmode="overlay", height=240, margin=dict(l=0,r=0,t=28,b=60),
                          title="Actual vs Expected Revenue — Top 10 SKUs",
                          yaxis=dict(tickprefix="$", gridcolor="#eee"),
                          plot_bgcolor="#fafafa", paper_bgcolor="white",
                          legend=dict(orientation="h", y=1.12),
                          xaxis=dict(tickangle=-35))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        cat_rev = df.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        fig2 = px.pie(cat_rev, names="category", values="revenue", hole=0.45,
                      color_discrete_sequence=["#D52B1E","#1565C0","#F57C00","#388E3C","#6A1B9A","#F9A825"],
                      title="Revenue by Category")
        fig2.update_layout(height=240, margin=dict(l=0,r=0,t=36,b=0),
                           legend=dict(orientation="v", x=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    # SKU table
    tbl = df[["product_name","brand","category","units_sold","revenue","expected_rev","variance_pct","promo_weeks"]].copy()
    tbl.columns = ["Product","Brand","Category","Units","Revenue","Expected","Var %","Promo Wks"]
    tbl["Revenue"]  = tbl["Revenue"].apply(fmt_currency)
    tbl["Expected"] = tbl["Expected"].apply(fmt_currency)
    tbl["Var %"]    = tbl["Var %"].apply(lambda v: f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%")
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=260)


# ── Audit & Compliance Tab ────────────────────────────────────────────────────

def render_audit_tab(ctx, store_id: int):

    ex    = ctx.execution[ctx.execution["store_id"] == store_id].copy()
    if ex.empty:
        st.info("No audit data for this store.")
        return

    prods = ctx.products[["sku_id", "product_name", "brand", "category"]].copy()
    df    = ex.merge(prods, on="sku_id", how="left").sort_values("compliance_rate")

    avg_comp  = df["compliance_rate"].mean() * 100
    oos_n     = (df["stock_status"] == "out_of_stock").sum()
    disp_miss = (~df["display_present"].astype(bool)).sum()
    low_face  = (df["facings_actual"] < df["facings_expected"]).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Compliance",    f"{avg_comp:.1f}%")
    k2.metric("Out of Stock SKUs", str(oos_n),     delta=f"-{oos_n}" if oos_n else None, delta_color="inverse")
    k3.metric("Display Missing",   str(disp_miss), delta=f"-{disp_miss}" if disp_miss else None, delta_color="inverse")
    k4.metric("Under Facings",     str(low_face))

    col_l, col_r = st.columns([3, 2])
    with col_l:
        top = df.sort_values("compliance_rate").head(15)
        names = top["product_name"].str.slice(0, 24)
        colors = ["#D52B1E" if v < 0.6 else "#F57C00" if v < 0.8 else "#388E3C"
                  for v in top["compliance_rate"]]
        fig = go.Figure(go.Bar(
            y=names, x=top["compliance_rate"] * 100,
            orientation="h", marker_color=colors,
            text=[f"{v*100:.0f}%" for v in top["compliance_rate"]], textposition="outside",
        ))
        fig.update_layout(height=260, margin=dict(l=0,r=40,t=28,b=0),
                          title="Compliance Rate by SKU (worst first)",
                          xaxis=dict(range=[0,115], ticksuffix="%", gridcolor="#eee"),
                          plot_bgcolor="#fafafa", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        stock_counts = df["stock_status"].value_counts().reset_index()
        stock_counts.columns = ["status", "count"]
        fig2 = px.pie(stock_counts, names="status", values="count", hole=0.4,
                      color_discrete_map={"in_stock": "#388E3C", "low_stock": "#F57C00", "out_of_stock": "#D52B1E"},
                      title="Stock Status Distribution")
        fig2.update_layout(height=260, margin=dict(l=0,r=0,t=36,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    tbl = df[["product_name","category","facings_actual","facings_expected","compliance_rate","stock_status","display_present"]].copy()
    tbl["compliance_rate"] = tbl["compliance_rate"].apply(lambda v: f"{v*100:.0f}%")
    tbl["stock_status"]    = tbl["stock_status"].apply(lambda s: STOCK_BADGE.get(s, ("",""))[1])
    tbl["display_present"] = tbl["display_present"].apply(lambda b: "✅ Yes" if bool(b) else "❌ No")
    tbl.columns = ["Product","Category","Actual Facings","Expected Facings","Compliance","Stock","Display"]
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=260)


# ── Promotions Tab ────────────────────────────────────────────────────────────

def render_promo_tab(ctx, store_id: int):

    promos = ctx.promotions[ctx.promotions["store_id"] == store_id].copy()
    if promos.empty:
        st.info("No promotions data for this store.")
        return

    prods  = ctx.products[["sku_id", "product_name", "brand", "category", "base_price"]].copy()
    df     = promos.merge(prods, on="sku_id", how="left")
    df["discount_pct"] = ((df["base_price"] - df["promo_price"]) / df["base_price"] * 100).round(1)
    df["promo_start"]  = pd.to_datetime(df["promo_start"])
    df["promo_end"]    = pd.to_datetime(df["promo_end"])
    df["status"]       = df.apply(
        lambda r: "🟢 Active" if r["promo_start"] <= pd.Timestamp("2024-06-15") <= r["promo_end"]
                  else ("🔵 Upcoming" if r["promo_start"] > pd.Timestamp("2024-06-15") else "⚫ Completed"),
        axis=1,
    )
    df = df.sort_values("promo_start", ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Promotions", str(len(df)))
    k2.metric("Active Now",       str((df["status"] == "🟢 Active").sum()))
    k3.metric("Avg Discount",     f"{df['discount_pct'].mean():.1f}%")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        tbl = df[["product_name","category","promo_type","promo_start","promo_end","promo_price","base_price","discount_pct","status"]].copy()
        tbl["promo_start"] = tbl["promo_start"].dt.strftime("%d %b %Y")
        tbl["promo_end"]   = tbl["promo_end"].dt.strftime("%d %b %Y")
        tbl["promo_price"] = tbl["promo_price"].apply(fmt_currency)
        tbl["base_price"]  = tbl["base_price"].apply(fmt_currency)
        tbl["discount_pct"] = tbl["discount_pct"].apply(lambda v: f"{v:.1f}%")
        tbl["promo_type"]   = tbl["promo_type"].str.replace("_", " ").str.title()
        tbl.columns = ["Product","Category","Type","Start","End","Promo $","Base $","Discount","Status"]
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=260)

    with col_r:
        pt_counts = df["promo_type"].str.replace("_"," ").str.title().value_counts().reset_index()
        pt_counts.columns = ["type","count"]
        fig = px.pie(pt_counts, names="type", values="count", hole=0.4,
                     color_discrete_sequence=["#D52B1E","#1565C0","#F57C00","#388E3C"],
                     title="Promotion Types")
        fig.update_layout(height=260, margin=dict(l=0,r=0,t=36,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ── Inventory Signals Tab ─────────────────────────────────────────────────────

def render_inventory_tab(ctx, store_id: int):

    ex    = ctx.execution[ctx.execution["store_id"] == store_id].copy()
    if ex.empty:
        st.info("No execution data for this store.")
        return

    prods = ctx.products[["sku_id", "product_name", "brand", "category"]].copy()
    df    = ex.merge(prods, on="sku_id", how="left")

    # Risk scoring: OOS=1.0, low_stock=0.4, no_display=0.3, compliance<0.6=0.3
    df["risk"] = (
        (df["stock_status"] == "out_of_stock").astype(float)
        + (df["stock_status"] == "low_stock").astype(float) * 0.4
        + (~df["display_present"].astype(bool)).astype(float) * 0.3
        + (df["compliance_rate"] < 0.6).astype(float) * 0.3
    ).clip(upper=1.0)
    df["risk_level"] = pd.cut(df["risk"], bins=[-0.01, 0.2, 0.5, 0.75, 1.01],
                               labels=["🟢 Low", "🟡 Medium", "🟠 High", "🔴 Critical"])
    df = df.sort_values("risk", ascending=False)

    oos_n    = (df["stock_status"] == "out_of_stock").sum()
    low_n    = (df["stock_status"] == "low_stock").sum()
    crit_n   = (df["risk"] >= 0.75).sum()
    avg_risk = df["risk"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Out of Stock",    str(oos_n),  delta=f"-{oos_n}" if oos_n else None, delta_color="inverse")
    k2.metric("Low Stock",       str(low_n),  delta=f"-{low_n}" if low_n else None, delta_color="inverse")
    k3.metric("Critical Risk SKUs", str(crit_n))
    k4.metric("Avg Risk Score",  f"{avg_risk:.2f}")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        top15 = df.head(15)
        names15 = top15["product_name"].str.slice(0, 24)
        bar_colors = ["#D52B1E" if v >= 0.75 else "#F57C00" if v >= 0.5 else
                      "#F9A825" if v >= 0.2 else "#388E3C" for v in top15["risk"]]
        fig = go.Figure(go.Bar(
            y=names15, x=top15["risk"],
            orientation="h", marker_color=bar_colors,
            text=[f"{v:.2f}" for v in top15["risk"]], textposition="outside",
        ))
        fig.update_layout(height=260, margin=dict(l=0,r=40,t=28,b=0),
                          title="Inventory Risk Score by SKU (highest risk first)",
                          xaxis=dict(range=[0,1.2], gridcolor="#eee"),
                          plot_bgcolor="#fafafa", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        cat_risk = df.groupby("category")["risk"].mean().reset_index().sort_values("risk", ascending=False)
        fig2 = px.bar(cat_risk, x="category", y="risk",
                      color="risk", color_continuous_scale=["#388E3C","#F9A825","#F57C00","#D52B1E"],
                      title="Avg Risk by Category", range_color=[0, 1])
        fig2.update_layout(height=260, margin=dict(l=0,r=0,t=36,b=0),
                           yaxis=dict(range=[0,1.1], gridcolor="#eee"),
                           plot_bgcolor="#fafafa", paper_bgcolor="white",
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    tbl = df[["product_name","category","stock_status","display_present","compliance_rate","risk","risk_level"]].copy()
    tbl["compliance_rate"] = tbl["compliance_rate"].apply(lambda v: f"{v*100:.0f}%")
    tbl["stock_status"]    = tbl["stock_status"].apply(lambda s: STOCK_BADGE.get(s, ("",""))[1])
    tbl["display_present"] = tbl["display_present"].apply(lambda b: "✅ Yes" if bool(b) else "❌ No")
    tbl["risk"]            = tbl["risk"].apply(lambda v: f"{v:.2f}")
    tbl.columns = ["Product","Category","Stock Status","Display","Compliance","Risk Score","Risk Level"]
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=260)


# ── Distribution Opportunity Tab ──────────────────────────────────────────────

def _compute_distribution_gaps(ctx, store_id: int) -> pd.DataFrame:
    """
    Find SKUs selling in >70% of same-chain stores but underperforming here.
    Underperformance = this store's avg weekly units < 50% of chain peer average.
    """
    store = ctx.stores[ctx.stores["store_id"] == store_id].iloc[0]
    chain = store["chain"]
    peer_ids = ctx.stores[
        (ctx.stores["chain"] == chain) & (ctx.stores["store_id"] != store_id)
    ]["store_id"].tolist()

    if len(peer_ids) < 2:
        return pd.DataFrame()

    last4 = sorted(ctx.pos_sales["week"].unique())[-4:]
    sales = ctx.pos_sales[ctx.pos_sales["week"].isin(last4)]

    sku_store = (sales.groupby(["store_id", "sku_id"])
                 .agg(avg_units=("units_sold", "mean"),
                      avg_rev=("revenue", "mean"),
                      avg_price=("price", "mean"))
                 .reset_index())

    this_store = sku_store[sku_store["store_id"] == store_id].set_index("sku_id")
    peers_data = sku_store[sku_store["store_id"].isin(peer_ids)]

    peer_agg = (peers_data.groupby("sku_id")
                .agg(peer_avg_units=("avg_units", "mean"),
                     peer_avg_rev=("avg_rev", "mean"),
                     peers_selling=("avg_units", lambda x: (x > 0).sum()))
                .reset_index())
    peer_agg["peer_pct"] = peer_agg["peers_selling"] / len(peer_ids)

    # Only SKUs widely sold by peers
    gaps = peer_agg[peer_agg["peer_pct"] >= 0.7].copy()

    this_vals = this_store[["avg_units", "avg_rev", "avg_price"]].rename(
        columns={"avg_units": "this_units", "avg_rev": "this_rev", "avg_price": "price"}
    )
    gaps = gaps.merge(this_vals, on="sku_id", how="left").fillna(0)

    # Flag underperformers: < 50% of peer average
    gaps["ratio"] = gaps["this_units"] / gaps["peer_avg_units"].replace(0, 1)
    gaps = gaps[gaps["ratio"] < 0.5].copy()

    if gaps.empty:
        return pd.DataFrame()

    gaps["weekly_opp"] = (gaps["peer_avg_units"] - gaps["this_units"]) * gaps["price"]
    gaps["annual_opp"] = gaps["weekly_opp"] * 52

    prods = ctx.products[["sku_id", "product_name", "brand", "category"]].copy()
    gaps = gaps.merge(prods, on="sku_id", how="left")

    ex_idx = ctx.execution[ctx.execution["store_id"] == store_id][
        ["sku_id", "stock_status", "compliance_rate"]
    ].set_index("sku_id")
    gaps["stock_status"]    = gaps["sku_id"].map(
        lambda s: ex_idx.loc[s, "stock_status"] if s in ex_idx.index else "unknown"
    )
    gaps["compliance_rate"] = gaps["sku_id"].map(
        lambda s: float(ex_idx.loc[s, "compliance_rate"]) if s in ex_idx.index else 0.0
    )

    return gaps.sort_values("weekly_opp", ascending=False).reset_index(drop=True)


def render_distribution_tab(ctx, store_id: int):
    store = ctx.stores[ctx.stores["store_id"] == store_id].iloc[0]
    chain = store["chain"]
    peer_count = len(ctx.stores[
        (ctx.stores["chain"] == chain) & (ctx.stores["store_id"] != store_id)
    ])

    st.caption(
        f"SKUs that sell in **>70%** of *{chain}* peer stores (n={peer_count}) "
        f"but generate <50% of the peer average at this store. "
        f"These represent untapped ranging or ranging-compliance opportunities."
    )

    gaps = _compute_distribution_gaps(ctx, store_id)

    if gaps.empty:
        st.success("✅ No significant distribution gaps vs chain peers — this store is well-ranged.")
        return

    total_weekly = gaps["weekly_opp"].sum()
    total_annual = gaps["annual_opp"].sum()
    oos_gaps     = (gaps["stock_status"] == "out_of_stock").sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SKUs with Gaps",      str(len(gaps)))
    k2.metric("Weekly Opportunity",  fmt_currency(total_weekly))
    k3.metric("Annual Opportunity",  fmt_currency(total_annual))
    k4.metric("OOS Gap SKUs",        str(oos_gaps),
              delta=f"-{oos_gaps}" if oos_gaps else None, delta_color="inverse")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        bar_colors = [
            "#D52B1E" if s == "out_of_stock" else
            "#F57C00" if s == "low_stock" else "#1565C0"
            for s in gaps["stock_status"]
        ]
        fig = go.Figure(go.Bar(
            y=gaps["product_name"].str.slice(0, 28),
            x=gaps["weekly_opp"],
            orientation="h",
            marker_color=bar_colors,
            text=[fmt_currency(v) for v in gaps["weekly_opp"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=max(220, len(gaps) * 32 + 60),
            margin=dict(l=0, r=90, t=28, b=0),
            title=f"Weekly Revenue Gap vs {chain} Peers",
            plot_bgcolor="#fafafa", paper_bgcolor="white",
            xaxis=dict(tickprefix="$", gridcolor="#eee"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Out of Stock &nbsp; 🟠 Low Stock &nbsp; 🔵 Listed but underselling")

    with col_r:
        cat_opp = (gaps.groupby("category")["weekly_opp"].sum()
                   .reset_index().sort_values("weekly_opp", ascending=False))
        fig2 = px.bar(cat_opp, x="category", y="weekly_opp",
                      color="weekly_opp",
                      color_continuous_scale=["#F9A825", "#F57C00", "#D52B1E"],
                      title="Opportunity by Category")
        fig2.update_layout(
            height=220, margin=dict(l=0, r=0, t=36, b=0),
            yaxis=dict(tickprefix="$", gridcolor="#eee"),
            plot_bgcolor="#fafafa", paper_bgcolor="white",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    tbl = gaps[[
        "product_name", "brand", "category",
        "this_units", "peer_avg_units", "peer_pct",
        "weekly_opp", "annual_opp", "stock_status", "compliance_rate"
    ]].copy()
    tbl["peer_pct"]       = tbl["peer_pct"].apply(lambda v: f"{v*100:.0f}%")
    tbl["this_units"]     = tbl["this_units"].apply(lambda v: f"{v:.1f}")
    tbl["peer_avg_units"] = tbl["peer_avg_units"].apply(lambda v: f"{v:.1f}")
    tbl["weekly_opp"]     = tbl["weekly_opp"].apply(fmt_currency)
    tbl["annual_opp"]     = tbl["annual_opp"].apply(fmt_currency)
    tbl["compliance_rate"] = tbl["compliance_rate"].apply(lambda v: f"{v*100:.0f}%")
    tbl["stock_status"]   = tbl["stock_status"].apply(lambda s: STOCK_BADGE.get(s, ("", s))[1])
    tbl.columns = [
        "Product", "Brand", "Category",
        "This Store Wkly Units", "Peer Avg Units", "Peers Selling",
        "Weekly Opp", "Annual Opp", "Stock Status", "Compliance"
    ]
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=260)

