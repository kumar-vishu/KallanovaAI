"""
Shared dashboard utilities — import from here, NOT from app.py.
"""


def fmt_currency(val: float) -> str:
    if val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.0f}"


def kpi_row(cols, metrics: list[tuple]):
    """Render a row of KPI metric cards into pre-created st.columns."""
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)

