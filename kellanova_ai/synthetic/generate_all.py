"""
Master synthetic data generation script.
Run:  python -m synthetic.generate_all
Outputs all CSVs and a DuckDB database to data/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import duckdb
import pandas as pd
from config.settings import DATA_DIR
from synthetic.reference_data    import generate_all_reference
from synthetic.transactional_data import generate_all_transactional
from synthetic.derived_data       import generate_all_derived


def save_table(df: pd.DataFrame, name: str, con: duckdb.DuckDBPyConnection):
    csv_path = DATA_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{csv_path.as_posix()}')")
    print(f"  ✓  {name:<30s}  {len(df):>6,} rows  →  {csv_path.name}")


def main():
    db_path = DATA_DIR / "kellanova_nz.duckdb"
    con = duckdb.connect(str(db_path))
    print("\n══════════════════════════════════════════════════")
    print("  Kellanova NZ Synthetic Data Generator")
    print("══════════════════════════════════════════════════\n")

    t0 = time.time()

    # ── Reference tables ──────────────────────────────────────────────────────
    print("▶  Generating reference tables …")
    ref = generate_all_reference()
    for name, df in ref.items():
        save_table(df, name, con)

    # ── Transactional tables ──────────────────────────────────────────────────
    print("\n▶  Generating transactional tables (this may take ~30 s) …")
    trans = generate_all_transactional(ref)
    for name, df in trans.items():
        save_table(df, name, con)

    # ── Derived / analytics tables ────────────────────────────────────────────
    print("\n▶  Generating derived/analytics tables …")
    derived = generate_all_derived(ref, trans)
    for name, df in derived.items():
        save_table(df, name, con)

    # ── Quick sanity checks ───────────────────────────────────────────────────
    print("\n▶  Sanity checks …")
    checks = {
        "stores":              50,
        "sales_reps":          10,
        "territories":          5,
        "products":            20,
        "local_events":        15,
    }
    for tbl, expected in checks.items():
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        status = "✓" if n == expected else "⚠"
        print(f"  {status}  {tbl}: {n} rows (expected {expected})")

    pos_n = con.execute("SELECT COUNT(*) FROM pos_sales").fetchone()[0]
    print(f"  ✓  pos_sales: {pos_n:,} rows")

    opp_n = con.execute("SELECT COUNT(*) FROM store_opportunities").fetchone()[0]
    print(f"  ✓  store_opportunities: {opp_n:,} rows")

    elapsed = time.time() - t0
    print(f"\n✅  All data generated in {elapsed:.1f}s")
    print(f"    Database  : {db_path}")
    print(f"    CSV files : {DATA_DIR}\n")
    con.close()
    return {**ref, **trans, **derived}


if __name__ == "__main__":
    main()

