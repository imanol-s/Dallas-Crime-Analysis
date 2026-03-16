"""Data quality metrics report against live processed data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def main() -> None:
    model = pd.read_csv(PROCESSED / "model_dataset.csv")
    crime = pd.read_csv(PROCESSED / "crime_zip.csv")
    housing = pd.read_csv(PROCESSED / "housing_zip.csv")
    history = pd.read_csv(PROCESSED / "crime_history_features.csv")
    completeness = pd.read_csv(PROCESSED / "source_completeness_scores.csv")

    print("=" * 60)
    print("DATA QUALITY METRICS REPORT")
    print("=" * 60)
    print()
    print("--- Dataset Dimensions ---")
    print(f"  model_dataset:            {model.shape[0]} rows x {model.shape[1]} cols")
    print(f"  crime_zip:                {crime.shape[0]} rows")
    print(f"  housing_zip:              {housing.shape[0]} rows")
    print(f"  crime_history_features:   {history.shape[0]} rows")
    print(f"  source_completeness:      {completeness.shape[0]} rows")
    print()

    # ── ACCURACY ──
    print("=" * 60)
    print("ACCURACY")
    print("=" * 60)

    dup_zips = int(model["zip"].duplicated().sum())
    print(f"  Duplicate ZIPs:           {dup_zips}  {_status(dup_zips == 0)}")

    id_cols = {"zip", "source", "period_start", "period_end", "as_of_date"}
    obj_cols = [c for c in model.select_dtypes(include=[object]).columns if c not in id_cols]
    print(f"  Unexpected object cols:   {len(obj_cols)}  {_status(len(obj_cols) == 0)}")
    if obj_cols:
        for c in obj_cols:
            print(f"    - {c}")

    rates = ["total_rate_per_1000", "violent_rate_per_1000", "property_rate_per_1000"]
    for r in rates:
        if r in model.columns:
            neg = int((pd.to_numeric(model[r], errors="coerce") < 0).sum())
            print(f"  {r} < 0:  {neg}  {_status(neg == 0)}")

    pop_col = "population" if "population" in model.columns else "population_acs"
    if pop_col in model.columns:
        zero_pop = int((pd.to_numeric(model[pop_col], errors="coerce") <= 0).sum())
        print(f"  {pop_col} <= 0:       {zero_pop}  {_status(zero_pop == 0)}")

    print()

    # ── COMPLETENESS ──
    print("=" * 60)
    print("COMPLETENESS")
    print("=" * 60)

    total_cells = model.shape[0] * model.shape[1]
    missing_cells = int(model.isna().sum().sum())
    missing_pct = missing_cells / total_cells * 100 if total_cells else 0
    print(f"  Cell-level missingness:   {missing_cells}/{total_cells} = {missing_pct:.2f}%  {_status(missing_pct <= 5)}")

    reg_cols = [
        "total_rate_per_1000",
        "median_household_income",
        "poverty_rate",
        "owner_occupied_share",
        "median_gross_rent",
        "educational_attainment",
        "log_home_value",
    ]
    print("  Regression column null rates (threshold: 30%):")
    for col in reg_cols:
        if col in model.columns:
            null_rate = model[col].isna().mean() * 100
            print(f"    {col:40s} {null_rate:5.1f}%  {_status(null_rate <= 30)}")

    high_null = []
    for col in model.columns:
        rate = model[col].isna().mean()
        if rate > 0.30:
            high_null.append((col, rate))
    high_null.sort(key=lambda x: -x[1])
    print(f"  Columns exceeding 30% null: {len(high_null)}")
    for col, rate in high_null[:10]:
        print(f"    {col:40s} {rate:.1%}")

    crime_zips = set(crime["zip"].astype(str))
    housing_zips = set(housing["zip"].astype(str)) if "zip" in housing.columns else set()
    model_zips = set(model["zip"].astype(str))
    print(f"  Crime ZIPs: {len(crime_zips)}, Housing ZIPs: {len(housing_zips)}, Model ZIPs: {len(model_zips)}")
    all_in_crime = model_zips <= crime_zips
    all_in_housing = model_zips <= housing_zips
    print(f"  Model ZIPs all in crime:   {_status(all_in_crime)}")
    print(f"  Model ZIPs all in housing: {_status(all_in_housing)}")

    print()

    # ── TIMELINESS ──
    print("=" * 60)
    print("TIMELINESS")
    print("=" * 60)

    if "crime_history_period_count" in history.columns:
        counts = pd.to_numeric(history["crime_history_period_count"], errors="coerce")
        ge8 = int((counts >= 8).sum())
        lt8 = int((counts < 8).sum())
        ge12 = int((counts >= 12).sum())
        print(f"  Crime history quarters:   min={counts.min():.0f}, max={counts.max():.0f}, median={counts.median():.0f}")
        print(f"    >= 8  (lag-4 eligible):  {ge8}/{len(counts)}  {_status(ge8 > 0)}")
        print(f"    >= 12 (forecast eligible): {ge12}/{len(counts)}")
        print(f"    < 8   (insufficient):   {lt8}/{len(counts)}")

    if "crime_history_sufficient_depth" in history.columns:
        depth = pd.to_numeric(history["crime_history_sufficient_depth"], errors="coerce")
        print(f"    sufficient_depth=1: {int(depth.sum())}, =0: {int((depth == 0).sum())}")

        insuff = history[history["crime_history_sufficient_depth"] == 0]
        if len(insuff) > 0 and "crime_history_lag4_total_rate_per_1000" in insuff.columns:
            lag4_null = insuff["crime_history_lag4_total_rate_per_1000"].isna().mean() * 100
            print(f"    Lag-4 null for insufficient ZIPs: {lag4_null:.0f}%  {_status(lag4_null == 100)}")

    print()

    # ── STATISTICAL BOUNDS ──
    print("=" * 60)
    print("STATISTICAL BOUNDS")
    print("=" * 60)

    bounds = {
        "home_value": (50_000, 3_000_000),
        "total_rate_per_1000": (0, 2_500),
    }
    for col, (lo, hi) in bounds.items():
        if col in model.columns:
            s = pd.to_numeric(model[col], errors="coerce").dropna()
            below = int((s < lo).sum())
            above = int((s > hi).sum())
            print(f"  {col}:")
            print(f"    min={s.min():.0f}, max={s.max():.0f}, mean={s.mean():.0f}, std={s.std():.0f}")
            print(f"    below {lo}: {below}, above {hi}: {above}  {_status(below + above == 0)}")

    if pop_col in model.columns:
        s = pd.to_numeric(model[pop_col], errors="coerce").dropna()
        below = int((s < 1).sum())
        print(f"  {pop_col}:")
        print(f"    min={s.min():.0f}, max={s.max():.0f}")
        print(f"    below 1: {below}  {_status(below == 0)}")

    print()

    # ── SOURCE COMPLETENESS ──
    print("=" * 60)
    print("SOURCE COMPLETENESS")
    print("=" * 60)

    if "source_completeness_overall_score" in model.columns:
        sc = pd.to_numeric(model["source_completeness_overall_score"], errors="coerce").dropna()
        print(f"  Overall score: min={sc.min():.3f}, max={sc.max():.3f}, mean={sc.mean():.3f}")

    if "completeness_ratio" in completeness.columns and "category" in completeness.columns:
        by_cat = completeness.groupby("category")["completeness_ratio"].agg(["mean", "min", "count"])
        print("  By category:")
        for cat, row in by_cat.iterrows():
            print(f"    {str(cat):25s} mean={row['mean']:.3f}  min={row['min']:.3f}  ZIPs={int(row['count'])}")

    print()

    # ── SUMMARY ──
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    checks = {
        "No duplicate ZIPs": dup_zips == 0,
        "No negative rates": all(
            (pd.to_numeric(model[r], errors="coerce") < 0).sum() == 0
            for r in rates if r in model.columns
        ),
        "Population > 0": zero_pop == 0 if pop_col in model.columns else True,
        "Cell missingness <= 5%": missing_pct <= 5,
        "Reg columns <= 30% null": all(
            model[c].isna().mean() <= 0.30 for c in reg_cols if c in model.columns
        ),
        "ZIP join integrity": all_in_crime and all_in_housing,
        "Home value in bounds": (
            (pd.to_numeric(model["home_value"], errors="coerce").dropna() >= 50_000).all()
            and (pd.to_numeric(model["home_value"], errors="coerce").dropna() <= 3_000_000).all()
        ) if "home_value" in model.columns else True,
        "Crime rate in bounds": (
            (pd.to_numeric(model["total_rate_per_1000"], errors="coerce").dropna() >= 0).all()
            and (pd.to_numeric(model["total_rate_per_1000"], errors="coerce").dropna() <= 2_500).all()
        ) if "total_rate_per_1000" in model.columns else True,
    }

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    for name, ok in checks.items():
        print(f"  [{_status(ok):4s}] {name}")
    print()
    print(f"  Result: {passed}/{total} checks passed")


if __name__ == "__main__":
    main()
