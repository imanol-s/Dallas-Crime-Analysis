"""Transformation helpers for ZIP-level crime, housing, and ACS data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
import pandas as pd
from dallas_crime.acquire.utils import AcquisitionError, load_dfw_zip_set

if TYPE_CHECKING:
    from dallas_crime.config import Settings

DEFAULT_VIOLENT_LABELS = frozenset({"violent"})
DEFAULT_PROPERTY_LABELS = frozenset({"property"})

ZIP_PATTERN = re.compile(r"(\d{5})")
MISSING_STRINGS = {"", "na", "nan", "none", "<na>", "null"}

QA_NON_NEGATIVE_COLUMNS = frozenset(
    {
        "population",
        "median_household_income",
        "median_gross_rent",
        "home_value",
        "median_rent",
        "total_incidents",
        "violent_incidents",
        "property_incidents",
        "other_incidents",
        "total_rate_per_1000",
        "violent_rate_per_1000",
        "property_rate_per_1000",
    }
)
QA_UNIT_INTERVAL_COLUMNS = frozenset(
    {
        "poverty_rate",
        "owner_occupied_share",
        "renter_occupied_share",
        "rent_burden",
        "vacancy_proxy",
        "educational_attainment",
    }
)
QA_TENURE_INTERVAL_COLUMNS = frozenset({"housing_tenure_mix"})
HOUSING_ANNUAL_CHANGE_BOUNDS = (-50.0, 50.0)


def normalize_zip(value: object) -> str | None:
    """Return a normalized 5-digit ZIP code or None for missing values."""

    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in MISSING_STRINGS:
        return None

    match = ZIP_PATTERN.search(text)
    return match.group(1) if match else None


def normalize_zip_series(values: pd.Series) -> pd.Series:
    """Normalize an entire ZIP code series."""

    normalized = values.map(normalize_zip).astype(object)
    return normalized.where(normalized.notna(), None)


def _lower_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _coerce_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_ratio(series: pd.Series) -> pd.Series:
    numeric = _coerce_numeric(series)
    values = np.where((numeric > 1) & (numeric <= 100), numeric / 100, numeric)
    return pd.Series(values, index=numeric.index, dtype="float64")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_num = _coerce_numeric(numerator)
    denominator_num = _coerce_numeric(denominator)
    values = np.where(denominator_num > 0, numerator_num / denominator_num, np.nan)
    return pd.Series(values, index=numerator_num.index, dtype="float64")


def _dedupe_zip_rows(
    frame: pd.DataFrame,
    *,
    zip_col: str = "zip",
    priority_desc: Iterable[str] = (),
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    working = frame.dropna(subset=[zip_col]).copy()
    if working.empty:
        return working

    working["_source_order"] = np.arange(len(working))
    priority_columns = [column for column in priority_desc if column in working.columns]
    sort_columns = [zip_col, *priority_columns, "_source_order"]
    ascending = [True, *([False] * len(priority_columns)), False]

    deduped = (
        working.sort_values(
            by=sort_columns,
            ascending=ascending,
            na_position="last",
            kind="mergesort",
        )
        .drop_duplicates(subset=[zip_col], keep="first")
        .drop(columns=["_source_order"])
    )
    return deduped.reset_index(drop=True)


def _first_available_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _zip_set(frame: pd.DataFrame, *, zip_col: str = "zip") -> set[str]:
    if zip_col not in frame.columns:
        return set()
    normalized = normalize_zip_series(frame[zip_col])
    return set(normalized.dropna().astype(str))


def _build_target_zip_universe(
    *,
    crime_raw: pd.DataFrame,
    housing_raw: pd.DataFrame,
    census_raw: pd.DataFrame,
    crime_zip: pd.DataFrame,
    housing_zip: pd.DataFrame,
    controls: pd.DataFrame,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    source_sets = {
        "crime_raw": _zip_set(crime_raw),
        "housing_raw": _zip_set(housing_raw),
        "census_raw": _zip_set(census_raw),
        "crime_zip": _zip_set(crime_zip),
        "housing_zip": _zip_set(housing_zip),
        "acs_controls": _zip_set(controls),
        "model_dataset": _zip_set(model_df),
    }
    target_zip_set = source_sets["crime_zip"] & source_sets["housing_zip"] & source_sets["acs_controls"]
    all_zips = sorted(set().union(*source_sets.values()))

    rows: list[dict[str, object]] = []
    for zip_code in all_zips:
        provenance_tables = sorted(
            dataset_name for dataset_name, zip_codes in source_sets.items() if zip_code in zip_codes
        )
        rows.append(
            {
                "zip": zip_code,
                "target_rule": "intersection(crime_zip, housing_zip, acs_controls)",
                "provenance_tables": ";".join(provenance_tables),
                "provenance_count": len(provenance_tables),
                "in_crime_raw": int(zip_code in source_sets["crime_raw"]),
                "in_housing_raw": int(zip_code in source_sets["housing_raw"]),
                "in_census_raw": int(zip_code in source_sets["census_raw"]),
                "in_crime_zip": int(zip_code in source_sets["crime_zip"]),
                "in_housing_zip": int(zip_code in source_sets["housing_zip"]),
                "in_acs_controls": int(zip_code in source_sets["acs_controls"]),
                "in_target_universe": int(zip_code in target_zip_set),
                "in_model_dataset": int(zip_code in source_sets["model_dataset"]),
            }
        )

    return pd.DataFrame.from_records(rows)


def _build_duplicate_zip_report(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, frame in datasets.items():
        if "zip" not in frame.columns:
            continue
        zip_counts = normalize_zip_series(frame["zip"]).dropna().value_counts()
        for zip_code, row_count in zip_counts.items():
            if row_count <= 1:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "zip": zip_code,
                    "row_count": int(row_count),
                }
            )

    report = pd.DataFrame.from_records(rows, columns=["dataset", "zip", "row_count"])
    if report.empty:
        return report
    return report.sort_values(["dataset", "zip"], kind="mergesort", ignore_index=True)


def _build_missingness_report(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, frame in datasets.items():
        total_rows = int(len(frame))
        for column in frame.columns:
            missing_count = int(frame[column].isna().sum())
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "total_rows": total_rows,
                    "missing_count": missing_count,
                    "missing_share": (missing_count / total_rows) if total_rows else np.nan,
                    "dtype": str(frame[column].dtype),
                }
            )

    report = pd.DataFrame.from_records(
        rows,
        columns=["dataset", "column", "total_rows", "missing_count", "missing_share", "dtype"],
    )
    if report.empty:
        return report
    return report.sort_values(["dataset", "column"], kind="mergesort", ignore_index=True)


def _build_impossible_values_report(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, frame in datasets.items():
        if frame.empty:
            continue
        zip_values = (
            normalize_zip_series(frame["zip"])
            if "zip" in frame.columns
            else pd.Series([None] * len(frame), index=frame.index)
        )

        for column in sorted(QA_NON_NEGATIVE_COLUMNS & set(frame.columns)):
            values = _coerce_numeric(frame[column])
            mask = values.notna() & (values < 0)
            for idx in frame.index[mask]:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "zip": zip_values.loc[idx],
                        "column": column,
                        "check": "negative_value",
                        "value": float(values.loc[idx]),
                    }
                )

        for column in sorted(QA_UNIT_INTERVAL_COLUMNS & set(frame.columns)):
            values = _coerce_numeric(frame[column])
            mask = values.notna() & ((values < 0) | (values > 1))
            for idx in frame.index[mask]:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "zip": zip_values.loc[idx],
                        "column": column,
                        "check": "outside_unit_interval",
                        "value": float(values.loc[idx]),
                    }
                )

        for column in sorted(QA_TENURE_INTERVAL_COLUMNS & set(frame.columns)):
            values = _coerce_numeric(frame[column])
            mask = values.notna() & ((values < -1) | (values > 1))
            for idx in frame.index[mask]:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "zip": zip_values.loc[idx],
                        "column": column,
                        "check": "outside_tenure_interval",
                        "value": float(values.loc[idx]),
                    }
                )

    report = pd.DataFrame.from_records(rows, columns=["dataset", "zip", "column", "check", "value"])
    if report.empty:
        return report
    return report.sort_values(["dataset", "column", "zip"], kind="mergesort", ignore_index=True)


def _build_outlier_markers(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, frame in datasets.items():
        if frame.empty:
            continue
        zip_values = (
            normalize_zip_series(frame["zip"])
            if "zip" in frame.columns
            else pd.Series([None] * len(frame), index=frame.index)
        )

        for column in frame.columns:
            if column == "zip":
                continue
            values = _coerce_numeric(frame[column])
            non_null = values.dropna()
            if len(non_null) < 4:
                continue

            q1 = float(non_null.quantile(0.25))
            q3 = float(non_null.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            mask = values.notna() & ((values < lower_bound) | (values > upper_bound))
            for idx in frame.index[mask]:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "zip": zip_values.loc[idx],
                        "column": column,
                        "value": float(values.loc[idx]),
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "direction": "low" if float(values.loc[idx]) < lower_bound else "high",
                    }
                )

    report = pd.DataFrame.from_records(
        rows,
        columns=[
            "dataset",
            "zip",
            "column",
            "value",
            "lower_bound",
            "upper_bound",
            "direction",
        ],
    )
    if report.empty:
        return report
    return report.sort_values(["dataset", "column", "zip"], kind="mergesort", ignore_index=True)


def _build_qa_summary(
    *,
    datasets: Mapping[str, pd.DataFrame],
    target_zip_universe: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    missingness_report: pd.DataFrame,
    impossible_report: pd.DataFrame,
    outlier_report: pd.DataFrame,
) -> dict[str, object]:
    dataset_row_counts = {name: int(len(frame)) for name, frame in datasets.items()}
    missing_columns = (
        missingness_report.loc[missingness_report["missing_count"] > 0, ["dataset", "column"]]
        if not missingness_report.empty
        else pd.DataFrame(columns=["dataset", "column"])
    )

    return {
        "dataset_row_counts": dataset_row_counts,
        "target_zip_universe_rows": int(len(target_zip_universe)),
        "target_zip_rows_for_modeling": (
            int(target_zip_universe["in_target_universe"].sum())
            if not target_zip_universe.empty
            else 0
        ),
        "duplicate_zip_rows": int(len(duplicate_report)),
        "duplicate_zip_datasets": (
            sorted(duplicate_report["dataset"].drop_duplicates().tolist())
            if not duplicate_report.empty
            else []
        ),
        "missing_columns_count": int(len(missing_columns)),
        "impossible_value_rows": int(len(impossible_report)),
        "outlier_marker_rows": int(len(outlier_report)),
        "zip_deduplication_rules": {
            "housing_zip": "latest as_of_date, then non-null score, then higher home_value, then latest source order",
            "acs_controls": "highest non-null score, then larger population/income, then latest source order",
            "model_dataset": "one row per ZIP after merged non-null and recency prioritization",
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _filter_to_study_zip_universe(
    frame: pd.DataFrame,
    settings: "Settings",
    *,
    allowed_zips: set[str] | None = None,
    zip_column: str = "zip",
) -> pd.DataFrame:
    if frame.empty or zip_column not in frame.columns:
        return frame.copy()
    filtered = frame.copy()
    normalized_zip = normalize_zip_series(filtered[zip_column])
    mask = normalized_zip.notna()
    if allowed_zips is not None:
        mask = mask & normalized_zip.isin(allowed_zips)
    if settings.study_zip_prefixes:
        mask = mask & normalized_zip.map(settings.allows_study_zip)
    filtered[zip_column] = normalized_zip
    return filtered.loc[mask].copy()


def aggregate_crime_data(
    crime_df: pd.DataFrame,
    population_df: pd.DataFrame,
    *,
    zip_col: str = "zip",
    date_col: str = "incident_date",
    category_col: str = "crime_category",
    population_zip_col: str = "zip",
    population_col: str = "population",
    violent_labels: Iterable[str] = DEFAULT_VIOLENT_LABELS,
    property_labels: Iterable[str] = DEFAULT_PROPERTY_LABELS,
) -> pd.DataFrame:
    """Aggregate incident-level crime records into a ZIP-level feature table."""

    required_crime_columns = {zip_col, date_col, category_col}
    missing = required_crime_columns - set(crime_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"crime_df is missing required columns: {missing_list}")

    required_population_columns = {population_zip_col, population_col}
    missing = required_population_columns - set(population_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"population_df is missing required columns: {missing_list}")

    optional_columns = [column for column in ("latitude", "longitude") if column in crime_df.columns]
    crimes = crime_df[[zip_col, date_col, category_col, *optional_columns]].copy()
    crimes["zip"] = normalize_zip_series(crimes[zip_col])
    crimes["incident_date"] = _coerce_dates(crimes[date_col])
    crimes["crime_category"] = _lower_text(crimes[category_col]).fillna("other")
    crimes = crimes.dropna(subset=["zip", "incident_date"])
    for column in optional_columns:
        crimes[column] = _coerce_numeric(crimes[column])

    if crimes.empty:
        return pd.DataFrame(
            columns=[
                "zip",
                "period_start",
                "period_end",
                "total_incidents",
                "violent_incidents",
                "property_incidents",
                "other_incidents",
                "centroid_latitude",
                "centroid_longitude",
                "population",
                "total_rate_per_1000",
                "violent_rate_per_1000",
                "property_rate_per_1000",
            ]
        )

    violent = {label.strip().lower() for label in violent_labels}
    property_ = {label.strip().lower() for label in property_labels}

    crimes["is_violent"] = crimes["crime_category"].isin(violent).astype(int)
    crimes["is_property"] = crimes["crime_category"].isin(property_).astype(int)

    aggregated = (
        crimes.groupby("zip", as_index=False)
        .agg(
            total_incidents=("incident_date", "size"),
            violent_incidents=("is_violent", "sum"),
            property_incidents=("is_property", "sum"),
            **(
                {
                    "centroid_latitude": ("latitude", "median"),
                    "centroid_longitude": ("longitude", "median"),
                }
                if {"latitude", "longitude"} <= set(crimes.columns)
                else {}
            ),
        )
        .assign(
            other_incidents=lambda frame: (
                frame["total_incidents"] - frame["violent_incidents"] - frame["property_incidents"]
            ).clip(lower=0),
            period_start=crimes["incident_date"].min().normalize(),
            period_end=crimes["incident_date"].max().normalize(),
        )
    )

    population = population_df[[population_zip_col, population_col]].copy()
    population["zip"] = normalize_zip_series(population[population_zip_col])
    population["population"] = _coerce_numeric(population[population_col])
    population = _dedupe_zip_rows(
        population,
        priority_desc=("population",),
    )

    merged = aggregated.merge(population[["zip", "population"]], on="zip", how="left")
    for incident_col, rate_col in (
        ("total_incidents", "total_rate_per_1000"),
        ("violent_incidents", "violent_rate_per_1000"),
        ("property_incidents", "property_rate_per_1000"),
    ):
        merged[rate_col] = np.where(
            merged["population"].fillna(0) > 0,
            (merged[incident_col] / merged["population"]) * 1000,
            np.nan,
        )

    return merged[
        [
            "zip",
            "period_start",
            "period_end",
            "total_incidents",
            "violent_incidents",
            "property_incidents",
            "other_incidents",
            *([column for column in ("centroid_latitude", "centroid_longitude") if column in merged.columns]),
            "population",
            "total_rate_per_1000",
            "violent_rate_per_1000",
            "property_rate_per_1000",
        ]
    ].sort_values("zip", ignore_index=True)


def build_model_dataset(
    crime_zip_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    controls_df: pd.DataFrame,
    *,
    crime_zip_col: str = "zip",
    housing_zip_col: str = "zip",
    controls_zip_col: str = "zip",
    home_value_col: str = "home_value",
) -> pd.DataFrame:
    """Merge ZIP-level crime, housing, and ACS control tables."""

    required_crime_columns = {
        crime_zip_col,
        "total_rate_per_1000",
        "violent_rate_per_1000",
        "property_rate_per_1000",
    }
    missing = required_crime_columns - set(crime_zip_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"crime_zip_df is missing required columns: {missing_list}")

    required_housing_columns = {housing_zip_col, home_value_col}
    missing = required_housing_columns - set(housing_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"housing_df is missing required columns: {missing_list}")

    if controls_zip_col not in controls_df.columns:
        raise KeyError(f"controls_df is missing required column: {controls_zip_col}")

    crime = crime_zip_df.copy()
    crime["zip"] = normalize_zip_series(crime[crime_zip_col])
    for column in ("total_rate_per_1000", "violent_rate_per_1000", "property_rate_per_1000"):
        crime[column] = _coerce_numeric(crime[column])
    if "total_incidents" in crime.columns:
        crime["total_incidents"] = _coerce_numeric(crime["total_incidents"])
    crime["_non_null_score"] = crime[
        ["total_rate_per_1000", "violent_rate_per_1000", "property_rate_per_1000"]
    ].notna().sum(axis=1)
    crime = _dedupe_zip_rows(
        crime,
        priority_desc=("_non_null_score", "total_incidents", "total_rate_per_1000"),
    ).drop(columns=["_non_null_score"], errors="ignore")

    housing = housing_df.copy()
    housing["zip"] = normalize_zip_series(housing[housing_zip_col])
    housing["home_value"] = _coerce_numeric(housing[home_value_col])
    for column in ("median_rent", "annual_change_pct"):
        if column in housing.columns:
            housing[column] = _coerce_numeric(housing[column])
    if "as_of_date" in housing.columns:
        housing["as_of_date"] = _coerce_dates(housing["as_of_date"])
    housing["_non_null_score"] = housing.notna().sum(axis=1)
    housing = _dedupe_zip_rows(
        housing.dropna(subset=["zip", "home_value"]),
        priority_desc=("as_of_date", "_non_null_score", "home_value"),
    ).drop(columns=["_non_null_score"], errors="ignore")

    controls = controls_df.copy()
    controls["zip"] = normalize_zip_series(controls[controls_zip_col])
    for column in controls.columns:
        if column == "zip":
            continue
        controls[column] = _coerce_numeric(controls[column])
    controls["_non_null_score"] = controls.notna().sum(axis=1)
    controls = _dedupe_zip_rows(
        controls,
        priority_desc=("_non_null_score", "population", "median_household_income"),
    ).drop(columns=["_non_null_score"], errors="ignore")

    model_df = crime.merge(housing, on="zip", how="inner", suffixes=("", "_housing"))
    model_df = model_df.merge(controls, on="zip", how="inner", suffixes=("", "_acs"))
    if "as_of_date" in model_df.columns:
        model_df["as_of_date"] = _coerce_dates(model_df["as_of_date"])
    if "source" in model_df.columns:
        model_df["source"] = model_df["source"].astype("string")

    model_df["home_value"] = _coerce_numeric(model_df["home_value"])
    model_df["_non_null_score"] = model_df.notna().sum(axis=1)
    model_df = _dedupe_zip_rows(
        model_df,
        priority_desc=("as_of_date", "_non_null_score", "home_value"),
    ).drop(columns=["_non_null_score"], errors="ignore")
    model_df["log_home_value"] = np.where(
        model_df["home_value"] > 0,
        np.log(model_df["home_value"]),
        np.nan,
    )

    return model_df.sort_values("zip", ignore_index=True)


def prepare_housing_features(housing_df: pd.DataFrame) -> pd.DataFrame:
    """Select the stable housing columns used by the downstream model."""

    frame = housing_df.copy()
    required = {"zip", "home_value"}
    missing = required - set(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"housing_df is missing required columns: {missing_list}")

    keep = ["zip", "home_value"]
    for optional in (
        "as_of_date",
        "annual_change_pct",
        "median_rent",
        "source",
        "source_url",
        "metric_label",
        "supplemental_sources",
        "realtor_listing_price",
        "realtor_listing_price_yy",
        "realtor_active_listing_count",
        "realtor_median_days_on_market",
        "realtor_listing_price_per_square_foot",
        "realtor_pending_ratio",
        "realtor_quality_flag",
        "realtor_hist_months_observed",
        "realtor_hist_listing_price_12m_avg",
        "realtor_hist_listing_price_12m_change",
        "realtor_hist_active_listing_count_12m_avg",
        "realtor_hist_median_days_on_market_12m_avg",
        "realtor_hist_pending_ratio_12m_avg",
        "realtor_hist_quality_flag_12m_max",
    ):
        if optional in frame.columns:
            keep.append(optional)

    frame = frame[keep].copy()
    frame["zip"] = normalize_zip_series(frame["zip"])
    frame["home_value"] = _coerce_numeric(frame["home_value"])
    for column in (
        "annual_change_pct",
        "median_rent",
        "realtor_listing_price",
        "realtor_listing_price_yy",
        "realtor_active_listing_count",
        "realtor_median_days_on_market",
        "realtor_listing_price_per_square_foot",
        "realtor_pending_ratio",
        "realtor_quality_flag",
        "realtor_hist_months_observed",
        "realtor_hist_listing_price_12m_avg",
        "realtor_hist_listing_price_12m_change",
        "realtor_hist_active_listing_count_12m_avg",
        "realtor_hist_median_days_on_market_12m_avg",
        "realtor_hist_pending_ratio_12m_avg",
        "realtor_hist_quality_flag_12m_max",
    ):
        if column in frame.columns:
            frame[column] = _coerce_numeric(frame[column])
    if "annual_change_pct" in frame.columns:
        lower_bound, upper_bound = HOUSING_ANNUAL_CHANGE_BOUNDS
        # Values far outside plausible YoY moves are parser/template artifacts, not usable controls.
        frame["annual_change_pct"] = frame["annual_change_pct"].where(
            frame["annual_change_pct"].between(lower_bound, upper_bound, inclusive="both")
        )
    if "as_of_date" in frame.columns:
        frame["as_of_date"] = _coerce_dates(frame["as_of_date"])
    frame = frame.dropna(subset=["zip", "home_value"])
    frame["_non_null_score"] = frame.notna().sum(axis=1)
    frame = _dedupe_zip_rows(
        frame,
        priority_desc=("as_of_date", "_non_null_score", "home_value"),
    ).drop(columns=["_non_null_score"], errors="ignore")
    return frame


def prepare_housing_history_panel(housing_history_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the long-form historical housing panel."""

    required = {"zip", "period_start", "period_year", "source"}
    missing = required - set(housing_history_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"housing_history_df is missing required columns: {missing_list}")

    frame = housing_history_df.copy()
    frame["zip"] = normalize_zip_series(frame["zip"])
    for column in ("period_start", "period_end"):
        if column in frame.columns:
            frame[column] = _coerce_dates(frame[column])
    for column in (
        "period_year",
        "period_month",
        "price_signal_value",
        "realtor_hist_listing_price",
        "realtor_hist_active_listing_count",
        "realtor_hist_median_days_on_market",
        "realtor_hist_pending_ratio",
        "realtor_hist_quality_flag",
        "fhfa_annual_change_pct",
        "fhfa_hpi",
        "fhfa_hpi_1990_base",
        "fhfa_hpi_2000_base",
    ):
        if column in frame.columns:
            frame[column] = _coerce_numeric(frame[column])

    for column in ("frequency", "source", "source_url", "metric_label", "price_signal_unit"):
        if column in frame.columns:
            frame[column] = frame[column].astype("string")

    frame = frame.dropna(subset=["zip", "period_start", "period_year", "source"]).copy()
    frame["_non_null_score"] = frame.notna().sum(axis=1)
    dedupe_priority = [
        "period_start",
        "_non_null_score",
        "price_signal_value",
    ]
    frame = (
        frame.sort_values(
            ["zip", "period_start", "source", *dedupe_priority],
            ascending=[True, True, True, False, False, False],
            na_position="last",
            kind="mergesort",
        )
        .drop_duplicates(subset=["zip", "period_start", "source"], keep="first")
        .drop(columns=["_non_null_score"], errors="ignore")
    )
    return frame.reset_index(drop=True)


def prepare_census_controls(census_df: pd.DataFrame) -> pd.DataFrame:
    """Select the ACS control columns used by the downstream model."""

    required = [
        "zip",
        "population",
        "median_household_income",
        "poverty_rate",
        "owner_occupied_share",
        "median_gross_rent",
    ]
    missing = set(required) - set(census_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"census_df is missing required columns: {missing_list}")

    frame = census_df.copy()
    frame["zip"] = normalize_zip_series(frame["zip"])
    for column in required:
        if column == "zip":
            continue
        frame[column] = _coerce_numeric(frame[column])
    for column in (
        "population",
        "median_household_income",
        "poverty_rate",
        "owner_occupied_share",
        "median_gross_rent",
        "renter_occupied_share",
        "rent_burden",
        "vacancy_proxy",
        "occupied_housing_units",
        "renter_occupied_units",
        "total_housing_units",
        "vacant_housing_units",
        "bachelors_or_higher_count",
        "education_population_25_plus",
        "bachelors_or_higher_share",
        "college_attainment_share",
        "education_bachelors_plus_share",
        "education_bachelors_plus_count",
        "population_bachelors_or_higher",
        "population_25_plus",
        "educational_attainment_universe",
    ):
        if column in frame.columns:
            numeric = _coerce_numeric(frame[column])
            frame[column] = numeric.where(numeric >= 0)

    owner_share = _normalize_ratio(frame["owner_occupied_share"])
    frame["owner_occupied_share"] = owner_share

    renter_share_column = _first_available_column(frame, ("renter_occupied_share",))
    if renter_share_column is not None:
        renter_share = _normalize_ratio(frame[renter_share_column])
    elif {"renter_occupied_units", "occupied_housing_units"} <= set(frame.columns):
        renter_share = _safe_divide(frame["renter_occupied_units"], frame["occupied_housing_units"])
    else:
        renter_share = pd.Series(np.where(owner_share.notna(), 1 - owner_share, np.nan), index=frame.index)
    frame["renter_occupied_share"] = _normalize_ratio(renter_share)

    rent_burden_column = _first_available_column(
        frame,
        (
            "rent_burden",
            "rent_burden_share",
            "gross_rent_to_income_ratio",
            "median_rent_burden",
        ),
    )
    if rent_burden_column is not None:
        frame["rent_burden"] = _normalize_ratio(frame[rent_burden_column])
    else:
        annual_rent = _coerce_numeric(frame["median_gross_rent"]) * 12
        frame["rent_burden"] = _normalize_ratio(
            _safe_divide(annual_rent, _coerce_numeric(frame["median_household_income"]))
        )

    vacancy_column = _first_available_column(
        frame,
        ("vacancy_proxy", "vacancy_rate", "housing_vacancy_rate", "rental_vacancy_rate"),
    )
    if vacancy_column is not None:
        frame["vacancy_proxy"] = _normalize_ratio(frame[vacancy_column])
    elif {"vacant_housing_units", "occupied_housing_units"} <= set(frame.columns):
        vacant_units = _coerce_numeric(frame["vacant_housing_units"])
        occupied_units = _coerce_numeric(frame["occupied_housing_units"])
        frame["vacancy_proxy"] = _normalize_ratio(_safe_divide(vacant_units, vacant_units + occupied_units))
    elif {"total_housing_units", "occupied_housing_units"} <= set(frame.columns):
        total_units = _coerce_numeric(frame["total_housing_units"])
        occupied_units = _coerce_numeric(frame["occupied_housing_units"])
        frame["vacancy_proxy"] = _normalize_ratio(_safe_divide(total_units - occupied_units, total_units))
    else:
        frame["vacancy_proxy"] = np.nan

    educational_attainment_column = _first_available_column(
        frame,
        (
            "educational_attainment",
            "bachelors_or_higher_share",
            "college_attainment_share",
            "education_bachelors_plus_share",
        ),
    )
    if educational_attainment_column is not None:
        frame["educational_attainment"] = _normalize_ratio(frame[educational_attainment_column])
    else:
        numerator_column = _first_available_column(
            frame,
            (
                "bachelors_or_higher_count",
                "education_bachelors_plus_count",
                "population_bachelors_or_higher",
            ),
        )
        denominator_column = _first_available_column(
            frame,
            (
                "education_population_25_plus",
                "population_25_plus",
                "educational_attainment_universe",
            ),
        )
        if numerator_column is not None and denominator_column is not None:
            frame["educational_attainment"] = _normalize_ratio(
                _safe_divide(frame[numerator_column], frame[denominator_column])
            )
        else:
            frame["educational_attainment"] = np.nan

    tenure_mix_column = _first_available_column(frame, ("housing_tenure_mix",))
    if tenure_mix_column is not None:
        frame["housing_tenure_mix"] = _coerce_numeric(frame[tenure_mix_column])
    else:
        frame["housing_tenure_mix"] = _coerce_numeric(frame["owner_occupied_share"]) - _coerce_numeric(
            frame["renter_occupied_share"]
        )

    frame["poverty_rate"] = _normalize_ratio(frame["poverty_rate"])

    keep = [
        "zip",
        "population",
        "median_household_income",
        "poverty_rate",
        "owner_occupied_share",
        "median_gross_rent",
        "renter_occupied_share",
        "rent_burden",
        "vacancy_proxy",
        "educational_attainment",
        "housing_tenure_mix",
    ]
    frame = frame[keep]
    frame = frame[frame["population"] > 0].copy()
    frame["_non_null_score"] = frame.notna().sum(axis=1)
    frame = _dedupe_zip_rows(
        frame,
        priority_desc=("_non_null_score", "population", "median_household_income"),
    ).drop(columns=["_non_null_score"], errors="ignore")
    return frame


def build_all(settings: "Settings") -> dict[str, str]:
    """Build processed crime, housing, ACS, and model datasets."""

    crime_path = settings.raw_dir / "crime_records.csv"
    housing_path = settings.raw_dir / "housing_market.csv"
    housing_history_path = settings.raw_dir / "housing_market_history.csv"
    census_path = settings.raw_dir / "acs_zcta.csv"

    missing_paths = [path for path in (crime_path, housing_path, census_path) if not path.exists()]
    if missing_paths:
        missing_list = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing raw inputs: {missing_list}")

    crime_raw = pd.read_csv(crime_path)
    housing_raw = pd.read_csv(housing_path)
    housing_history_raw = pd.read_csv(housing_history_path) if housing_history_path.exists() else pd.DataFrame()
    census_raw = pd.read_csv(census_path)

    candidate_zip_pool = set().union(
        _zip_set(crime_raw),
        _zip_set(housing_raw),
        _zip_set(housing_history_raw),
        _zip_set(census_raw),
    )
    dfw_zip_set = candidate_zip_pool.copy()
    crosswalk_cache_path = settings.raw_dir / "zcta_county_crosswalk_2020.csv"
    if candidate_zip_pool:
        try:
            resolved_dfw_zip_set = load_dfw_zip_set(
                sorted(candidate_zip_pool),
                cache_path=crosswalk_cache_path,
                timeout_seconds=max(5, min(settings.acquire_timeout_seconds, 20)),
                max_attempts=1,
                backoff_seconds=settings.acquire_backoff_seconds,
            )
            if resolved_dfw_zip_set:
                dfw_zip_set = resolved_dfw_zip_set
            else:
                print(
                    "[build] Census ZCTA county crosswalk returned no DFW ZIP matches; "
                    "falling back to prefix-only filtering.",
                    flush=True,
                )
        except AcquisitionError as exc:
            print(
                "[build] Census ZCTA county crosswalk unavailable; falling back to prefix-only "
                f"filtering. reason={exc}",
                flush=True,
            )

    crime_raw = _filter_to_study_zip_universe(crime_raw, settings, allowed_zips=dfw_zip_set)
    housing_raw = _filter_to_study_zip_universe(housing_raw, settings, allowed_zips=dfw_zip_set)
    housing_history_raw = _filter_to_study_zip_universe(
        housing_history_raw,
        settings,
        allowed_zips=dfw_zip_set,
    )
    census_raw = _filter_to_study_zip_universe(census_raw, settings, allowed_zips=dfw_zip_set)

    controls = prepare_census_controls(census_raw)
    crime_zip = aggregate_crime_data(
        crime_raw,
        controls[["zip", "population"]],
        zip_col="zip",
        date_col="reported_at",
        category_col="offense_family",
    )
    crime_zip = crime_zip[crime_zip["total_incidents"] >= settings.min_total_incidents_per_zip].copy()
    housing_zip = prepare_housing_features(housing_raw)
    housing_history_panel = (
        prepare_housing_history_panel(housing_history_raw) if not housing_history_raw.empty else pd.DataFrame()
    )
    model_df = build_model_dataset(crime_zip, housing_zip, controls)

    target_zip_universe = _build_target_zip_universe(
        crime_raw=crime_raw,
        housing_raw=housing_raw,
        census_raw=census_raw,
        crime_zip=crime_zip,
        housing_zip=housing_zip,
        controls=controls,
        model_df=model_df,
    )

    duplicate_report = _build_duplicate_zip_report(
        {
            "crime_raw": crime_raw,
            "housing_raw": housing_raw,
            "census_raw": census_raw,
            "crime_zip": crime_zip,
            "housing_zip": housing_zip,
            "acs_controls": controls,
            "model_dataset": model_df,
        }
    )
    missingness_report = _build_missingness_report(
        {
            "crime_zip": crime_zip,
            "housing_zip": housing_zip,
            "acs_controls": controls,
            "model_dataset": model_df,
            "target_zip_universe": target_zip_universe,
        }
    )
    impossible_report = _build_impossible_values_report(
        {
            "crime_zip": crime_zip,
            "housing_zip": housing_zip,
            "acs_controls": controls,
            "model_dataset": model_df,
        }
    )
    outlier_report = _build_outlier_markers(
        {
            "crime_zip": crime_zip,
            "housing_zip": housing_zip,
            "acs_controls": controls,
            "model_dataset": model_df,
        }
    )
    qa_summary = _build_qa_summary(
        datasets={
            "crime_zip": crime_zip,
            "housing_zip": housing_zip,
            "acs_controls": controls,
            "model_dataset": model_df,
            **({"housing_history_panel": housing_history_panel} if not housing_history_panel.empty else {}),
        },
        target_zip_universe=target_zip_universe,
        duplicate_report=duplicate_report,
        missingness_report=missingness_report,
        impossible_report=impossible_report,
        outlier_report=outlier_report,
    )

    outputs = {
        "crime_zip": settings.processed_dir / "crime_zip.csv",
        "housing_zip": settings.processed_dir / "housing_zip.csv",
        "acs_controls": settings.processed_dir / "acs_controls.csv",
        "model_dataset": settings.processed_dir / "model_dataset.csv",
        "target_zip_universe": settings.processed_dir / "target_zip_universe.csv",
        "qa_duplicate_zip_report": settings.processed_dir / "qa_duplicate_zip_report.csv",
        "qa_missingness_report": settings.processed_dir / "qa_missingness_report.csv",
        "qa_impossible_values_report": settings.processed_dir / "qa_impossible_values_report.csv",
        "qa_outlier_markers": settings.processed_dir / "qa_outlier_markers.csv",
        "qa_summary": settings.processed_dir / "qa_summary.json",
    }
    if not housing_history_panel.empty:
        outputs["housing_history_panel"] = settings.processed_dir / "housing_history_panel.csv"
    crime_zip.to_csv(outputs["crime_zip"], index=False)
    housing_zip.to_csv(outputs["housing_zip"], index=False)
    controls.to_csv(outputs["acs_controls"], index=False)
    model_df.to_csv(outputs["model_dataset"], index=False)
    if not housing_history_panel.empty:
        housing_history_panel.to_csv(outputs["housing_history_panel"], index=False)
    target_zip_universe.to_csv(outputs["target_zip_universe"], index=False)
    duplicate_report.to_csv(outputs["qa_duplicate_zip_report"], index=False)
    missingness_report.to_csv(outputs["qa_missingness_report"], index=False)
    impossible_report.to_csv(outputs["qa_impossible_values_report"], index=False)
    outlier_report.to_csv(outputs["qa_outlier_markers"], index=False)
    _write_json(outputs["qa_summary"], qa_summary)

    return {label: str(path) for label, path in outputs.items()}
