"""Transformation helpers for ZIP-level crime, housing, and ACS data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
import pandas as pd
from dallas_crime.acquire.utils import AcquisitionError, load_dfw_zip_set
from dallas_crime.utils import _coerce_numeric, _safe_divide

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
ACS_SNAPSHOT_METRICS = (
    "population",
    "median_household_income",
    "poverty_rate",
    "owner_occupied_share",
    "median_gross_rent",
    "renter_occupied_share",
    "housing_tenure_mix",
)
OPTIONAL_ZIP_SIDECAR_FILES = {
    "enrichment": "dfw_zip_enrichment.csv",
    "economic": "dfw_zip_economic_sidecar.csv",
    "real_estate": "dfw_zip_real_estate_sidecar.csv",
    "law_enforcement": "dfw_zip_law_enforcement_sidecar.csv",
    "social_services": "dfw_zip_social_services_sidecar.csv",
    "infrastructure": "dfw_zip_infrastructure_sidecar.csv",
}


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


def _optional_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series for *column*, or NaN series if missing."""
    if column in frame.columns:
        return _coerce_numeric(frame[column])
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _normalize_ratio(series: pd.Series) -> pd.Series:
    numeric = _coerce_numeric(series)
    values = np.where((numeric > 1) & (numeric <= 100), numeric / 100, numeric)
    return pd.Series(values, index=numeric.index, dtype="float64")


def _trend_slope(
    values: pd.Series,
    *,
    x_values: pd.Series | None = None,
) -> float:
    numeric = _coerce_numeric(values).dropna()
    if len(numeric) < 2:
        return np.nan

    if x_values is None:
        x_numeric = np.arange(len(numeric), dtype=float)
    else:
        x_numeric = _coerce_numeric(x_values.loc[numeric.index]).dropna()
        aligned = numeric.loc[x_numeric.index]
        if len(aligned) < 2:
            return np.nan
        numeric = aligned
    return float(np.polyfit(x_numeric, numeric.to_numpy(), 1)[0])


def _format_period_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.normalize().strftime("%Y-%m-%d")
    return text


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
    target_zip_set = (
        source_sets["crime_zip"] & source_sets["housing_zip"] & source_sets["acs_controls"]
    )
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
            values = _coerce_numeric(frame[column]).astype("float64")
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

    optional_columns = [
        column for column in ("latitude", "longitude") if column in crime_df.columns
    ]
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
            *(
                [
                    column
                    for column in ("centroid_latitude", "centroid_longitude")
                    if column in merged.columns
                ]
            ),
            "population",
            "total_rate_per_1000",
            "violent_rate_per_1000",
            "property_rate_per_1000",
        ]
    ].sort_values("zip", ignore_index=True)


def prepare_crime_history_panel(
    crime_df: pd.DataFrame,
    population_df: pd.DataFrame,
    *,
    zip_col: str = "zip",
    date_col: str = "reported_at",
    category_col: str = "offense_family",
    population_zip_col: str = "zip",
    population_col: str = "population",
    period_freq: str = "Q",
    violent_labels: Iterable[str] = DEFAULT_VIOLENT_LABELS,
    property_labels: Iterable[str] = DEFAULT_PROPERTY_LABELS,
) -> pd.DataFrame:
    """Aggregate incident-level crime records into a quarterly ZIP-level history panel."""

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

    crimes = crime_df[[zip_col, date_col, category_col]].copy()
    crimes["zip"] = normalize_zip_series(crimes[zip_col])
    crimes["incident_date"] = _coerce_dates(crimes[date_col])
    crimes["crime_category"] = _lower_text(crimes[category_col]).fillna("other")
    crimes = crimes.dropna(subset=["zip", "incident_date"]).copy()

    panel_columns = [
        "zip",
        "period_start",
        "period_end",
        "period_year",
        "period_quarter",
        "frequency",
        "total_incidents",
        "violent_incidents",
        "property_incidents",
        "other_incidents",
        "population",
        "total_rate_per_1000",
        "violent_rate_per_1000",
        "property_rate_per_1000",
    ]
    if crimes.empty:
        return pd.DataFrame(columns=panel_columns)

    violent = {label.strip().lower() for label in violent_labels}
    property_ = {label.strip().lower() for label in property_labels}
    crimes["is_violent"] = crimes["crime_category"].isin(violent).astype(int)
    crimes["is_property"] = crimes["crime_category"].isin(property_).astype(int)

    periods = crimes["incident_date"].dt.to_period(period_freq)
    crimes["period_start"] = periods.dt.start_time.dt.normalize()
    crimes["period_end"] = periods.dt.end_time.dt.normalize()
    crimes["period_year"] = crimes["period_start"].dt.year
    crimes["period_quarter"] = crimes["period_start"].dt.quarter
    crimes["frequency"] = "quarterly" if period_freq.upper().startswith("Q") else period_freq

    panel = (
        crimes.groupby(
            ["zip", "period_start", "period_end", "period_year", "period_quarter", "frequency"],
            as_index=False,
        )
        .agg(
            total_incidents=("incident_date", "size"),
            violent_incidents=("is_violent", "sum"),
            property_incidents=("is_property", "sum"),
        )
        .assign(
            other_incidents=lambda frame: (
                frame["total_incidents"] - frame["violent_incidents"] - frame["property_incidents"]
            ).clip(lower=0)
        )
    )

    population = population_df[[population_zip_col, population_col]].copy()
    population["zip"] = normalize_zip_series(population[population_zip_col])
    population["population"] = _coerce_numeric(population[population_col])
    population = _dedupe_zip_rows(
        population,
        priority_desc=("population",),
    )

    panel = panel.merge(population[["zip", "population"]], on="zip", how="left")
    for incident_col, rate_col in (
        ("total_incidents", "total_rate_per_1000"),
        ("violent_incidents", "violent_rate_per_1000"),
        ("property_incidents", "property_rate_per_1000"),
    ):
        panel[rate_col] = np.where(
            panel["population"].fillna(0) > 0,
            (panel[incident_col] / panel["population"]) * 1000,
            np.nan,
        )

    return panel[panel_columns].sort_values(["zip", "period_start"], ignore_index=True)


def build_crime_history_features(crime_history_panel: pd.DataFrame) -> pd.DataFrame:
    """Derive per-ZIP temporal crime features from the quarterly history panel."""

    _MIN_QUARTERS_FOR_LAG_FEATURES = 8

    columns = [
        "zip",
        "crime_history_period_count",
        "crime_history_sufficient_depth",
        "crime_history_first_period_start",
        "crime_history_last_period_start",
        "crime_history_years_covered",
        "crime_history_latest_total_incidents",
        "crime_history_latest_total_rate_per_1000",
        "crime_history_lag1_total_rate_per_1000",
        "crime_history_lag2_total_rate_per_1000",
        "crime_history_lag4_total_rate_per_1000",
        "crime_history_latest_quarter_number",
        "crime_history_latest_is_q1",
        "crime_history_latest_is_q2",
        "crime_history_latest_is_q3",
        "crime_history_latest_is_q4",
        "crime_history_q1_mean_rate_per_1000",
        "crime_history_q2_mean_rate_per_1000",
        "crime_history_q3_mean_rate_per_1000",
        "crime_history_q4_mean_rate_per_1000",
        "crime_history_mean_total_incidents",
        "crime_history_mean_total_rate_per_1000",
        "crime_history_total_rate_change_pct",
        "crime_history_total_rate_trend_per_period",
        "crime_history_latest_vs_lag1_rate_change",
        "crime_history_latest_vs_lag4_rate_change",
        "crime_history_rate_momentum_2q",
        "crime_history_rate_acceleration",
    ]
    if crime_history_panel.empty:
        return pd.DataFrame(columns=columns)

    required = {"zip", "period_start", "period_quarter", "total_incidents", "total_rate_per_1000"}
    missing = required - set(crime_history_panel.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"crime_history_panel is missing required columns: {missing_list}")

    panel = crime_history_panel.copy()
    panel["zip"] = normalize_zip_series(panel["zip"])
    panel["period_start"] = _coerce_dates(panel["period_start"])
    panel["period_quarter"] = _coerce_numeric(panel["period_quarter"])
    panel["total_incidents"] = _coerce_numeric(panel["total_incidents"])
    panel["total_rate_per_1000"] = _coerce_numeric(panel["total_rate_per_1000"])
    panel = panel.dropna(subset=["zip", "period_start"]).copy()
    if panel.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for zip_code, zip_frame in panel.groupby("zip", sort=True):
        ordered = zip_frame.sort_values("period_start", ignore_index=True).copy()
        total_rates = ordered["total_rate_per_1000"].dropna()
        latest_row = ordered.iloc[-1]
        latest_quarter = int(latest_row["period_quarter"]) if pd.notna(latest_row["period_quarter"]) else np.nan
        lag1_rate = (
            float(ordered.iloc[-2]["total_rate_per_1000"])
            if len(ordered) >= 2 and pd.notna(ordered.iloc[-2]["total_rate_per_1000"])
            else np.nan
        )
        lag2_rate = (
            float(ordered.iloc[-3]["total_rate_per_1000"])
            if len(ordered) >= 3 and pd.notna(ordered.iloc[-3]["total_rate_per_1000"])
            else np.nan
        )
        lag4_rate = (
            float(ordered.iloc[-5]["total_rate_per_1000"])
            if len(ordered) >= 5 and pd.notna(ordered.iloc[-5]["total_rate_per_1000"])
            else np.nan
        )
        rate_change_pct = np.nan
        rate_trend = np.nan
        if len(total_rates) >= 2:
            first_rate = float(total_rates.iloc[0])
            latest_rate = float(total_rates.iloc[-1])
            if first_rate > 0:
                rate_change_pct = ((latest_rate / first_rate) - 1.0) * 100.0
            rate_trend = _trend_slope(total_rates)

        first_period = ordered["period_start"].min()
        last_period = ordered["period_start"].max()
        quarterly_mean_rates = (
            ordered.groupby("period_quarter")["total_rate_per_1000"].mean().to_dict()
            if "period_quarter" in ordered.columns
            else {}
        )
        latest_rate_value = (
            float(latest_row["total_rate_per_1000"])
            if pd.notna(latest_row["total_rate_per_1000"])
            else np.nan
        )
        latest_vs_lag1 = (
            latest_rate_value - lag1_rate
            if pd.notna(latest_rate_value) and pd.notna(lag1_rate)
            else np.nan
        )
        latest_vs_lag4 = (
            latest_rate_value - lag4_rate
            if pd.notna(latest_rate_value) and pd.notna(lag4_rate)
            else np.nan
        )
        momentum_2q = (
            latest_rate_value - lag2_rate
            if pd.notna(latest_rate_value) and pd.notna(lag2_rate)
            else np.nan
        )
        acceleration = (
            latest_rate_value - (2 * lag1_rate) + lag2_rate
            if pd.notna(latest_rate_value) and pd.notna(lag1_rate) and pd.notna(lag2_rate)
            else np.nan
        )
        sufficient_depth = int(len(ordered) >= _MIN_QUARTERS_FOR_LAG_FEATURES)
        if not sufficient_depth:
            # NaN-out features that require deep history (lag-4 and derivatives)
            lag4_rate = np.nan
            latest_vs_lag4 = np.nan

        rows.append(
            {
                "zip": zip_code,
                "crime_history_period_count": int(len(ordered)),
                "crime_history_sufficient_depth": sufficient_depth,
                "crime_history_first_period_start": first_period,
                "crime_history_last_period_start": last_period,
                "crime_history_years_covered": int(last_period.year - first_period.year + 1),
                "crime_history_latest_total_incidents": float(latest_row["total_incidents"]),
                "crime_history_latest_total_rate_per_1000": latest_rate_value,
                "crime_history_lag1_total_rate_per_1000": lag1_rate,
                "crime_history_lag2_total_rate_per_1000": lag2_rate,
                "crime_history_lag4_total_rate_per_1000": lag4_rate,
                "crime_history_latest_quarter_number": latest_quarter,
                "crime_history_latest_is_q1": int(latest_quarter == 1) if pd.notna(latest_quarter) else 0,
                "crime_history_latest_is_q2": int(latest_quarter == 2) if pd.notna(latest_quarter) else 0,
                "crime_history_latest_is_q3": int(latest_quarter == 3) if pd.notna(latest_quarter) else 0,
                "crime_history_latest_is_q4": int(latest_quarter == 4) if pd.notna(latest_quarter) else 0,
                "crime_history_q1_mean_rate_per_1000": float(quarterly_mean_rates.get(1))
                if 1 in quarterly_mean_rates
                else np.nan,
                "crime_history_q2_mean_rate_per_1000": float(quarterly_mean_rates.get(2))
                if 2 in quarterly_mean_rates
                else np.nan,
                "crime_history_q3_mean_rate_per_1000": float(quarterly_mean_rates.get(3))
                if 3 in quarterly_mean_rates
                else np.nan,
                "crime_history_q4_mean_rate_per_1000": float(quarterly_mean_rates.get(4))
                if 4 in quarterly_mean_rates
                else np.nan,
                "crime_history_mean_total_incidents": float(ordered["total_incidents"].mean()),
                "crime_history_mean_total_rate_per_1000": float(total_rates.mean())
                if not total_rates.empty
                else np.nan,
                "crime_history_total_rate_change_pct": rate_change_pct,
                "crime_history_total_rate_trend_per_period": rate_trend,
                "crime_history_latest_vs_lag1_rate_change": latest_vs_lag1,
                "crime_history_latest_vs_lag4_rate_change": latest_vs_lag4,
                "crime_history_rate_momentum_2q": momentum_2q,
                "crime_history_rate_acceleration": acceleration,
            }
        )

    return pd.DataFrame.from_records(rows, columns=columns).sort_values("zip", ignore_index=True)


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
    crime["_non_null_score"] = (
        crime[["total_rate_per_1000", "violent_rate_per_1000", "property_rate_per_1000"]]
        .notna()
        .sum(axis=1)
    )
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

    # ── Completeness gate: cell-level missingness ≤ 5% ──
    total_cells = model_df.shape[0] * model_df.shape[1]
    if total_cells > 0:
        missing_cells = int(model_df.isna().sum().sum())
        missing_share = missing_cells / total_cells
        if missing_share > 0.05:
            import warnings

            warnings.warn(
                f"model_dataset cell-level missingness is {missing_share:.1%} "
                f"({missing_cells}/{total_cells}), exceeding the 5% threshold. "
                "Downstream regression results may be unreliable.",
                stacklevel=2,
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
        "supplemental_sources",  # TODO (DQA-C3): 100% null in real data — not yet populated by housing acquisition sources
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
        "DFW_inclusion",
        "latest_FHFA_HPI",
        "residential_validity",
        "HUD_FY2026_SAFMR",
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


def build_housing_history_features(housing_history_panel: pd.DataFrame) -> pd.DataFrame:
    """Derive per-ZIP historical housing coverage and simple trend features."""

    columns = [
        "zip",
        "housing_history_observations_total",
        "housing_history_source_count",
        "housing_history_first_period_year",
        "housing_history_last_period_year",
        "housing_history_years_covered",
        "housing_history_latest_period_start",
        "housing_history_latest_source",
        "housing_history_latest_price_signal_value",
        "housing_history_latest_price_signal_unit",
        "fhfa_history_observations",
        "fhfa_history_latest_hpi",
        "fhfa_history_full_change_pct",
        "realtor_history_observations",
        "realtor_history_latest_listing_price",
        "realtor_history_full_change_pct",
    ]
    if housing_history_panel.empty:
        return pd.DataFrame(columns=columns)

    required = {"zip", "period_start", "period_year", "source"}
    missing = required - set(housing_history_panel.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"housing_history_panel is missing required columns: {missing_list}")

    panel = housing_history_panel.copy()
    panel["zip"] = normalize_zip_series(panel["zip"])
    panel["period_start"] = _coerce_dates(panel["period_start"])
    panel["period_year"] = _coerce_numeric(panel["period_year"])
    panel["source"] = panel["source"].astype("string")
    if "price_signal_value" in panel.columns:
        panel["price_signal_value"] = _coerce_numeric(panel["price_signal_value"])
    if "price_signal_unit" in panel.columns:
        panel["price_signal_unit"] = panel["price_signal_unit"].astype("string")
    panel = panel.dropna(subset=["zip", "period_start", "period_year", "source"]).copy()
    if panel.empty:
        return pd.DataFrame(columns=columns)

    overall = (
        panel.groupby("zip", as_index=False)
        .agg(
            housing_history_observations_total=("period_start", "size"),
            housing_history_source_count=("source", "nunique"),
            housing_history_first_period_year=("period_year", "min"),
            housing_history_last_period_year=("period_year", "max"),
        )
        .assign(
            housing_history_years_covered=lambda frame: (
                frame["housing_history_last_period_year"]
                - frame["housing_history_first_period_year"]
                + 1
            )
        )
    )

    price_signal_rows = panel.dropna(subset=["price_signal_value"]).copy()
    if price_signal_rows.empty:
        latest_price_signal = pd.DataFrame(
            columns=[
                "zip",
                "housing_history_latest_period_start",
                "housing_history_latest_source",
                "housing_history_latest_price_signal_value",
                "housing_history_latest_price_signal_unit",
            ]
        )
    else:
        latest_price_signal = (
            price_signal_rows.sort_values(
                ["zip", "period_start", "price_signal_value"],
                ascending=[True, False, False],
                na_position="last",
                kind="mergesort",
            )
            .drop_duplicates(subset=["zip"], keep="first")
            .rename(
                columns={
                    "period_start": "housing_history_latest_period_start",
                    "source": "housing_history_latest_source",
                    "price_signal_value": "housing_history_latest_price_signal_value",
                    "price_signal_unit": "housing_history_latest_price_signal_unit",
                }
            )[
                [
                    "zip",
                    "housing_history_latest_period_start",
                    "housing_history_latest_source",
                    "housing_history_latest_price_signal_value",
                    "housing_history_latest_price_signal_unit",
                ]
            ]
        )

    def _source_features(
        frame: pd.DataFrame,
        *,
        source_name: str,
        observation_col: str,
        latest_value_col: str,
        change_col: str,
    ) -> pd.DataFrame:
        source_frame = frame.loc[
            (frame["source"] == source_name) & frame["price_signal_value"].notna(),
            ["zip", "period_start", "price_signal_value"],
        ].copy()
        if source_frame.empty:
            return pd.DataFrame(columns=["zip", observation_col, latest_value_col, change_col])

        source_frame = source_frame.sort_values(
            ["zip", "period_start"],
            ascending=[True, True],
            na_position="last",
            kind="mergesort",
        )
        rows: list[dict[str, object]] = []
        for zip_code, zip_frame in source_frame.groupby("zip", sort=True):
            values = zip_frame["price_signal_value"].dropna()
            if values.empty:
                continue
            first_value = float(values.iloc[0])
            latest_value = float(values.iloc[-1])
            change_pct = np.nan
            if len(values) >= 2 and first_value > 0:
                change_pct = ((latest_value / first_value) - 1.0) * 100.0
            rows.append(
                {
                    "zip": zip_code,
                    observation_col: int(len(values)),
                    latest_value_col: latest_value,
                    change_col: change_pct,
                }
            )

        return pd.DataFrame.from_records(
            rows,
            columns=["zip", observation_col, latest_value_col, change_col],
        )

    fhfa_features = _source_features(
        price_signal_rows,
        source_name="fhfa_zip5",
        observation_col="fhfa_history_observations",
        latest_value_col="fhfa_history_latest_hpi",
        change_col="fhfa_history_full_change_pct",
    )
    realtor_features = _source_features(
        price_signal_rows,
        source_name="realtor_history",
        observation_col="realtor_history_observations",
        latest_value_col="realtor_history_latest_listing_price",
        change_col="realtor_history_full_change_pct",
    )

    features = overall.merge(latest_price_signal, on="zip", how="left")
    features = features.merge(fhfa_features, on="zip", how="left")
    features = features.merge(realtor_features, on="zip", how="left")
    return features.sort_values("zip", ignore_index=True)


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
        renter_share = pd.Series(
            np.where(owner_share.notna(), 1 - owner_share, np.nan), index=frame.index
        )
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
        frame["vacancy_proxy"] = _normalize_ratio(
            _safe_divide(vacant_units, vacant_units + occupied_units)
        )
    elif {"total_housing_units", "occupied_housing_units"} <= set(frame.columns):
        total_units = _coerce_numeric(frame["total_housing_units"])
        occupied_units = _coerce_numeric(frame["occupied_housing_units"])
        frame["vacancy_proxy"] = _normalize_ratio(
            _safe_divide(total_units - occupied_units, total_units)
        )
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
        frame["housing_tenure_mix"] = _coerce_numeric(
            frame["owner_occupied_share"]
        ) - _coerce_numeric(frame["renter_occupied_share"])

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


def prepare_census_snapshot_panel(census_snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize stacked ACS snapshot rows into a ZIP-by-year control panel."""

    required = {"zip", "snapshot_year"}
    missing = required - set(census_snapshots_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"census_snapshots_df is missing required columns: {missing_list}")

    frame = census_snapshots_df.copy()
    frame["zip"] = normalize_zip_series(frame["zip"])
    frame["snapshot_year"] = _coerce_numeric(frame["snapshot_year"])
    for column in (
        "population",
        "median_household_income",
        "occupied_housing_units",
        "owner_occupied_units",
        "renter_occupied_units",
        "median_gross_rent",
        "poverty_universe",
        "poverty_count",
        "poverty_rate",
        "owner_occupied_share",
        "renter_occupied_share",
        "housing_tenure_mix",
    ):
        if column in frame.columns:
            frame[column] = _coerce_numeric(frame[column])

    occupied_units = (
        _coerce_numeric(frame["occupied_housing_units"])
        if "occupied_housing_units" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    owner_units = (
        _coerce_numeric(frame["owner_occupied_units"])
        if "owner_occupied_units" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    renter_units = (
        _coerce_numeric(frame["renter_occupied_units"])
        if "renter_occupied_units" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    poverty_universe = (
        _coerce_numeric(frame["poverty_universe"])
        if "poverty_universe" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    poverty_count = (
        _coerce_numeric(frame["poverty_count"])
        if "poverty_count" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )

    owner_share_input = (
        frame["owner_occupied_share"]
        if "owner_occupied_share" in frame.columns
        else _safe_divide(owner_units, occupied_units)
    )
    renter_share_input = (
        frame["renter_occupied_share"]
        if "renter_occupied_share" in frame.columns
        else _safe_divide(renter_units, occupied_units)
    )
    poverty_rate_input = (
        frame["poverty_rate"]
        if "poverty_rate" in frame.columns
        else _safe_divide(poverty_count, poverty_universe)
    )

    frame["owner_occupied_share"] = _normalize_ratio(owner_share_input)
    frame["renter_occupied_share"] = _normalize_ratio(renter_share_input)
    frame["poverty_rate"] = _normalize_ratio(poverty_rate_input)
    frame["housing_tenure_mix"] = _coerce_numeric(
        frame.get("housing_tenure_mix", frame["owner_occupied_share"] - frame["renter_occupied_share"])
    )

    keep = ["zip", "snapshot_year", *ACS_SNAPSHOT_METRICS]
    for column in ACS_SNAPSHOT_METRICS:
        if column not in frame.columns:
            frame[column] = np.nan

    frame = frame[keep].dropna(subset=["zip", "snapshot_year"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=keep)

    frame["_non_null_score"] = frame.notna().sum(axis=1)
    frame = (
        frame.sort_values(
            ["zip", "snapshot_year", "_non_null_score"],
            ascending=[True, True, False],
            na_position="last",
            kind="mergesort",
        )
        .drop_duplicates(subset=["zip", "snapshot_year"], keep="first")
        .drop(columns=["_non_null_score"], errors="ignore")
        .reset_index(drop=True)
    )
    return frame


def build_acs_snapshot_features(census_snapshot_panel: pd.DataFrame) -> pd.DataFrame:
    """Derive compact ZIP-level features from stacked ACS snapshot controls."""

    columns = [
        "zip",
        "acs_snapshot_observation_count",
        "acs_snapshot_first_year",
        "acs_snapshot_last_year",
        "acs_snapshot_years_covered",
    ]
    for metric in ACS_SNAPSHOT_METRICS:
        columns.extend(
            [
                f"acs_snapshot_latest_{metric}",
                f"acs_snapshot_first_{metric}",
                f"acs_snapshot_{metric}_change",
                f"acs_snapshot_{metric}_change_pct",
                f"acs_snapshot_{metric}_trend_per_year",
            ]
        )

    if census_snapshot_panel.empty:
        return pd.DataFrame(columns=columns)

    required = {"zip", "snapshot_year", *ACS_SNAPSHOT_METRICS}
    missing = required - set(census_snapshot_panel.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"census_snapshot_panel is missing required columns: {missing_list}")

    panel = census_snapshot_panel.copy()
    panel["zip"] = normalize_zip_series(panel["zip"])
    panel["snapshot_year"] = _coerce_numeric(panel["snapshot_year"])
    for metric in ACS_SNAPSHOT_METRICS:
        panel[metric] = _coerce_numeric(panel[metric])
    panel = panel.dropna(subset=["zip", "snapshot_year"]).copy()
    if panel.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for zip_code, zip_frame in panel.groupby("zip", sort=True):
        ordered = zip_frame.sort_values("snapshot_year", kind="mergesort", ignore_index=True).copy()
        first_year = int(ordered["snapshot_year"].min())
        last_year = int(ordered["snapshot_year"].max())
        row: dict[str, object] = {
            "zip": zip_code,
            "acs_snapshot_observation_count": int(len(ordered)),
            "acs_snapshot_first_year": first_year,
            "acs_snapshot_last_year": last_year,
            "acs_snapshot_years_covered": int(last_year - first_year + 1),
        }

        for metric in ACS_SNAPSHOT_METRICS:
            metric_frame = ordered.loc[
                ordered[metric].notna(),
                ["snapshot_year", metric],
            ].copy()
            latest_col = f"acs_snapshot_latest_{metric}"
            first_col = f"acs_snapshot_first_{metric}"
            change_col = f"acs_snapshot_{metric}_change"
            change_pct_col = f"acs_snapshot_{metric}_change_pct"
            trend_col = f"acs_snapshot_{metric}_trend_per_year"

            row[latest_col] = np.nan
            row[first_col] = np.nan
            row[change_col] = np.nan
            row[change_pct_col] = np.nan
            row[trend_col] = np.nan
            if metric_frame.empty:
                continue

            first_value = float(metric_frame.iloc[0][metric])
            latest_value = float(metric_frame.iloc[-1][metric])
            row[latest_col] = latest_value
            row[first_col] = first_value
            if len(metric_frame) >= 2:
                row[change_col] = latest_value - first_value
                if first_value > 0:
                    row[change_pct_col] = ((latest_value / first_value) - 1.0) * 100.0
                row[trend_col] = _trend_slope(
                    metric_frame[metric],
                    x_values=metric_frame["snapshot_year"],
                )
            elif metric == "housing_tenure_mix" and len(metric_frame) == 1:
                # DQA-M2: single-snapshot ZIPs lack a prior year; impute 0-change assumption
                row[change_col] = 0.0
                row[change_pct_col] = 0.0

        rows.append(row)

    return pd.DataFrame.from_records(rows, columns=columns).sort_values("zip", ignore_index=True)


def _load_optional_zip_sidecar(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "zip" not in frame.columns:
        return pd.DataFrame(columns=["zip"])

    sidecar = frame.copy()
    sidecar["zip"] = normalize_zip_series(sidecar["zip"])
    sidecar = sidecar.dropna(subset=["zip"]).copy()
    if sidecar.empty:
        return sidecar

    sidecar["_non_null_score"] = sidecar.notna().sum(axis=1)
    return _dedupe_zip_rows(
        sidecar,
        priority_desc=("_non_null_score",),
    ).drop(columns=["_non_null_score"], errors="ignore")


def _merge_optional_sidecars(
    model_df: pd.DataFrame,
    *,
    raw_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    joined = model_df.copy()
    sidecar_frames: dict[str, pd.DataFrame] = {}
    used_columns = set(joined.columns)

    for category, filename in OPTIONAL_ZIP_SIDECAR_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            continue

        original_frame = _load_optional_zip_sidecar(path)
        sidecar_frames[category] = original_frame.copy()
        if original_frame.empty or set(original_frame.columns) == {"zip"}:
            continue

        rename_map: dict[str, str] = {}
        for column in original_frame.columns:
            if column == "zip":
                continue
            candidate = column
            if candidate in used_columns:
                candidate = f"{category}_{column}"
            suffix = 2
            while candidate in used_columns:
                candidate = f"{category}_{column}_{suffix}"
                suffix += 1
            rename_map[column] = candidate
            used_columns.add(candidate)

        sidecar = original_frame.rename(columns=rename_map)
        joined = joined.merge(sidecar, on="zip", how="left")
        print(
            f"[build] Joined optional sidecar: {path.name} "
            f"({len(sidecar)} rows, cols={list(rename_map.values())})",
            flush=True,
        )

    return joined, sidecar_frames


def _build_single_period_completeness_rows(
    zip_codes: list[str],
    *,
    category: str,
    available_zips: set[str],
    latest_period: object = None,
) -> list[dict[str, object]]:
    latest_period_text = _format_period_value(latest_period)
    rows: list[dict[str, object]] = []
    for zip_code in zip_codes:
        observed = int(zip_code in available_zips)
        rows.append(
            {
                "zip": zip_code,
                "category": category,
                "expected_periods": 1,
                "observed_periods": observed,
                "completeness_ratio": float(observed),
                "first_observed_period": latest_period_text if observed else None,
                "last_observed_period": latest_period_text if observed else None,
                "latest_available_period": latest_period_text,
            }
        )
    return rows


def _build_panel_completeness_rows(
    zip_codes: list[str],
    *,
    category: str,
    frame: pd.DataFrame,
    period_col: str,
) -> list[dict[str, object]]:
    if frame.empty or period_col not in frame.columns or "zip" not in frame.columns:
        return [
            dict(
                zip=zip_code,
                category=category,
                expected_periods=0,
                observed_periods=0,
                completeness_ratio=np.nan,
                first_observed_period=None,
                last_observed_period=None,
                latest_available_period=None,
            )
            for zip_code in zip_codes
        ]

    working = frame[["zip", period_col]].copy()
    working["zip"] = normalize_zip_series(working["zip"])
    if period_col.endswith("year"):
        working[period_col] = _coerce_numeric(working[period_col])
    else:
        working[period_col] = pd.to_datetime(working[period_col], errors="coerce")
    working = working.dropna(subset=["zip", period_col]).copy()
    if working.empty:
        return [
            dict(
                zip=zip_code,
                category=category,
                expected_periods=0,
                observed_periods=0,
                completeness_ratio=np.nan,
                first_observed_period=None,
                last_observed_period=None,
                latest_available_period=None,
            )
            for zip_code in zip_codes
        ]

    expected_periods = int(working[period_col].nunique())
    latest_available_period = _format_period_value(working[period_col].max())
    period_summary = working.groupby("zip")[period_col].agg(["nunique", "min", "max"])

    rows: list[dict[str, object]] = []
    for zip_code in zip_codes:
        if zip_code in period_summary.index:
            observed_periods = int(period_summary.loc[zip_code, "nunique"])
            first_period = _format_period_value(period_summary.loc[zip_code, "min"])
            last_period = _format_period_value(period_summary.loc[zip_code, "max"])
        else:
            observed_periods = 0
            first_period = None
            last_period = None
        ratio = (
            float(observed_periods / expected_periods)
            if expected_periods > 0
            else np.nan
        )
        rows.append(
            {
                "zip": zip_code,
                "category": category,
                "expected_periods": expected_periods,
                "observed_periods": observed_periods,
                "completeness_ratio": ratio,
                "first_observed_period": first_period,
                "last_observed_period": last_period,
                "latest_available_period": latest_available_period,
            }
        )
    return rows


def build_source_completeness_scores(
    *,
    target_zips: Iterable[object],
    crime_zip: pd.DataFrame,
    crime_history_panel: pd.DataFrame,
    housing_zip: pd.DataFrame,
    housing_history_panel: pd.DataFrame,
    controls: pd.DataFrame,
    census_snapshot_panel: pd.DataFrame,
    sidecar_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Summarize per-ZIP source completeness across core and optional datasets."""

    zip_codes = sorted(
        {
            normalized
            for normalized in (normalize_zip(value) for value in target_zips)
            if normalized is not None
        }
    )
    columns = [
        "zip",
        "category",
        "expected_periods",
        "observed_periods",
        "completeness_ratio",
        "first_observed_period",
        "last_observed_period",
        "latest_available_period",
    ]
    if not zip_codes:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    rows.extend(
        _build_single_period_completeness_rows(
            zip_codes,
            category="crime_current",
            available_zips=_zip_set(crime_zip),
            latest_period=crime_zip["period_start"].max() if "period_start" in crime_zip.columns else None,
        )
    )
    rows.extend(
        _build_panel_completeness_rows(
            zip_codes,
            category="crime_history",
            frame=crime_history_panel,
            period_col="period_start",
        )
    )
    rows.extend(
        _build_single_period_completeness_rows(
            zip_codes,
            category="housing_current",
            available_zips=_zip_set(housing_zip),
            latest_period=housing_zip["as_of_date"].max() if "as_of_date" in housing_zip.columns else None,
        )
    )
    rows.extend(
        _build_panel_completeness_rows(
            zip_codes,
            category="housing_history",
            frame=housing_history_panel,
            period_col="period_start",
        )
    )
    rows.extend(
        _build_single_period_completeness_rows(
            zip_codes,
            category="acs_current",
            available_zips=_zip_set(controls),
        )
    )
    rows.extend(
        _build_panel_completeness_rows(
            zip_codes,
            category="acs_snapshots",
            frame=census_snapshot_panel,
            period_col="snapshot_year",
        )
    )

    for category, frame in (sidecar_frames or {}).items():
        rows.extend(
            _build_single_period_completeness_rows(
                zip_codes,
                category=category,
                available_zips=_zip_set(frame),
            )
        )

    report = pd.DataFrame.from_records(rows, columns=columns)
    if report.empty:
        return report
    return report.sort_values(["zip", "category"], kind="mergesort", ignore_index=True)


def _zscore_series(series: pd.Series) -> pd.Series:
    numeric = _coerce_numeric(series)
    std = float(numeric.std(ddof=0))
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=numeric.index, dtype="float64")
    return (numeric - float(numeric.mean())) / std


def build_interaction_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """Derive additive interaction and aggregate terms for downstream modeling."""

    columns = [
        "zip",
        "crime_income_interaction",
        "crime_rent_burden_interaction",
        "crime_poverty_interaction",
        "crime_unemployment_interaction",
        "crime_density_interaction",  # TODO (DQA-C3): 100% null in real data — pop_density requires population/area data not yet acquired
        "vacancy_poverty_interaction",
        "momentum_home_value_pressure_interaction",
        "market_momentum_interaction",
        "rent_income_stress_interaction",
        "completeness_weighted_crime_risk",
        "aggregate_distress_index",
        "aggregate_market_pressure_index",
        "crime_history_lag2_trend_ratio",
    ]
    if model_df.empty:
        return pd.DataFrame(columns=columns)
    if "zip" not in model_df.columns:
        raise KeyError("model_df is missing required column: zip")

    frame = model_df[["zip"]].copy()
    frame["zip"] = normalize_zip_series(frame["zip"])
    crime_rate = _optional_numeric(model_df, "total_rate_per_1000")
    median_income = _optional_numeric(model_df, "median_household_income")
    rent_burden = _optional_numeric(model_df, "rent_burden")
    vacancy_proxy = _optional_numeric(model_df, "vacancy_proxy")
    poverty_rate = _optional_numeric(model_df, "poverty_rate")
    unemployment_rate = _optional_numeric(model_df, "unemployment_rate")
    pop_density = _optional_numeric(model_df, "pop_density")
    momentum = (
        _coerce_numeric(model_df["crime_history_rate_momentum_2q"])
        if "crime_history_rate_momentum_2q" in model_df.columns
        else (
            _coerce_numeric(model_df["crime_history_latest_vs_lag1_rate_change"])
            if "crime_history_latest_vs_lag1_rate_change" in model_df.columns
            else pd.Series(np.nan, index=model_df.index, dtype="float64")
        )
    )
    log_home_value = (
        _coerce_numeric(model_df["log_home_value"])
        if "log_home_value" in model_df.columns
        else (
            np.log(_coerce_numeric(model_df["home_value"]).where(_coerce_numeric(model_df["home_value"]) > 0))
            if "home_value" in model_df.columns
            else pd.Series(np.nan, index=model_df.index, dtype="float64")
        )
    )
    completeness = _optional_numeric(model_df, "source_completeness_overall_score")
    annual_change_pct = _optional_numeric(model_df, "annual_change_pct")
    median_rent = _optional_numeric(model_df, "median_rent")
    home_value = _optional_numeric(model_df, "home_value")

    frame["crime_income_interaction"] = crime_rate * np.log1p(median_income.clip(lower=0))
    frame["crime_rent_burden_interaction"] = crime_rate * rent_burden
    frame["crime_poverty_interaction"] = crime_rate * poverty_rate
    frame["crime_unemployment_interaction"] = crime_rate * unemployment_rate
    frame["crime_density_interaction"] = crime_rate * np.log1p(pop_density.clip(lower=0))  # TODO (DQA-C3): null when pop_density is unavailable
    frame["vacancy_poverty_interaction"] = vacancy_proxy * poverty_rate
    frame["momentum_home_value_pressure_interaction"] = momentum * log_home_value
    frame["market_momentum_interaction"] = annual_change_pct * momentum
    frame["rent_income_stress_interaction"] = rent_burden * _safe_divide(median_rent, median_income)
    frame["completeness_weighted_crime_risk"] = crime_rate * completeness

    lag1_rate_hist = _optional_numeric(model_df, "crime_history_lag1_total_rate_per_1000")
    lag2_rate_hist = _optional_numeric(model_df, "crime_history_lag2_total_rate_per_1000")
    frame["crime_history_lag2_trend_ratio"] = _safe_divide(lag1_rate_hist, lag2_rate_hist)

    distress_components = pd.concat(
        [
            _zscore_series(crime_rate),
            _zscore_series(rent_burden),
            _zscore_series(vacancy_proxy),
            _zscore_series(poverty_rate),
        ],
        axis=1,
    )
    frame["aggregate_distress_index"] = distress_components.mean(axis=1)

    market_pressure_components = pd.concat(
        [
            _zscore_series(crime_rate),
            _zscore_series(annual_change_pct),
            _zscore_series(_safe_divide(median_rent, home_value)),
            _zscore_series(
                _optional_numeric(model_df, "acs_snapshot_median_household_income_trend_per_year")
            ),
        ],
        axis=1,
    )
    frame["aggregate_market_pressure_index"] = market_pressure_components.mean(axis=1)

    frame = frame.dropna(subset=["zip"]).copy()
    return frame[columns].sort_values("zip", ignore_index=True)


def build_all(settings: "Settings") -> dict[str, str]:
    """Build processed crime, housing, ACS, and model datasets."""

    crime_path = settings.raw_dir / "crime_records.csv"
    crime_history_path = settings.raw_dir / "crime_history_records.csv"
    housing_path = settings.raw_dir / "housing_market.csv"
    housing_history_path = settings.raw_dir / "housing_market_history.csv"
    census_path = settings.raw_dir / "acs_zcta.csv"
    census_snapshot_path = settings.raw_dir / "acs_zcta_snapshots.csv"

    missing_paths = [path for path in (crime_path, housing_path, census_path) if not path.exists()]
    if missing_paths:
        missing_list = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing raw inputs: {missing_list}")

    crime_raw = pd.read_csv(crime_path)
    crime_history_raw = pd.read_csv(crime_history_path) if crime_history_path.exists() else crime_raw.copy()
    housing_raw = pd.read_csv(housing_path)
    housing_history_raw = (
        pd.read_csv(housing_history_path) if housing_history_path.exists() else pd.DataFrame()
    )
    census_raw = pd.read_csv(census_path)
    census_snapshot_raw = (
        pd.read_csv(census_snapshot_path) if census_snapshot_path.exists() else pd.DataFrame()
    )

    candidate_zip_pool = set().union(
        _zip_set(crime_raw),
        _zip_set(crime_history_raw),
        _zip_set(housing_raw),
        _zip_set(housing_history_raw),
        _zip_set(census_raw),
        _zip_set(census_snapshot_raw),
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
    crime_history_raw = _filter_to_study_zip_universe(
        crime_history_raw,
        settings,
        allowed_zips=dfw_zip_set,
    )
    housing_raw = _filter_to_study_zip_universe(housing_raw, settings, allowed_zips=dfw_zip_set)
    housing_history_raw = _filter_to_study_zip_universe(
        housing_history_raw,
        settings,
        allowed_zips=dfw_zip_set,
    )
    census_raw = _filter_to_study_zip_universe(census_raw, settings, allowed_zips=dfw_zip_set)
    census_snapshot_raw = _filter_to_study_zip_universe(
        census_snapshot_raw,
        settings,
        allowed_zips=dfw_zip_set,
    )

    controls = prepare_census_controls(census_raw)
    crime_zip = aggregate_crime_data(
        crime_raw,
        controls[["zip", "population"]],
        zip_col="zip",
        date_col="reported_at",
        category_col="offense_family",
    )
    crime_zip = crime_zip[
        crime_zip["total_incidents"] >= settings.min_total_incidents_per_zip
    ].copy()
    crime_history_panel = prepare_crime_history_panel(
        crime_history_raw,
        controls[["zip", "population"]],
        zip_col="zip",
        date_col="reported_at",
        category_col="offense_family",
    )
    crime_history_features = (
        build_crime_history_features(crime_history_panel)
        if not crime_history_panel.empty
        else pd.DataFrame()
    )
    if crime_zip.empty:
        raise ValueError(
            "Crime aggregation produced no usable ZIP-level rows. "
            "Check crime acquisition output and DCA_MIN_TOTAL_INCIDENTS_PER_ZIP."
        )

    housing_zip = prepare_housing_features(housing_raw)
    if housing_zip.empty:
        raise ValueError(
            "Housing preparation produced no usable rows. "
            "Check housing acquisition output and Firecrawl configuration."
        )
    if controls.empty:
        raise ValueError(
            "Census controls produced no usable rows. "
            "Check census acquisition output and DCA_CENSUS_YEAR."
        )

    housing_history_panel = (
        prepare_housing_history_panel(housing_history_raw)
        if not housing_history_raw.empty
        else pd.DataFrame()
    )
    housing_history_features = (
        build_housing_history_features(housing_history_panel)
        if not housing_history_panel.empty
        else pd.DataFrame()
    )
    census_snapshot_panel = (
        prepare_census_snapshot_panel(census_snapshot_raw)
        if not census_snapshot_raw.empty
        else pd.DataFrame(columns=["zip", "snapshot_year", *ACS_SNAPSHOT_METRICS])
    )
    acs_snapshot_features = build_acs_snapshot_features(census_snapshot_panel)
    model_df = build_model_dataset(crime_zip, housing_zip, controls)
    if not crime_history_features.empty:
        model_df = model_df.merge(crime_history_features, on="zip", how="left")
    if not housing_history_features.empty:
        model_df = model_df.merge(housing_history_features, on="zip", how="left")
    if not acs_snapshot_features.empty:
        model_df = model_df.merge(acs_snapshot_features, on="zip", how="left")
    model_df, sidecar_frames = _merge_optional_sidecars(model_df, raw_dir=settings.raw_dir)

    target_zip_universe = _build_target_zip_universe(
        crime_raw=crime_raw,
        housing_raw=housing_raw,
        census_raw=census_raw,
        crime_zip=crime_zip,
        housing_zip=housing_zip,
        controls=controls,
        model_df=model_df,
    )
    source_completeness_scores = build_source_completeness_scores(
        target_zips=target_zip_universe["zip"] if "zip" in target_zip_universe.columns else [],
        crime_zip=crime_zip,
        crime_history_panel=crime_history_panel,
        housing_zip=housing_zip,
        housing_history_panel=housing_history_panel,
        controls=controls,
        census_snapshot_panel=census_snapshot_panel,
        sidecar_frames=sidecar_frames,
    )
    source_completeness_summary = (
        source_completeness_scores.groupby("zip", as_index=False)
        .agg(
            source_completeness_overall_score=("completeness_ratio", "mean"),
            source_completeness_category_count=("category", "nunique"),
        )
        .sort_values("zip", ignore_index=True)
        if not source_completeness_scores.empty
        else pd.DataFrame(
            columns=[
                "zip",
                "source_completeness_overall_score",
                "source_completeness_category_count",
            ]
        )
    )
    if not source_completeness_summary.empty:
        model_df = model_df.merge(source_completeness_summary, on="zip", how="left")
    interaction_features = build_interaction_features(model_df)
    if not interaction_features.empty:
        model_df = model_df.merge(interaction_features, on="zip", how="left")

    qa_core_datasets: dict[str, pd.DataFrame] = {
        "crime_zip": crime_zip,
        "housing_zip": housing_zip,
        "acs_controls": controls,
        "model_dataset": model_df,
    }
    if not acs_snapshot_features.empty:
        qa_core_datasets["acs_snapshot_features"] = acs_snapshot_features
    if not interaction_features.empty:
        qa_core_datasets["interaction_features"] = interaction_features

    duplicate_report = _build_duplicate_zip_report(
        {
            "crime_raw": crime_raw,
            "housing_raw": housing_raw,
            "census_raw": census_raw,
            **qa_core_datasets,
        }
    )
    missingness_datasets = {**qa_core_datasets, "target_zip_universe": target_zip_universe}
    if not source_completeness_scores.empty:
        missingness_datasets["source_completeness_scores"] = source_completeness_scores
    missingness_report = _build_missingness_report(missingness_datasets)
    impossible_report = _build_impossible_values_report(qa_core_datasets)
    outlier_report = _build_outlier_markers(qa_core_datasets)
    qa_summary = _build_qa_summary(
        datasets={
            "crime_zip": crime_zip,
            **({"crime_history_panel": crime_history_panel} if not crime_history_panel.empty else {}),
            **(
                {"crime_history_features": crime_history_features}
                if not crime_history_features.empty
                else {}
            ),
            "housing_zip": housing_zip,
            "acs_controls": controls,
            **(
                {"acs_snapshot_features": acs_snapshot_features}
                if not acs_snapshot_features.empty
                else {}
            ),
            **(
                {"source_completeness_scores": source_completeness_scores}
                if not source_completeness_scores.empty
                else {}
            ),
            **(
                {"interaction_features": interaction_features}
                if not interaction_features.empty
                else {}
            ),
            "model_dataset": model_df,
            **(
                {"housing_history_panel": housing_history_panel}
                if not housing_history_panel.empty
                else {}
            ),
            **(
                {"housing_history_features": housing_history_features}
                if not housing_history_features.empty
                else {}
            ),
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
        "acs_snapshot_features": settings.processed_dir / "acs_snapshot_features.csv",
        "model_dataset": settings.processed_dir / "model_dataset.csv",
        "source_completeness_scores": settings.processed_dir / "source_completeness_scores.csv",
        "interaction_features": settings.processed_dir / "interaction_features.csv",
        "target_zip_universe": settings.processed_dir / "target_zip_universe.csv",
        "qa_duplicate_zip_report": settings.processed_dir / "qa_duplicate_zip_report.csv",
        "qa_missingness_report": settings.processed_dir / "qa_missingness_report.csv",
        "qa_impossible_values_report": settings.processed_dir / "qa_impossible_values_report.csv",
        "qa_outlier_markers": settings.processed_dir / "qa_outlier_markers.csv",
        "qa_summary": settings.processed_dir / "qa_summary.json",
    }
    if not crime_history_panel.empty:
        outputs["crime_history_panel"] = settings.processed_dir / "crime_history_panel.csv"
    if not crime_history_features.empty:
        outputs["crime_history_features"] = settings.processed_dir / "crime_history_features.csv"
    if not housing_history_panel.empty:
        outputs["housing_history_panel"] = settings.processed_dir / "housing_history_panel.csv"
    if not housing_history_features.empty:
        outputs["housing_history_features"] = (
            settings.processed_dir / "housing_history_features.csv"
        )
    crime_zip.to_csv(outputs["crime_zip"], index=False)
    if not crime_history_panel.empty:
        crime_history_panel.to_csv(outputs["crime_history_panel"], index=False)
    if not crime_history_features.empty:
        crime_history_features.to_csv(outputs["crime_history_features"], index=False)
    housing_zip.to_csv(outputs["housing_zip"], index=False)
    controls.to_csv(outputs["acs_controls"], index=False)
    acs_snapshot_features.to_csv(outputs["acs_snapshot_features"], index=False)
    model_df.to_csv(outputs["model_dataset"], index=False)
    source_completeness_scores.to_csv(outputs["source_completeness_scores"], index=False)
    interaction_features.to_csv(outputs["interaction_features"], index=False)
    if not housing_history_panel.empty:
        housing_history_panel.to_csv(outputs["housing_history_panel"], index=False)
    if not housing_history_features.empty:
        housing_history_features.to_csv(outputs["housing_history_features"], index=False)
    target_zip_universe.to_csv(outputs["target_zip_universe"], index=False)
    duplicate_report.to_csv(outputs["qa_duplicate_zip_report"], index=False)
    missingness_report.to_csv(outputs["qa_missingness_report"], index=False)
    impossible_report.to_csv(outputs["qa_impossible_values_report"], index=False)
    outlier_report.to_csv(outputs["qa_outlier_markers"], index=False)
    _write_json(outputs["qa_summary"], qa_summary)

    return {label: str(path) for label, path in outputs.items()}
