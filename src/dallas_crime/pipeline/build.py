"""Transformation helpers for ZIP-level crime, housing, and ACS data."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dallas_crime.config import Settings

DEFAULT_VIOLENT_LABELS = frozenset({"violent"})
DEFAULT_PROPERTY_LABELS = frozenset({"property"})

ZIP_PATTERN = re.compile(r"(\d{5})")
MISSING_STRINGS = {"", "na", "nan", "none", "<na>", "null"}


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

    crimes = crime_df[[zip_col, date_col, category_col]].copy()
    crimes["zip"] = normalize_zip_series(crimes[zip_col])
    crimes["incident_date"] = _coerce_dates(crimes[date_col])
    crimes["crime_category"] = _lower_text(crimes[category_col]).fillna("other")
    crimes = crimes.dropna(subset=["zip", "incident_date"])

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
    population = (
        population.dropna(subset=["zip"])
        .sort_values("population")
        .drop_duplicates(subset=["zip"], keep="last")
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
    crime = crime.dropna(subset=["zip"]).drop_duplicates(subset=["zip"], keep="last")

    housing = housing_df.copy()
    housing["zip"] = normalize_zip_series(housing[housing_zip_col])
    housing["home_value"] = _coerce_numeric(housing[home_value_col])
    if "as_of_date" in housing.columns:
        housing["as_of_date"] = _coerce_dates(housing["as_of_date"])
        housing = housing.sort_values("as_of_date")
    housing = housing.dropna(subset=["zip"]).drop_duplicates(subset=["zip"], keep="last")

    controls = controls_df.copy()
    controls["zip"] = normalize_zip_series(controls[controls_zip_col])
    controls = controls.dropna(subset=["zip"]).drop_duplicates(subset=["zip"], keep="last")

    model_df = crime.merge(housing, on="zip", how="inner", suffixes=("", "_housing"))
    model_df = model_df.merge(controls, on="zip", how="left", suffixes=("", "_acs"))

    if "source" in model_df.columns:
        model_df["source"] = model_df["source"].astype("string")

    model_df["home_value"] = _coerce_numeric(model_df["home_value"])
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
    for optional in ("as_of_date", "annual_change_pct", "median_rent", "source", "source_url"):
        if optional in frame.columns:
            keep.append(optional)

    frame = frame[keep].copy()
    frame["zip"] = normalize_zip_series(frame["zip"])
    frame["home_value"] = _coerce_numeric(frame["home_value"])
    if "as_of_date" in frame.columns:
        frame["as_of_date"] = _coerce_dates(frame["as_of_date"])
        frame = frame.sort_values("as_of_date")
    return frame.dropna(subset=["zip", "home_value"]).drop_duplicates(subset=["zip"], keep="last")


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

    frame = census_df[required].copy()
    frame["zip"] = normalize_zip_series(frame["zip"])
    for column in required:
        if column == "zip":
            continue
        frame[column] = _coerce_numeric(frame[column])
    return frame.dropna(subset=["zip"]).drop_duplicates(subset=["zip"], keep="last")


def build_all(settings: "Settings") -> dict[str, str]:
    """Build processed crime, housing, ACS, and model datasets."""

    crime_path = settings.raw_dir / "crime_records.csv"
    housing_path = settings.raw_dir / "housing_market.csv"
    census_path = settings.raw_dir / "acs_zcta.csv"

    missing_paths = [path for path in (crime_path, housing_path, census_path) if not path.exists()]
    if missing_paths:
        missing_list = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing raw inputs: {missing_list}")

    crime_raw = pd.read_csv(crime_path)
    housing_raw = pd.read_csv(housing_path)
    census_raw = pd.read_csv(census_path)

    controls = prepare_census_controls(census_raw)
    crime_zip = aggregate_crime_data(
        crime_raw,
        controls[["zip", "population"]],
        zip_col="zip",
        date_col="reported_at",
        category_col="offense_family",
    )
    housing_zip = prepare_housing_features(housing_raw)
    model_df = build_model_dataset(crime_zip, housing_zip, controls)

    outputs = {
        "crime_zip": settings.processed_dir / "crime_zip.csv",
        "housing_zip": settings.processed_dir / "housing_zip.csv",
        "acs_controls": settings.processed_dir / "acs_controls.csv",
        "model_dataset": settings.processed_dir / "model_dataset.csv",
    }
    crime_zip.to_csv(outputs["crime_zip"], index=False)
    housing_zip.to_csv(outputs["housing_zip"], index=False)
    controls.to_csv(outputs["acs_controls"], index=False)
    model_df.to_csv(outputs["model_dataset"], index=False)

    return {label: str(path) for label, path in outputs.items()}
