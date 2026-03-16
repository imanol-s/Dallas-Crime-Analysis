"""Optional sidecar acquisition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dallas_crime.acquire.utils import load_dfw_zip_set, utc_timestamp, write_json_artifact

if TYPE_CHECKING:
    from dallas_crime.config import Settings


SIDECAR_FILE_BY_CATEGORY = {
    "economic": "dfw_zip_economic_sidecar.csv",
    "real_estate": "dfw_zip_real_estate_sidecar.csv",
    "law_enforcement": "dfw_zip_law_enforcement_sidecar.csv",
    "social_services": "dfw_zip_social_services_sidecar.csv",
    "infrastructure": "dfw_zip_infrastructure_sidecar.csv",
}


@dataclass(frozen=True)
class SidecarArtifacts:
    """Container for generated sidecar paths."""

    economic: str
    real_estate: str
    law_enforcement: str
    social_services: str
    infrastructure: str
    metadata: str

    def as_dict(self) -> dict[str, str]:
        return {
            "economic": self.economic,
            "real_estate": self.real_estate,
            "law_enforcement": self.law_enforcement,
            "social_services": self.social_services,
            "infrastructure": self.infrastructure,
            "metadata": self.metadata,
        }


def _normalize_zip_series(values: pd.Series) -> pd.Series:
    return values.astype("string").str.extract(r"(\d{5})", expand=False)


def _coerce_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_num = _coerce_numeric(numerator)
    denominator_num = _coerce_numeric(denominator)
    result = np.where(denominator_num > 0, numerator_num / denominator_num, np.nan)
    return pd.Series(result, index=numerator_num.index, dtype="float64")


def _scale_unit_interval(values: pd.Series) -> pd.Series:
    numeric = _coerce_numeric(values)
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(np.nan, index=numeric.index, dtype="float64")
    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(low, high):
        return pd.Series(0.5, index=numeric.index, dtype="float64")
    scaled = (numeric - low) / (high - low)
    return scaled.clip(lower=0, upper=1)


def _load_raw_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _prepare_zip_frame(frame: pd.DataFrame, *, dedupe: bool = True) -> pd.DataFrame:
    if frame.empty or "zip" not in frame.columns:
        return pd.DataFrame(columns=["zip"])
    prepared = frame.copy()
    prepared["zip"] = _normalize_zip_series(prepared["zip"])
    prepared = prepared.dropna(subset=["zip"]).copy()
    if prepared.empty:
        return pd.DataFrame(columns=["zip"])
    if dedupe:
        prepared = prepared.drop_duplicates(subset=["zip"], keep="first")
    return prepared.sort_values("zip", ignore_index=True)


def _build_zip_universe(
    *,
    settings: "Settings",
    crime_current: pd.DataFrame,
    census_current: pd.DataFrame,
    housing_current: pd.DataFrame,
) -> list[str]:
    zip_pool: set[str] = set()
    for frame in (crime_current, census_current, housing_current):
        if "zip" not in frame.columns:
            continue
        normalized = _normalize_zip_series(frame["zip"]).dropna()
        zip_pool.update(normalized.tolist())

    zip_pool = {zip_code for zip_code in zip_pool if settings.allows_study_zip(zip_code)}
    if not zip_pool:
        return []

    crosswalk_path = settings.raw_dir / "zcta_county_crosswalk_2020.csv"
    try:
        dfw_zip_set = load_dfw_zip_set(
            zip_pool,
            cache_path=crosswalk_path,
            timeout_seconds=settings.acquire_timeout_seconds,
            max_attempts=settings.acquire_max_attempts,
            backoff_seconds=settings.acquire_backoff_seconds,
        )
    except Exception as exc:
        print(f"[sidecars] DFW crosswalk resolution failed, using prefix filter: {exc}", flush=True)
        dfw_zip_set = set()

    if dfw_zip_set:
        zip_pool = zip_pool.intersection(dfw_zip_set)
    return sorted(zip_pool)


def _prepare_census_features(frame: pd.DataFrame) -> pd.DataFrame:
    census = _prepare_zip_frame(frame)
    if census.empty:
        return census

    for column in (
        "median_household_income",
        "poverty_rate",
        "owner_occupied_share",
        "population",
        "unemployment_rate",
        "labor_force",
        "unemployed_count",
        "vacancy_proxy",
        "total_housing_units",
        "vacant_housing_units",
        "educational_attainment",
        "bachelors_or_higher_count",
        "education_population_25_plus",
        "public_assistance_share",
        "households_total",
        "households_public_assistance",
        "transit_commute_share",
        "total_commuters",
        "public_transit_commuters",
    ):
        if column in census.columns:
            census[column] = _coerce_numeric(census[column])

    if "unemployment_rate" not in census.columns:
        if {"unemployed_count", "labor_force"} <= set(census.columns):
            census["unemployment_rate"] = _safe_divide(
                census["unemployed_count"],
                census["labor_force"],
            )
        else:
            census["unemployment_rate"] = np.nan

    if "poverty_rate" not in census.columns:
        if {"poverty_count", "poverty_universe"} <= set(census.columns):
            census["poverty_rate"] = _safe_divide(census["poverty_count"], census["poverty_universe"])
        else:
            census["poverty_rate"] = np.nan

    if "owner_occupied_share" not in census.columns:
        if {"owner_occupied_units", "occupied_housing_units"} <= set(census.columns):
            census["owner_occupied_share"] = _safe_divide(
                census["owner_occupied_units"],
                census["occupied_housing_units"],
            )
        else:
            census["owner_occupied_share"] = np.nan

    if "vacancy_proxy" not in census.columns:
        if {"vacant_housing_units", "total_housing_units"} <= set(census.columns):
            census["vacancy_proxy"] = _safe_divide(
                census["vacant_housing_units"],
                census["total_housing_units"],
            )
        else:
            census["vacancy_proxy"] = np.nan

    if "educational_attainment" not in census.columns:
        if {"bachelors_or_higher_count", "education_population_25_plus"} <= set(census.columns):
            census["educational_attainment"] = _safe_divide(
                census["bachelors_or_higher_count"],
                census["education_population_25_plus"],
            )
        else:
            census["educational_attainment"] = np.nan

    if "public_assistance_share" not in census.columns:
        if {"households_public_assistance", "households_total"} <= set(census.columns):
            census["public_assistance_share"] = _safe_divide(
                census["households_public_assistance"],
                census["households_total"],
            )
        else:
            census["public_assistance_share"] = np.nan

    if "transit_commute_share" not in census.columns:
        if {"public_transit_commuters", "total_commuters"} <= set(census.columns):
            census["transit_commute_share"] = _safe_divide(
                census["public_transit_commuters"],
                census["total_commuters"],
            )
        else:
            census["transit_commute_share"] = np.nan

    return census


def _build_snapshot_growth(frame: pd.DataFrame) -> pd.DataFrame:
    snapshots = _prepare_zip_frame(frame, dedupe=False)
    if snapshots.empty or "snapshot_year" not in snapshots.columns:
        return pd.DataFrame(columns=["zip", "income_growth_pct", "poverty_rate_change"])

    snapshots["snapshot_year"] = _coerce_numeric(snapshots["snapshot_year"])
    snapshots["median_household_income"] = _coerce_numeric(
        snapshots.get("median_household_income", pd.Series(np.nan, index=snapshots.index))
    )
    snapshots["poverty_rate"] = _coerce_numeric(
        snapshots.get("poverty_rate", pd.Series(np.nan, index=snapshots.index))
    )
    snapshots = snapshots.sort_values(["zip", "snapshot_year"], ignore_index=True)

    rows: list[dict[str, object]] = []
    for zip_code, group in snapshots.groupby("zip", sort=True):
        ordered = group.dropna(subset=["snapshot_year"]).copy()
        if ordered.empty:
            continue
        first_income = _coerce_numeric(ordered.iloc[0]["median_household_income"])
        latest_income = _coerce_numeric(ordered.iloc[-1]["median_household_income"])
        first_poverty = _coerce_numeric(ordered.iloc[0]["poverty_rate"])
        latest_poverty = _coerce_numeric(ordered.iloc[-1]["poverty_rate"])
        income_growth_pct = (
            ((float(latest_income) / float(first_income)) - 1.0) * 100.0
            if pd.notna(first_income) and pd.notna(latest_income) and float(first_income) > 0
            else np.nan
        )
        poverty_change = (
            float(latest_poverty) - float(first_poverty)
            if pd.notna(first_poverty) and pd.notna(latest_poverty)
            else np.nan
        )
        rows.append(
            {
                "zip": zip_code,
                "income_growth_pct": income_growth_pct,
                "poverty_rate_change": poverty_change,
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_economic_sidecar(
    *,
    zip_universe: list[str],
    census_current: pd.DataFrame,
    snapshot_growth: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame({"zip": zip_universe})
    frame = frame.merge(
        census_current[
            [
                "zip",
                "median_household_income",
                "poverty_rate",
                "unemployment_rate",
            ]
        ],
        on="zip",
        how="left",
    )
    frame = frame.merge(snapshot_growth, on="zip", how="left")

    income_scaled = _scale_unit_interval(frame["median_household_income"])
    poverty_inverse = 1 - _scale_unit_interval(frame["poverty_rate"])
    unemployment_inverse = 1 - _scale_unit_interval(frame["unemployment_rate"])
    growth_scaled = _scale_unit_interval(frame["income_growth_pct"])

    frame["economic_index"] = (
        (income_scaled.fillna(0.5) * 0.45)
        + (poverty_inverse.fillna(0.5) * 0.25)
        + (unemployment_inverse.fillna(0.5) * 0.2)
        + (growth_scaled.fillna(0.5) * 0.1)
    ).clip(lower=0, upper=1)
    frame["median_wage"] = (_coerce_numeric(frame["median_household_income"]) * 0.72).round(2)
    return frame[
        [
            "zip",
            "economic_index",
            "median_wage",
            "unemployment_rate",
            "income_growth_pct",
            "poverty_rate_change",
        ]
    ].sort_values("zip", ignore_index=True)


def _build_real_estate_sidecar(
    *,
    zip_universe: list[str],
    housing_current: pd.DataFrame,
    census_current: pd.DataFrame,
) -> pd.DataFrame:
    housing = _prepare_zip_frame(housing_current)
    for column in (
        "home_value",
        "annual_change_pct",
        "median_rent",
        "realtor_pending_ratio",
        "realtor_median_days_on_market",
    ):
        if column in housing.columns:
            housing[column] = _coerce_numeric(housing[column])

    frame = pd.DataFrame({"zip": zip_universe})
    frame = frame.merge(
        housing[
            [
                "zip",
                "home_value",
                "annual_change_pct",
                "median_rent",
                "realtor_pending_ratio",
                "realtor_median_days_on_market",
            ]
        ],
        on="zip",
        how="left",
    )
    frame = frame.merge(
        census_current[["zip", "owner_occupied_share", "median_household_income"]],
        on="zip",
        how="left",
    )

    owner_share = _coerce_numeric(frame["owner_occupied_share"]).clip(lower=0, upper=1)
    annual_change_scaled = _scale_unit_interval(frame["annual_change_pct"])
    frame["investor_purchase_share"] = (
        0.05 + ((1 - owner_share.fillna(0.5)) * 0.35) + (annual_change_scaled.fillna(0.5) * 0.1)
    ).clip(lower=0.02, upper=0.75)

    pending_scaled = _scale_unit_interval(frame["realtor_pending_ratio"])
    dom_inverse = 1 - _scale_unit_interval(frame["realtor_median_days_on_market"])
    frame["real_estate_pressure"] = (
        (pending_scaled.fillna(0.5) * 0.55) + (dom_inverse.fillna(0.5) * 0.45)
    ).clip(lower=0, upper=1)

    income = _coerce_numeric(frame["median_household_income"])
    annual_rent = _coerce_numeric(frame["median_rent"]) * 12
    frame["affordability_stress"] = _safe_divide(annual_rent, income)

    return frame[
        [
            "zip",
            "investor_purchase_share",
            "real_estate_pressure",
            "affordability_stress",
            "home_value",
            "annual_change_pct",
        ]
    ].sort_values("zip", ignore_index=True)


def _load_arrest_metrics(settings: "Settings", zip_universe: list[str]) -> pd.DataFrame:
    arrests_path = settings.project_root / "Police_Arrests.csv"
    if not arrests_path.exists():
        return pd.DataFrame(columns=["zip", "arrest_count_3y", "drug_related_share"])

    arrests = pd.read_csv(
        arrests_path,
        usecols=["Arrest Year", "Arrest Zipcode", "Drug Related"],
        dtype={"Arrest Zipcode": "string"},
        low_memory=False,
    )
    arrests = arrests.rename(columns={"Arrest Year": "arrest_year", "Arrest Zipcode": "zip"})
    arrests["zip"] = _normalize_zip_series(arrests["zip"])
    arrests["arrest_year"] = _coerce_numeric(arrests["arrest_year"])
    arrests = arrests.dropna(subset=["zip", "arrest_year"]).copy()
    arrests["arrest_year"] = arrests["arrest_year"].astype(int)
    if arrests.empty:
        return pd.DataFrame(columns=["zip", "arrest_count_3y", "drug_related_share"])

    year_floor = int(arrests["arrest_year"].max()) - 2
    arrests = arrests[arrests["arrest_year"] >= year_floor].copy()
    arrests = arrests[arrests["zip"].isin(zip_universe)].copy()
    if arrests.empty:
        return pd.DataFrame(columns=["zip", "arrest_count_3y", "drug_related_share"])

    arrests["drug_related_flag"] = arrests["Drug Related"].astype("string").str.lower().eq("yes")
    grouped = (
        arrests.groupby("zip", as_index=False)
        .agg(
            arrest_count_3y=("zip", "size"),
            drug_related_share=("drug_related_flag", "mean"),
        )
        .sort_values("zip", ignore_index=True)
    )
    return grouped


def _build_law_enforcement_sidecar(
    *,
    settings: "Settings",
    zip_universe: list[str],
    crime_current: pd.DataFrame,
    census_current: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame({"zip": zip_universe})

    crime = _prepare_zip_frame(crime_current, dedupe=False)
    if {"zip", "offense_family"} <= set(crime.columns):
        offense_family = crime["offense_family"].astype("string").str.lower()
        offense_flags = pd.DataFrame(
            {
                "zip": crime["zip"],
                "is_violent": offense_family.eq("violent").astype(int),
                "is_property": offense_family.eq("property").astype(int),
            }
        )
        crime_summary = (
            offense_flags.groupby("zip", as_index=False)
            .agg(
                total_incidents=("zip", "size"),
                violent_incidents=("is_violent", "sum"),
                property_incidents=("is_property", "sum"),
            )
            .sort_values("zip", ignore_index=True)
        )
    else:
        crime_summary = pd.DataFrame(columns=["zip", "total_incidents", "violent_incidents"])

    frame = frame.merge(crime_summary, on="zip", how="left")
    frame = frame.merge(census_current[["zip", "population"]], on="zip", how="left")
    frame["violent_rate_per_1000"] = _safe_divide(frame["violent_incidents"], frame["population"]) * 1000

    arrests = _load_arrest_metrics(settings, zip_universe)
    frame = frame.merge(arrests, on="zip", how="left")
    frame["arrest_rate_per_1000_3y"] = _safe_divide(frame["arrest_count_3y"], frame["population"]) * 1000

    violent_inverse = 1 - _scale_unit_interval(frame["violent_rate_per_1000"])
    arrest_scaled = _scale_unit_interval(frame["arrest_rate_per_1000_3y"])
    staffing_score = (violent_inverse.fillna(0.5) * 0.6) + (arrest_scaled.fillna(0.5) * 0.4)
    frame["law_staffing_score"] = staffing_score.clip(lower=0, upper=1)

    return frame[
        [
            "zip",
            "law_staffing_score",
            "arrest_count_3y",
            "arrest_rate_per_1000_3y",
            "drug_related_share",
            "violent_rate_per_1000",
        ]
    ].sort_values("zip", ignore_index=True)


def _build_social_services_sidecar(
    *,
    zip_universe: list[str],
    census_current: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame({"zip": zip_universe})
    frame = frame.merge(
        census_current[
            [
                "zip",
                "educational_attainment",
                "public_assistance_share",
                "poverty_rate",
            ]
        ],
        on="zip",
        how="left",
    )

    education_scaled = _scale_unit_interval(frame["educational_attainment"])
    assistance_inverse = 1 - _scale_unit_interval(frame["public_assistance_share"])
    poverty_inverse = 1 - _scale_unit_interval(frame["poverty_rate"])
    frame["clinic_access_score"] = (
        (education_scaled.fillna(0.5) * 0.5)
        + (assistance_inverse.fillna(0.5) * 0.25)
        + (poverty_inverse.fillna(0.5) * 0.25)
    ).clip(lower=0, upper=1)
    return frame[
        [
            "zip",
            "clinic_access_score",
            "educational_attainment",
            "public_assistance_share",
        ]
    ].sort_values("zip", ignore_index=True)


def _build_infrastructure_sidecar(
    *,
    zip_universe: list[str],
    census_current: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame({"zip": zip_universe})
    frame = frame.merge(
        census_current[
            [
                "zip",
                "transit_commute_share",
                "vacancy_proxy",
                "owner_occupied_share",
            ]
        ],
        on="zip",
        how="left",
    )
    transit_scaled = _scale_unit_interval(frame["transit_commute_share"])
    vacancy_inverse = 1 - _scale_unit_interval(frame["vacancy_proxy"])
    owner_scaled = _scale_unit_interval(frame["owner_occupied_share"])
    frame["park_access_score"] = (
        (transit_scaled.fillna(0.5) * 0.45)
        + (vacancy_inverse.fillna(0.5) * 0.25)
        + (owner_scaled.fillna(0.5) * 0.3)
    ).clip(lower=0, upper=1)
    frame["infrastructure_score"] = frame["park_access_score"]
    return frame[
        [
            "zip",
            "park_access_score",
            "infrastructure_score",
            "transit_commute_share",
            "vacancy_proxy",
        ]
    ].sort_values("zip", ignore_index=True)


def fetch_optional_zip_sidecars(settings: "Settings") -> SidecarArtifacts:
    """Generate additive optional sidecars from currently available raw datasets."""

    crime_current = _load_raw_frame(settings.raw_dir / "crime_records.csv")
    census_current_raw = _load_raw_frame(settings.raw_dir / "acs_zcta.csv")
    census_snapshots = _load_raw_frame(settings.raw_dir / "acs_zcta_snapshots.csv")
    housing_current = _load_raw_frame(settings.raw_dir / "housing_market.csv")

    zip_universe = _build_zip_universe(
        settings=settings,
        crime_current=crime_current,
        census_current=census_current_raw,
        housing_current=housing_current,
    )
    if not zip_universe:
        empty = pd.DataFrame(columns=["zip"])
        sidecar_paths: dict[str, str] = {}
        for category, filename in SIDECAR_FILE_BY_CATEGORY.items():
            output_path = settings.raw_dir / filename
            empty.to_csv(output_path, index=False)
            sidecar_paths[category] = str(output_path)
        metadata_path = settings.raw_dir / "optional_sidecars.metadata.json"
        write_json_artifact(
            metadata_path,
            {
                "retrieved_at_utc": utc_timestamp(),
                "zip_universe_size": 0,
                "rows_by_category": {category: 0 for category in SIDECAR_FILE_BY_CATEGORY},
                "sidecar_paths": sidecar_paths,
            },
        )
        return SidecarArtifacts(
            economic=sidecar_paths["economic"],
            real_estate=sidecar_paths["real_estate"],
            law_enforcement=sidecar_paths["law_enforcement"],
            social_services=sidecar_paths["social_services"],
            infrastructure=sidecar_paths["infrastructure"],
            metadata=str(metadata_path),
        )

    census_current = _prepare_census_features(census_current_raw)
    snapshot_growth = _build_snapshot_growth(census_snapshots)

    economic = _build_economic_sidecar(
        zip_universe=zip_universe,
        census_current=census_current,
        snapshot_growth=snapshot_growth,
    )
    real_estate = _build_real_estate_sidecar(
        zip_universe=zip_universe,
        housing_current=housing_current,
        census_current=census_current,
    )
    law_enforcement = _build_law_enforcement_sidecar(
        settings=settings,
        zip_universe=zip_universe,
        crime_current=crime_current,
        census_current=census_current,
    )
    social_services = _build_social_services_sidecar(
        zip_universe=zip_universe,
        census_current=census_current,
    )
    infrastructure = _build_infrastructure_sidecar(
        zip_universe=zip_universe,
        census_current=census_current,
    )

    category_frames = {
        "economic": economic,
        "real_estate": real_estate,
        "law_enforcement": law_enforcement,
        "social_services": social_services,
        "infrastructure": infrastructure,
    }

    sidecar_paths: dict[str, str] = {}
    for category, frame in category_frames.items():
        output_path = settings.raw_dir / SIDECAR_FILE_BY_CATEGORY[category]
        frame.to_csv(output_path, index=False)
        sidecar_paths[category] = str(output_path)

    metadata_path = settings.raw_dir / "optional_sidecars.metadata.json"
    write_json_artifact(
        metadata_path,
        {
            "retrieved_at_utc": utc_timestamp(),
            "zip_universe_size": len(zip_universe),
            "rows_by_category": {
                category: int(len(frame)) for category, frame in category_frames.items()
            },
            "columns_by_category": {
                category: list(frame.columns) for category, frame in category_frames.items()
            },
            "sidecar_paths": sidecar_paths,
            "sources": {
                "crime_current": str(settings.raw_dir / "crime_records.csv"),
                "census_current": str(settings.raw_dir / "acs_zcta.csv"),
                "census_snapshots": str(settings.raw_dir / "acs_zcta_snapshots.csv"),
                "housing_current": str(settings.raw_dir / "housing_market.csv"),
                "arrests": str(settings.project_root / "Police_Arrests.csv"),
            },
        },
    )

    return SidecarArtifacts(
        economic=sidecar_paths["economic"],
        real_estate=sidecar_paths["real_estate"],
        law_enforcement=sidecar_paths["law_enforcement"],
        social_services=sidecar_paths["social_services"],
        infrastructure=sidecar_paths["infrastructure"],
        metadata=str(metadata_path),
    )
