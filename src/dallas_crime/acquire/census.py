"""Census ACS acquisition helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from dallas_crime.acquire.utils import (
    REQUEST_TIMEOUT_SECONDS,
    AcquisitionError,
    run_with_retry,
    utc_timestamp,
    write_json_artifact,
)

if TYPE_CHECKING:
    from dallas_crime.config import Settings

CensusOpener = Callable[[Request], Any]
ZCTA_GEO_ID_PREFIX = "860Z200US"
BULK_TABLE_BASE_URL = "https://www2.census.gov/programs-surveys/acs/summary_file"
BULK_TABLE_SPECS = {
    "b01003": {"B01003_E001": "population"},
    "b19013": {"B19013_E001": "median_household_income"},
    "b25003": {
        "B25003_E001": "occupied_housing_units",
        "B25003_E002": "owner_occupied_units",
        "B25003_E003": "renter_occupied_units",
    },
    "b25064": {"B25064_E001": "median_gross_rent"},
    "b17001": {
        "B17001_E001": "poverty_universe",
        "B17001_E002": "poverty_count",
    },
}

DEFAULT_VARIABLES = (
    "B01003_001E",
    "B19013_001E",
    "B25003_001E",
    "B25003_002E",
    "B25003_003E",
    "B25064_001E",
    "B17001_001E",
    "B17001_002E",
)
OPTIONAL_VARIABLES = (
    "B23025_003E",
    "B23025_005E",
    "B25002_001E",
    "B25002_003E",
    "B15003_001E",
    "B15003_022E",
    "B15003_023E",
    "B15003_024E",
    "B15003_025E",
    "B19057_001E",
    "B19057_002E",
    "B08301_001E",
    "B08301_010E",
)
DEFAULT_RENAME_MAP = {
    "B01003_001E": "population",
    "B19013_001E": "median_household_income",
    "B25003_001E": "occupied_housing_units",
    "B25003_002E": "owner_occupied_units",
    "B25003_003E": "renter_occupied_units",
    "B25064_001E": "median_gross_rent",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "poverty_count",
}
OPTIONAL_RENAME_MAP = {
    "B23025_003E": "labor_force",
    "B23025_005E": "unemployed_count",
    "B25002_001E": "total_housing_units",
    "B25002_003E": "vacant_housing_units",
    "B15003_001E": "education_population_25_plus",
    "B15003_022E": "education_bachelors_count",
    "B15003_023E": "education_masters_count",
    "B15003_024E": "education_professional_count",
    "B15003_025E": "education_doctorate_count",
    "B19057_001E": "households_total",
    "B19057_002E": "households_public_assistance",
    "B08301_001E": "total_commuters",
    "B08301_010E": "public_transit_commuters",
}


@dataclass(frozen=True)
class CensusRequest:
    """Minimal request definition for Census API pulls."""

    year: int
    variables: tuple[str, ...]
    dataset: str = "acs/acs5"
    geography: str = "zip code tabulation area:*"
    within: str | None = None
    api_key: str | None = None
    base_url: str = "https://api.census.gov/data"


@dataclass(frozen=True)
class CensusYearResult:
    """Normalized ACS result for a single snapshot year."""

    year: int
    frame: pd.DataFrame
    source_kind: str
    source_url: str
    rows_returned: int
    rows_before_zip_filter: int
    rows_after_zip_filter: int
    fallback_reason: str | None = None


def build_census_url(request: CensusRequest) -> str:
    """Build a stable Census API URL."""

    params = {
        "get": ",".join(request.variables),
        "for": request.geography,
    }
    if request.within:
        params["in"] = request.within
    if request.api_key:
        params["key"] = request.api_key
    return f"{request.base_url}/{request.year}/{request.dataset}?{urlencode(params)}"


def fetch_census_payload(
    request: CensusRequest,
    *,
    opener: CensusOpener | None = None,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    max_attempts: int = 1,
    backoff_seconds: float = 0.0,
) -> list[list[str]]:
    """Fetch Census API JSON payload."""

    url = build_census_url(request)

    def _read_payload() -> list[list[str]]:
        http_request = Request(url)
        if opener is None:
            response = urlopen(http_request, timeout=timeout_seconds)
        else:
            response = opener(http_request)
        payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Census payload must be a JSON array.")
        return payload

    return run_with_retry(
        "Census ACS request",
        _read_payload,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        retryable_exceptions=(
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            ValueError,
        ),
        hint=(
            "Verify DCA_CENSUS_YEAR and CENSUS_API_KEY, and consider raising "
            "DCA_ACQUIRE_MAX_ATTEMPTS or DCA_ACQUIRE_TIMEOUT_SECONDS."
        ),
    )


def normalize_census_payload(
    payload: Sequence[Sequence[str]],
    *,
    rename_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Convert Census header/value arrays into a typed dataframe."""

    if not payload:
        return pd.DataFrame(columns=["zip"])

    header = list(payload[0])
    values = [list(row) for row in payload[1:]]
    frame = pd.DataFrame(values, columns=header)

    geography_key = _detect_geography_key(header)
    frame["zip"] = frame[geography_key].astype(str).str.zfill(5)

    rename_map = dict(rename_map or {})
    drop_columns = [column for column in ("state", geography_key) if column in frame.columns]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)
    frame = frame.rename(columns=rename_map)

    for column in frame.columns:
        if column == "zip":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def fetch_census_zcta_data(settings: "Settings") -> str:
    """Fetch ACS ZCTA data and persist a normalized CSV."""

    crime_zips = _load_crime_zip_universe(settings)
    if not crime_zips:
        raise AcquisitionError(
            "Could not derive ZIP universe from crime_records.csv for ACS filtering."
        )

    result = fetch_census_year_data(
        settings,
        crime_zips=crime_zips,
        year=settings.census_year,
    )

    output_path = settings.raw_dir / "acs_zcta.csv"
    result.frame.to_csv(output_path, index=False)
    metadata_path = settings.raw_dir / "acs_zcta.metadata.json"
    write_json_artifact(
        metadata_path,
        {
            "dataset": "acs_zcta",
            "retrieved_at_utc": utc_timestamp(),
            "source_kind": result.source_kind,
            "source_url": result.source_url,
            "query": {
                "year": result.year,
                "dataset": "acs/acs5",
                "variables": list((*DEFAULT_VARIABLES, *OPTIONAL_VARIABLES)),
                "geography": "zip code tabulation area:*",
                "within": None,
                "state_fips": settings.census_state_fips,
                "zip_filter_source": str(settings.raw_dir / "crime_records.csv"),
            },
            "row_counts": {
                "rows_returned": result.rows_returned,
                "rows_before_zip_filter": result.rows_before_zip_filter,
                "rows_after_zip_filter": result.rows_after_zip_filter,
                "crime_zip_universe_size": int(len(crime_zips)),
            },
            "fallback": {
                "activated": result.fallback_reason is not None,
                "reason": result.fallback_reason,
            },
            "output_path": str(output_path),
        },
    )
    return str(output_path)


def fetch_census_zcta_snapshot_data(settings: "Settings") -> str:
    """Fetch multi-year ACS ZCTA snapshots and persist a stacked CSV."""

    crime_zips = _load_crime_zip_universe(settings)
    if not crime_zips:
        raise AcquisitionError(
            "Could not derive ZIP universe from crime_records.csv for ACS snapshot filtering."
        )

    snapshot_frames: list[pd.DataFrame] = []
    year_results: list[dict[str, Any]] = []
    for year in settings.census_snapshot_years:
        result = fetch_census_year_data(
            settings,
            crime_zips=crime_zips,
            year=year,
        )
        frame = result.frame.copy()
        frame.insert(1, "snapshot_year", year)
        snapshot_frames.append(frame)
        year_results.append(
            {
                "year": year,
                "source_kind": result.source_kind,
                "source_url": result.source_url,
                "rows_returned": result.rows_returned,
                "rows_before_zip_filter": result.rows_before_zip_filter,
                "rows_after_zip_filter": result.rows_after_zip_filter,
                "fallback_activated": result.fallback_reason is not None,
                "fallback_reason": result.fallback_reason,
            }
        )

    if not snapshot_frames:
        raise AcquisitionError("No ACS snapshot years were configured for acquisition.")

    combined = pd.concat(snapshot_frames, ignore_index=True)
    combined = combined.sort_values(["zip", "snapshot_year"], ignore_index=True)

    output_path = settings.raw_dir / "acs_zcta_snapshots.csv"
    combined.to_csv(output_path, index=False)
    metadata_path = settings.raw_dir / "acs_zcta_snapshots.metadata.json"
    write_json_artifact(
        metadata_path,
        {
            "dataset": "acs_zcta_snapshots",
            "retrieved_at_utc": utc_timestamp(),
            "snapshot_years": list(settings.census_snapshot_years),
            "row_counts": {
                "rows_written": int(len(combined)),
                "crime_zip_universe_size": int(len(crime_zips)),
            },
            "year_results": year_results,
            "output_path": str(output_path),
        },
    )
    return str(output_path)


def fetch_census_year_data(
    settings: "Settings",
    *,
    crime_zips: set[str],
    year: int,
) -> CensusYearResult:
    """Fetch a single ACS ZCTA snapshot year and normalize it."""

    request = CensusRequest(
        year=year,
        dataset="acs/acs5",
        variables=(*DEFAULT_VARIABLES, *OPTIONAL_VARIABLES),
        within=None,
        api_key=settings.census_api_key,
    )
    payload: list[list[str]] | None = None
    source_kind = "api"
    source_url = f"{request.base_url}/{request.year}/{request.dataset}"
    fallback_reason: str | None = None

    try:
        payload = fetch_census_payload(
            request,
            timeout_seconds=settings.acquire_timeout_seconds,
            max_attempts=1,
            backoff_seconds=settings.acquire_backoff_seconds,
        )
        frame = normalize_census_payload(
            payload,
            rename_map={**DEFAULT_RENAME_MAP, **OPTIONAL_RENAME_MAP},
        )
        if frame.empty:
            raise AcquisitionError("Census ACS payload returned no rows.")
    except (AcquisitionError, KeyError) as exc:
        fallback_reason = str(exc)
        source_kind = "bulk_table_based"
        source_url = _build_bulk_table_directory_url(year)
        frame = fetch_census_bulk_dataset(
            settings,
            crime_zips=crime_zips,
            year=year,
        )

    pre_filter_rows = int(len(frame))
    frame = frame[frame["zip"].isin(crime_zips)].copy()
    if frame.empty:
        raise AcquisitionError(
            "No ACS rows matched the crime ZIP universe. Verify Dallas crime and ACS "
            "geography settings."
        )

    frame = _finalize_census_frame(frame)
    return CensusYearResult(
        year=year,
        frame=frame,
        source_kind=source_kind,
        source_url=source_url,
        rows_returned=max(len(payload) - 1, 0) if payload is not None else pre_filter_rows,
        rows_before_zip_filter=pre_filter_rows,
        rows_after_zip_filter=int(len(frame)),
        fallback_reason=fallback_reason,
    )


def fetch_census_bulk_dataset(
    settings: "Settings",
    *,
    crime_zips: set[str],
    year: int,
) -> pd.DataFrame:
    """Fetch ACS ZCTA data from official Census bulk table files."""

    frames: list[pd.DataFrame] = []
    for table_id, rename_map in BULK_TABLE_SPECS.items():
        frame = _fetch_bulk_table_frame(
            year=year,
            table_id=table_id,
            rename_map=rename_map,
            target_zips=crime_zips,
            timeout_seconds=settings.acquire_timeout_seconds,
            max_attempts=settings.acquire_max_attempts,
            backoff_seconds=settings.acquire_backoff_seconds,
        )
        frames.append(frame)

    merged: pd.DataFrame | None = None
    for frame in frames:
        if merged is None:
            merged = frame
            continue
        merged = merged.merge(frame, on="zip", how="outer")

    if merged is None or merged.empty:
        raise AcquisitionError(
            "Official Census bulk ACS tables returned no ZIP rows for the crime ZIP universe."
        )
    return merged


def _fetch_bulk_table_frame(
    *,
    year: int,
    table_id: str,
    rename_map: Mapping[str, str],
    target_zips: set[str],
    timeout_seconds: int,
    max_attempts: int,
    backoff_seconds: float,
) -> pd.DataFrame:
    """Stream an ACS table-based summary file and retain only target ZIPs."""

    url = _build_bulk_table_url(year, table_id)

    def _read_table() -> pd.DataFrame:
        request = Request(url)
        with urlopen(request, timeout=timeout_seconds) as response:
            text_stream = io.TextIOWrapper(response, encoding="utf-8")
            reader = csv.DictReader(text_stream, delimiter="|")
            rows: list[dict[str, Any]] = []
            for record in reader:
                geo_id = str(record.get("GEO_ID", ""))
                zip_code = _extract_zcta_zip(geo_id)
                if zip_code is None or zip_code not in target_zips:
                    continue

                row: dict[str, Any] = {"zip": zip_code}
                for source_column, target_column in rename_map.items():
                    row[target_column] = record.get(source_column)
                rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["zip", *rename_map.values()])

        frame = frame.drop_duplicates(subset=["zip"], keep="first")
        for target_column in rename_map.values():
            frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
        return frame[["zip", *rename_map.values()]]

    return run_with_retry(
        f"Census ACS bulk table request ({table_id})",
        _read_table,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        retryable_exceptions=(
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            csv.Error,
            UnicodeDecodeError,
        ),
        hint=(
            f"Verify the official ACS bulk table path for {year} and table {table_id}: "
            f"{_build_bulk_table_directory_url(year)}"
        ),
    )


def _build_bulk_table_url(year: int, table_id: str) -> str:
    table_name = table_id.lower()
    return (
        f"{_build_bulk_table_directory_url(year)}/"
        f"acsdt5y{year}-{table_name}.dat"
    )


def _build_bulk_table_directory_url(year: int) -> str:
    return f"{BULK_TABLE_BASE_URL}/{year}/table-based-SF/data/5YRData"


def _extract_zcta_zip(geo_id: str) -> str | None:
    if not geo_id.startswith(ZCTA_GEO_ID_PREFIX):
        return None
    zip_code = geo_id[len(ZCTA_GEO_ID_PREFIX) : len(ZCTA_GEO_ID_PREFIX) + 5]
    if len(zip_code) != 5 or not zip_code.isdigit():
        return None
    return zip_code


def _finalize_census_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"zip", *DEFAULT_RENAME_MAP.values()}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise AcquisitionError(f"Census ACS dataset is missing required columns: {missing}")

    frame = frame.copy()
    numeric_columns = [column for column in frame.columns if column != "zip"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "median_household_income" in frame.columns:
        neg_mask = frame["median_household_income"].notna() & (frame["median_household_income"] < 0)
        if neg_mask.any():
            bad_zips = frame.loc[neg_mask, "zip"].tolist()
            print(
                f"[census] WARNING: {int(neg_mask.sum())} ZIP(s) have sentinel "
                f"median_household_income < 0 ({bad_zips}); replacing with NaN.",
                flush=True,
            )
            frame.loc[neg_mask, "median_household_income"] = np.nan

    occupied = frame["occupied_housing_units"].replace({0: np.nan})
    poverty_universe = frame["poverty_universe"].replace({0: np.nan})
    frame["owner_occupied_share"] = frame["owner_occupied_units"] / occupied
    frame["renter_occupied_share"] = frame["renter_occupied_units"] / occupied
    frame["poverty_rate"] = frame["poverty_count"] / poverty_universe

    if {"unemployed_count", "labor_force"} <= set(frame.columns):
        labor_force = frame["labor_force"].replace({0: np.nan})
        frame["unemployment_rate"] = frame["unemployed_count"] / labor_force

    if {"vacant_housing_units", "total_housing_units"} <= set(frame.columns):
        housing_units = frame["total_housing_units"].replace({0: np.nan})
        frame["vacancy_proxy"] = frame["vacant_housing_units"] / housing_units

    education_parts = [
        "education_bachelors_count",
        "education_masters_count",
        "education_professional_count",
        "education_doctorate_count",
    ]
    if set(education_parts).issubset(frame.columns):
        frame["bachelors_or_higher_count"] = frame[education_parts].sum(
            axis=1,
            min_count=1,
        )
    if {"bachelors_or_higher_count", "education_population_25_plus"} <= set(frame.columns):
        education_universe = frame["education_population_25_plus"].replace({0: np.nan})
        frame["educational_attainment"] = frame["bachelors_or_higher_count"] / education_universe

    if {"households_public_assistance", "households_total"} <= set(frame.columns):
        households_total = frame["households_total"].replace({0: np.nan})
        frame["public_assistance_share"] = (
            frame["households_public_assistance"] / households_total
        )

    if {"public_transit_commuters", "total_commuters"} <= set(frame.columns):
        total_commuters = frame["total_commuters"].replace({0: np.nan})
        frame["transit_commute_share"] = frame["public_transit_commuters"] / total_commuters
    return frame


def _detect_geography_key(header: Sequence[str]) -> str:
    for candidate in ("zip code tabulation area", "zip code tabulation area:*", "zip"):
        if candidate in header:
            return candidate
    raise KeyError("Could not find ZIP geography column in Census payload.")


def _load_crime_zip_universe(settings: "Settings") -> set[str]:
    crime_path = settings.raw_dir / "crime_records.csv"
    if not crime_path.exists():
        raise AcquisitionError(
            f"Crime raw dataset not found at {crime_path}. Run crime acquisition before census."
        )

    frame = pd.read_csv(crime_path, usecols=["zip"])
    return {
        str(value).zfill(5)
        for value in frame["zip"].dropna().astype(str)
        if str(value).zfill(5).isdigit() and len(str(value).zfill(5)) == 5
    }
