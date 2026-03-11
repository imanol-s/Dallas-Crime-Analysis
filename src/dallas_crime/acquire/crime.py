"""Dallas OpenData crime acquisition helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

if TYPE_CHECKING:
    from dallas_crime.config import Settings

SocrataOpener = Callable[[Request], Any]
REQUEST_TIMEOUT_SECONDS = 30

_ZIP_ALIASES = (
    "zipcode",
    "zip",
    "zip_code",
    "arrest_zipcode",
    "reporting_area_zip",
    "arlzip",
    "hzip",
)
_INCIDENT_ALIASES = (
    "incidentnum",
    "incident_number",
    "incident_id",
    "case_number",
    "servnumid",
)
_DATE_ALIASES = (
    "date1",
    "reporteddate",
    "reported_date",
    "occurred_date",
    "offense_date",
    "datetime",
    "ararrestdate",
)
_OFFENSE_ALIASES = (
    "offincident",
    "offense",
    "offense_description",
    "type_of_incident",
    "ucr_description",
    "ucr_offdesc",
    "nibrs_crime",
)
_LAT_ALIASES = ("latitude", "lat")
_LON_ALIASES = ("longitude", "lon", "lng")

_VIOLENT_KEYWORDS = (
    "ASSAULT",
    "AGGRAVATED",
    "HOMICIDE",
    "MURDER",
    "ROBBERY",
    "RAPE",
    "KIDNAPPING",
    "WEAPON",
    "SHOOTING",
)
_PROPERTY_KEYWORDS = (
    "BURGLARY",
    "THEFT",
    "MOTOR VEHICLE",
    "AUTO THEFT",
    "LARCENY",
    "FRAUD",
    "ARSON",
    "CRIMINAL MISCHIEF",
    "VANDALISM",
    "PROPERTY",
)


@dataclass(frozen=True)
class DallasCrimeSourceConfig:
    """Config for the official Dallas OpenData crime source."""

    dataset_url: str
    limit: int = 50_000
    select: tuple[str, ...] = (
        "incidentnum",
        "date1",
        "reporteddate",
        "offincident",
        "zip_code",
        "geocoded_column",
    )
    where_clause: str | None = None
    order_by: str | None = "date1"
    app_token: str | None = None


def build_crime_url(config: DallasCrimeSourceConfig, *, offset: int = 0) -> str:
    """Build a Socrata query URL for the configured dataset."""

    params = {
        "$limit": str(config.limit),
        "$offset": str(offset),
        "$select": ",".join(config.select),
    }
    if config.where_clause:
        params["$where"] = config.where_clause
    if config.order_by:
        params["$order"] = config.order_by
    if config.app_token:
        params["$$app_token"] = config.app_token
    return f"{config.dataset_url}?{urlencode(params)}"


def fetch_crime_payload(
    config: DallasCrimeSourceConfig,
    *,
    offset: int = 0,
    opener: SocrataOpener | None = None,
) -> list[dict[str, Any]]:
    """Fetch one page of JSON records from Dallas OpenData."""

    request = Request(build_crime_url(config, offset=offset))
    if opener is None:
        response = urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)
    else:
        response = opener(request)
    payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Crime payload must be a JSON array of records.")
    return payload


def fetch_all_crime_records(
    config: DallasCrimeSourceConfig,
    *,
    opener: SocrataOpener | None = None,
    max_pages: int = 20,
) -> list[dict[str, Any]]:
    """Fetch all configured pages from the crime endpoint."""

    records: list[dict[str, Any]] = []
    offset = 0
    for _ in range(max_pages):
        payload = fetch_crime_payload(config, offset=offset, opener=opener)
        if not payload:
            break
        records.extend(payload)
        if len(payload) < config.limit:
            break
        offset += config.limit
    return records


def normalize_crime_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalize Dallas crime records into a stable tabular shape."""

    rows: list[dict[str, Any]] = []
    for record in records:
        offense = _coalesce(record, _OFFENSE_ALIASES)
        rows.append(
            {
                "incident_id": _coalesce(record, _INCIDENT_ALIASES),
                "reported_at": pd.to_datetime(_coalesce(record, _DATE_ALIASES), errors="coerce"),
                "offense": offense,
                "offense_family": classify_offense_family(offense),
                "zip": _normalize_zip(_coalesce(record, _ZIP_ALIASES)),
                "latitude": _coerce_float(_extract_coordinate(record, _LAT_ALIASES)),
                "longitude": _coerce_float(_extract_coordinate(record, _LON_ALIASES)),
            }
        )

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "reported_at",
                "offense",
                "offense_family",
                "zip",
                "latitude",
                "longitude",
            ]
        )

    return frame.dropna(subset=["zip", "reported_at"]).drop_duplicates(
        subset=["incident_id"], keep="last"
    )


def fetch_crime_dataset(settings: "Settings") -> str:
    """Fetch Dallas crime records and persist them as normalized CSV."""

    config = DallasCrimeSourceConfig(
        dataset_url=settings.crime_source_url,
        limit=settings.crime_limit,
        where_clause=settings.resolved_crime_where_clause(),
    )
    records = fetch_all_crime_records(config)
    frame = normalize_crime_records(records)
    if frame.empty:
        raise ValueError("Dallas crime source returned no records for the configured time window.")

    output_path = settings.raw_dir / "crime_records.csv"
    frame.to_csv(output_path, index=False)
    return str(output_path)


def classify_offense_family(offense: Any) -> str:
    """Map offense text to a broad family used downstream."""

    normalized = re.sub(r"[^A-Z0-9 ]+", " ", str(offense or "").upper()).strip()
    if any(token in normalized for token in _VIOLENT_KEYWORDS):
        return "violent"
    if any(token in normalized for token in _PROPERTY_KEYWORDS):
        return "property"
    return "other"


def _coalesce(record: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _extract_coordinate(record: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    direct_value = _coalesce(record, aliases)
    if direct_value not in (None, ""):
        return direct_value
    location = record.get("location") or record.get("location1") or record.get("geocoded_column")
    if isinstance(location, Mapping):
        for key in aliases:
            if key in location and location[key] not in (None, ""):
                return location[key]
    return None


def _normalize_zip(value: Any) -> str | None:
    if value in (None, ""):
        return None
    digits = re.findall(r"\d", str(value))
    if len(digits) < 5:
        return None
    return "".join(digits[:5])


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
