"""Census ACS acquisition helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dallas_crime.config import Settings

CensusOpener = Callable[[Request], Any]
REQUEST_TIMEOUT_SECONDS = 30

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
) -> list[list[str]]:
    """Fetch Census API JSON payload."""

    http_request = Request(build_census_url(request))
    if opener is None:
        response = urlopen(http_request, timeout=REQUEST_TIMEOUT_SECONDS)
    else:
        response = opener(http_request)
    payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Census payload must be a JSON array.")
    return payload


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

    request = CensusRequest(
        year=settings.census_year,
        dataset="acs/acs5",
        variables=DEFAULT_VARIABLES,
        within=None,
        api_key=settings.census_api_key,
    )
    payload = fetch_census_payload(request)
    frame = normalize_census_payload(payload, rename_map=DEFAULT_RENAME_MAP)

    occupied = frame["occupied_housing_units"].replace({0: np.nan})
    poverty_universe = frame["poverty_universe"].replace({0: np.nan})
    frame["owner_occupied_share"] = frame["owner_occupied_units"] / occupied
    frame["poverty_rate"] = frame["poverty_count"] / poverty_universe

    output_path = settings.raw_dir / "acs_zcta.csv"
    frame.to_csv(output_path, index=False)
    return str(output_path)


def _detect_geography_key(header: Sequence[str]) -> str:
    for candidate in ("zip code tabulation area", "zip code tabulation area:*", "zip"):
        if candidate in header:
            return candidate
    raise KeyError("Could not find ZIP geography column in Census payload.")
