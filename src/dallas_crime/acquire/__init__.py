"""Source acquisition entrypoints for Dallas crime analysis."""

from __future__ import annotations

from dallas_crime.acquire.census import fetch_census_zcta_data
from dallas_crime.acquire.crime import fetch_crime_dataset
from dallas_crime.acquire.housing import fetch_housing_dataset


def run_acquire(settings) -> dict[str, str]:
    """Fetch and persist all raw datasets used by the project."""

    return {
        "crime": fetch_crime_dataset(settings),
        "census": fetch_census_zcta_data(settings),
        "housing": fetch_housing_dataset(settings),
    }
