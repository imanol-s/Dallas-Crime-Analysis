"""Source acquisition entrypoints for Dallas crime analysis."""

from __future__ import annotations

from dallas_crime.acquire.utils import utc_timestamp, write_json_artifact
from dallas_crime.acquire.census import fetch_census_zcta_data
from dallas_crime.acquire.crime import fetch_crime_dataset
from dallas_crime.acquire.housing import fetch_housing_dataset


def run_acquire(settings) -> dict[str, str]:
    """Fetch and persist all raw datasets used by the project."""

    print("[acquire] Starting Dallas crime acquisition...", flush=True)
    crime_path = fetch_crime_dataset(settings)
    print(f"[acquire] Crime dataset written: {crime_path}", flush=True)

    print("[acquire] Starting Census ACS acquisition...", flush=True)
    census_path = fetch_census_zcta_data(settings)
    print(f"[acquire] Census dataset written: {census_path}", flush=True)

    print("[acquire] Starting housing acquisition...", flush=True)
    housing_path = fetch_housing_dataset(settings)
    print(f"[acquire] Housing dataset written: {housing_path}", flush=True)

    manifest_path = settings.raw_dir / "acquisition_metadata.json"
    write_json_artifact(
        manifest_path,
        {
            "retrieved_at_utc": utc_timestamp(),
            "datasets": {
                "crime": {
                    "data_path": crime_path,
                    "metadata_path": str(settings.raw_dir / "crime_records.metadata.json"),
                    "candidate_data_path": str(settings.raw_dir / "crime_zip_candidates.csv"),
                },
                "census": {
                    "data_path": census_path,
                    "metadata_path": str(settings.raw_dir / "acs_zcta.metadata.json"),
                },
                "housing": {
                    "data_path": housing_path,
                    "metadata_path": str(settings.raw_dir / "housing_market.metadata.json"),
                    "coverage_path": str(settings.raw_dir / "housing_zip_coverage.json"),
                    "history_data_path": str(settings.raw_dir / "housing_market_history.csv"),
                    "history_metadata_path": str(settings.raw_dir / "housing_market_history.metadata.json"),
                },
            },
        },
    )

    return {
        "crime": crime_path,
        "census": census_path,
        "housing": housing_path,
        "acquisition_metadata": str(manifest_path),
        "crime_candidates": str(settings.raw_dir / "crime_zip_candidates.csv"),
        "housing_coverage": str(settings.raw_dir / "housing_zip_coverage.json"),
        "housing_history": str(settings.raw_dir / "housing_market_history.csv"),
    }
