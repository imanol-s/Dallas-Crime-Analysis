"""Source acquisition entrypoints for Dallas crime analysis."""

from __future__ import annotations

from dallas_crime.acquire.utils import utc_timestamp, write_json_artifact
from dallas_crime.acquire.census import (
    fetch_census_zcta_data,
    fetch_census_zcta_snapshot_data,
)
from dallas_crime.acquire.crime import (
    fetch_crime_dataset,
    fetch_crime_history_dataset,
)
from dallas_crime.acquire.housing import fetch_housing_dataset
from dallas_crime.acquire.sidecars import fetch_optional_zip_sidecars


def run_acquire(settings) -> dict[str, str]:
    """Fetch and persist all raw datasets used by the project."""

    print("[acquire] Starting Dallas crime acquisition...", flush=True)
    crime_path = fetch_crime_dataset(settings)
    print(f"[acquire] Crime dataset written: {crime_path}", flush=True)

    print("[acquire] Starting long-horizon Dallas crime acquisition...", flush=True)
    crime_history_path = fetch_crime_history_dataset(settings)
    print(f"[acquire] Crime history dataset written: {crime_history_path}", flush=True)

    print("[acquire] Starting Census ACS acquisition...", flush=True)
    census_path = fetch_census_zcta_data(settings)
    print(f"[acquire] Census dataset written: {census_path}", flush=True)

    print("[acquire] Starting Census ACS snapshot acquisition...", flush=True)
    census_snapshots_path = fetch_census_zcta_snapshot_data(settings)
    print(f"[acquire] Census snapshot dataset written: {census_snapshots_path}", flush=True)

    print("[acquire] Starting housing acquisition...", flush=True)
    housing_path = fetch_housing_dataset(settings)
    print(f"[acquire] Housing dataset written: {housing_path}", flush=True)

    print("[acquire] Building optional ZIP sidecars...", flush=True)
    sidecar_artifacts = fetch_optional_zip_sidecars(settings)
    print(
        "[acquire] Optional sidecars written: "
        f"economic={sidecar_artifacts.economic}, "
        f"real_estate={sidecar_artifacts.real_estate}, "
        f"law_enforcement={sidecar_artifacts.law_enforcement}, "
        f"social_services={sidecar_artifacts.social_services}, "
        f"infrastructure={sidecar_artifacts.infrastructure}",
        flush=True,
    )

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
                    "history_data_path": crime_history_path,
                    "history_metadata_path": str(
                        settings.raw_dir / "crime_history_records.metadata.json"
                    ),
                },
                "census": {
                    "data_path": census_path,
                    "metadata_path": str(settings.raw_dir / "acs_zcta.metadata.json"),
                    "snapshot_data_path": census_snapshots_path,
                    "snapshot_metadata_path": str(
                        settings.raw_dir / "acs_zcta_snapshots.metadata.json"
                    ),
                },
                "housing": {
                    "data_path": housing_path,
                    "metadata_path": str(settings.raw_dir / "housing_market.metadata.json"),
                    "coverage_path": str(settings.raw_dir / "housing_zip_coverage.json"),
                    "history_data_path": str(settings.raw_dir / "housing_market_history.csv"),
                    "history_metadata_path": str(
                        settings.raw_dir / "housing_market_history.metadata.json"
                    ),
                },
                "sidecars": {
                    "economic_data_path": sidecar_artifacts.economic,
                    "real_estate_data_path": sidecar_artifacts.real_estate,
                    "law_enforcement_data_path": sidecar_artifacts.law_enforcement,
                    "social_services_data_path": sidecar_artifacts.social_services,
                    "infrastructure_data_path": sidecar_artifacts.infrastructure,
                    "metadata_path": sidecar_artifacts.metadata,
                },
            },
        },
    )

    return {
        "crime": crime_path,
        "crime_history": crime_history_path,
        "census": census_path,
        "census_snapshots": census_snapshots_path,
        "housing": housing_path,
        "acquisition_metadata": str(manifest_path),
        "crime_candidates": str(settings.raw_dir / "crime_zip_candidates.csv"),
        "housing_coverage": str(settings.raw_dir / "housing_zip_coverage.json"),
        "housing_history": str(settings.raw_dir / "housing_market_history.csv"),
        "economic_sidecar": sidecar_artifacts.economic,
        "real_estate_sidecar": sidecar_artifacts.real_estate,
        "law_enforcement_sidecar": sidecar_artifacts.law_enforcement,
        "social_services_sidecar": sidecar_artifacts.social_services,
        "infrastructure_sidecar": sidecar_artifacts.infrastructure,
        "sidecar_metadata": sidecar_artifacts.metadata,
    }
