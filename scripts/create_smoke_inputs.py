"""Create minimal raw inputs for build/analyze CLI smoke checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def create_smoke_inputs(project_root: Path) -> None:
    raw = project_root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zip_codes = ["75201", "75202", "75203", "75204", "75205", "75206", "75207", "75208", "75209", "75210"]

    pd.DataFrame(
        {
            "incident_id": [
                "a1",
                "a2",
                "b1",
                "b2",
                "c1",
                "c2",
                "d1",
                "d2",
                "e1",
                "e2",
                "f1",
                "f2",
                "g1",
                "g2",
                "h1",
                "h2",
                "i1",
                "i2",
                "j1",
                "j2",
            ],
            "reported_at": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
                "2025-01-07",
                "2025-01-08",
                "2025-01-09",
                "2025-01-10",
                "2025-01-11",
                "2025-01-12",
                "2025-01-13",
                "2025-01-14",
                "2025-01-15",
                "2025-01-16",
                "2025-01-17",
                "2025-01-18",
                "2025-01-19",
                "2025-01-20",
            ],
            "offense": [
                "Assault",
                "Burglary",
                "Theft",
                "Assault",
                "Burglary",
                "Theft",
                "Assault",
                "Burglary",
                "Theft",
                "Burglary",
                "Assault",
                "Theft",
                "Burglary",
                "Assault",
                "Theft",
                "Burglary",
                "Assault",
                "Theft",
                "Assault",
                "Burglary",
            ],
            "offense_family": [
                "violent",
                "property",
                "other",
                "violent",
                "property",
                "other",
                "violent",
                "property",
                "other",
                "property",
                "violent",
                "other",
                "property",
                "violent",
                "other",
                "property",
                "violent",
                "other",
                "violent",
                "property",
            ],
            "zip": [
                "75201",
                "75201",
                "75202",
                "75202",
                "75203",
                "75203",
                "75204",
                "75204",
                "75205",
                "75205",
                "75206",
                "75206",
                "75207",
                "75207",
                "75208",
                "75208",
                "75209",
                "75209",
                "75210",
                "75210",
            ],
            "latitude": [
                32.770,
                32.771,
                32.772,
                32.773,
                32.774,
                32.775,
                32.776,
                32.777,
                32.778,
                32.779,
                32.780,
                32.781,
                32.782,
                32.783,
                32.784,
                32.785,
                32.786,
                32.787,
                32.788,
                32.789,
            ],
            "longitude": [
                -96.820,
                -96.819,
                -96.818,
                -96.817,
                -96.816,
                -96.815,
                -96.814,
                -96.813,
                -96.812,
                -96.811,
                -96.810,
                -96.809,
                -96.808,
                -96.807,
                -96.806,
                -96.805,
                -96.804,
                -96.803,
                -96.802,
                -96.801,
            ],
        }
    ).to_csv(raw / "crime_records.csv", index=False)

    history_rows: list[dict[str, object]] = []
    quarterly_dates = [
        "2024-01-15",
        "2024-04-15",
        "2024-07-15",
        "2024-10-15",
        "2025-01-15",
        "2025-04-15",
        "2025-07-15",
        "2025-10-15",
    ]
    offense_lookup = {
        "violent": "Assault",
        "property": "Burglary",
        "other": "Theft",
    }
    offense_cycle = ("violent", "property", "other")
    for quarter_index, reported_at in enumerate(quarterly_dates):
        for zip_index, zip_code in enumerate(zip_codes):
            offense_family = offense_cycle[(quarter_index + zip_index) % len(offense_cycle)]
            history_rows.append(
                {
                    "incident_id": f"h{quarter_index}_{zip_index}",
                    "reported_at": reported_at,
                    "offense": offense_lookup[offense_family],
                    "offense_family": offense_family,
                    "zip": zip_code,
                    "latitude": 32.770 + (zip_index * 0.002),
                    "longitude": -96.820 + (zip_index * 0.002),
                }
            )
    pd.DataFrame(history_rows).to_csv(raw / "crime_history_records.csv", index=False)

    pd.DataFrame(
        {
            "zip": ["75201", "75202", "75203", "75204", "75205", "75206", "75207", "75208", "75209", "75210"],
            "home_value": [240000, 250000, 260000, 270000, 280000, 290000, 300000, 310000, 320000, 330000],
            "as_of_date": ["2026-01-31"] * 10,
            "annual_change_pct": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
            "median_rent": [1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380],
            "source": ["smoke"] * 10,
            "source_url": ["https://example.org"] * 10,
        }
    ).to_csv(raw / "housing_market.csv", index=False)

    pd.DataFrame(
        {
            "zip": ["75201", "75202", "75203", "75204", "75205", "75206", "75207", "75208", "75209", "75210"],
            "population": [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000],
            "median_household_income": [52000, 54000, 56000, 58000, 60000, 62000, 64000, 66000, 68000, 70000],
            "poverty_rate": [0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12],
            "owner_occupied_share": [0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49],
            "median_gross_rent": [1100, 1140, 1180, 1220, 1260, 1300, 1340, 1380, 1420, 1460],
            "educational_attainment": [0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56],
        }
    ).to_csv(raw / "acs_zcta.csv", index=False)

    housing_history_rows: list[dict[str, object]] = []
    for zip_index, zip_code in enumerate(zip_codes):
        for year_offset, year in enumerate(range(2021, 2026)):
            housing_history_rows.append(
                {
                    "zip": zip_code,
                    "period_start": f"{year}-01-01",
                    "period_end": f"{year}-12-31",
                    "period_year": year,
                    "period_month": None,
                    "frequency": "annual",
                    "source": "fhfa_zip5",
                    "source_url": "https://example.org/fhfa",
                    "metric_label": "fhfa_hpi",
                    "price_signal_value": 100 + (zip_index * 3) + (year_offset * 2),
                    "price_signal_unit": "index_2000_base",
                }
            )
    pd.DataFrame(housing_history_rows).to_csv(raw / "housing_market_history.csv", index=False)

    acs_snapshot_rows: list[dict[str, object]] = []
    for zip_index, zip_code in enumerate(zip_codes):
        for year_offset, year in enumerate(range(2020, 2025)):
            occupied_units = 3800 + (zip_index * 110) + (year_offset * 20)
            owner_units = int(round(occupied_units * (0.30 + (zip_index * 0.01))))
            renter_units = occupied_units - owner_units
            poverty_universe = 9000 + (zip_index * 500) + (year_offset * 80)
            poverty_count = 1800 - (year_offset * 30) + (zip_index * 20)
            acs_snapshot_rows.append(
                {
                    "zip": zip_code,
                    "snapshot_year": year,
                    "population": 10000 + (zip_index * 1000) + (year_offset * 120),
                    "median_household_income": 50000 + (zip_index * 2000) + (year_offset * 1200),
                    "occupied_housing_units": occupied_units,
                    "owner_occupied_units": owner_units,
                    "renter_occupied_units": renter_units,
                    "median_gross_rent": 1100 + (zip_index * 40) + (year_offset * 25),
                    "poverty_universe": poverty_universe,
                    "poverty_count": poverty_count,
                }
            )
    pd.DataFrame(acs_snapshot_rows).to_csv(raw / "acs_zcta_snapshots.csv", index=False)

    pd.DataFrame(
        {
            "zip": zip_codes,
            "economic_index": [0.90 + (idx * 0.03) for idx in range(len(zip_codes))],
            "median_wage": [42000 + (idx * 2500) for idx in range(len(zip_codes))],
        }
    ).to_csv(raw / "dfw_zip_economic_sidecar.csv", index=False)
    pd.DataFrame(
        {
            "zip": zip_codes,
            "investor_purchase_share": [0.05 + (idx * 0.01) for idx in range(len(zip_codes))],
        }
    ).to_csv(raw / "dfw_zip_real_estate_sidecar.csv", index=False)
    pd.DataFrame(
        {
            "zip": zip_codes,
            "law_staffing_score": [0.65 + (idx * 0.02) for idx in range(len(zip_codes))],
        }
    ).to_csv(raw / "dfw_zip_law_enforcement_sidecar.csv", index=False)
    pd.DataFrame(
        {
            "zip": zip_codes,
            "clinic_access_score": [0.55 + (idx * 0.03) for idx in range(len(zip_codes))],
        }
    ).to_csv(raw / "dfw_zip_social_services_sidecar.csv", index=False)
    pd.DataFrame(
        {
            "zip": zip_codes,
            "park_access_score": [0.50 + (idx * 0.025) for idx in range(len(zip_codes))],
        }
    ).to_csv(raw / "dfw_zip_infrastructure_sidecar.csv", index=False)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: create_smoke_inputs.py <project_root>")
        return 2

    create_smoke_inputs(Path(sys.argv[1]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
