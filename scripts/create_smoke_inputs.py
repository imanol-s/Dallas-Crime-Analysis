"""Create minimal raw inputs for build/analyze CLI smoke checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def create_smoke_inputs(project_root: Path) -> None:
    raw = project_root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

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
            ],
        }
    ).to_csv(raw / "crime_records.csv", index=False)

    pd.DataFrame(
        {
            "zip": ["75201", "75202", "75203", "75204", "75205", "75206", "75207", "75208", "75209"],
            "home_value": [240000, 250000, 260000, 270000, 280000, 290000, 300000, 310000, 320000],
            "as_of_date": ["2026-01-31"] * 9,
            "annual_change_pct": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "median_rent": [1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360],
            "source": ["smoke"] * 9,
            "source_url": ["https://example.org"] * 9,
        }
    ).to_csv(raw / "housing_market.csv", index=False)

    pd.DataFrame(
        {
            "zip": ["75201", "75202", "75203", "75204", "75205", "75206", "75207", "75208", "75209"],
            "population": [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000],
            "median_household_income": [52000, 54000, 56000, 58000, 60000, 62000, 64000, 66000, 68000],
            "poverty_rate": [0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13],
            "owner_occupied_share": [0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47],
            "median_gross_rent": [1100, 1140, 1180, 1220, 1260, 1300, 1340, 1380, 1420],
        }
    ).to_csv(raw / "acs_zcta.csv", index=False)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: create_smoke_inputs.py <project_root>")
        return 2

    create_smoke_inputs(Path(sys.argv[1]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
