import math
from pathlib import Path

import pandas as pd
import pytest

from dallas_crime.config import Settings
from dallas_crime.pipeline.analyze import run_zip_regression
from dallas_crime.pipeline.build import (
    aggregate_crime_data,
    build_all,
    build_model_dataset,
    normalize_zip,
    normalize_zip_series,
)


def test_normalize_zip_variants():
    assert normalize_zip("75201-1234") == "75201"
    assert normalize_zip(75202.0) == "75202"
    assert normalize_zip(" 75203 ") == "75203"
    assert normalize_zip(None) is None
    assert normalize_zip("unknown") is None

    series = normalize_zip_series(pd.Series(["75204", "75205-9876", None]))
    assert series.tolist() == ["75204", "75205", None]


def test_aggregate_crime_data_builds_zip_level_rates():
    crime_df = pd.DataFrame(
        {
            "zip": ["75201", "75201", "75201-1234", "75202", "75202", None],
            "incident_date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
            ],
            "crime_category": ["violent", "property", "other", "violent", "property", "violent"],
        }
    )
    population_df = pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "population": [1000, 2000],
        }
    )

    result = aggregate_crime_data(crime_df, population_df)

    assert result["zip"].tolist() == ["75201", "75202"]
    assert result.loc[0, "total_incidents"] == 3
    assert result.loc[0, "violent_incidents"] == 1
    assert result.loc[0, "property_incidents"] == 1
    assert result.loc[0, "other_incidents"] == 1
    assert result.loc[0, "total_rate_per_1000"] == pytest.approx(3.0)
    assert result.loc[1, "total_rate_per_1000"] == pytest.approx(1.0)
    assert str(result.loc[0, "period_start"].date()) == "2025-01-01"
    assert str(result.loc[0, "period_end"].date()) == "2025-01-05"


def test_build_model_dataset_merges_crime_housing_and_controls():
    crime_zip_df = pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "total_rate_per_1000": [12.0, 8.0],
            "violent_rate_per_1000": [4.0, 2.0],
            "property_rate_per_1000": [6.0, 5.0],
        }
    )
    housing_df = pd.DataFrame(
        {
            "zip": ["75201-1234", "75202"],
            "home_value": [500000, 350000],
            "as_of_date": ["2026-01-01", "2026-01-01"],
            "source": ["zillow", "zillow"],
        }
    )
    controls_df = pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "median_household_income": [90000, 70000],
            "poverty_rate": [0.08, 0.12],
            "owner_occupied_share": [0.42, 0.55],
            "median_gross_rent": [1800, 1450],
        }
    )

    result = build_model_dataset(crime_zip_df, housing_df, controls_df)

    assert result["zip"].tolist() == ["75201", "75202"]
    assert result.loc[0, "home_value"] == 500000
    assert result.loc[0, "source"] == "zillow"
    assert result.loc[0, "median_household_income"] == 90000
    assert result.loc[0, "log_home_value"] == pytest.approx(math.log(500000))


def test_run_zip_regression_returns_compact_result():
    model_df = pd.DataFrame(
        {
            "zip": [f"75{i:03d}" for i in range(100, 110)],
            "home_value": [
                240000,
                255000,
                262000,
                280000,
                295000,
                315000,
                325000,
                340000,
                360000,
                378000,
            ],
            "violent_rate_per_1000": [9.2, 8.7, 8.1, 7.5, 7.1, 6.8, 6.1, 5.7, 5.2, 4.9],
            "property_rate_per_1000": [22.0, 21.1, 20.4, 19.0, 18.5, 17.8, 16.9, 16.2, 15.1, 14.6],
            "median_household_income": [
                52000,
                54000,
                56000,
                59000,
                61500,
                64000,
                66500,
                69000,
                72000,
                75500,
            ],
            "poverty_rate": [0.22, 0.21, 0.2, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12],
            "owner_occupied_share": [0.31, 0.33, 0.34, 0.36, 0.39, 0.41, 0.43, 0.46, 0.48, 0.5],
            "median_gross_rent": [1150, 1180, 1210, 1250, 1290, 1330, 1375, 1425, 1480, 1540],
        }
    )

    result = run_zip_regression(model_df)

    assert result.nobs == 10
    assert result.formula.startswith("log_home_value ~")
    assert result.r_squared >= 0.0
    assert {"Intercept", "violent_rate_per_1000", "property_rate_per_1000"} <= set(
        result.coefficients["term"]
    )

    metrics = result.metrics_table()
    assert metrics.loc[0, "nobs"] == 10
    assert metrics.loc[0, "dependent_variable"] == "log_home_value"


def test_build_all_creates_processed_outputs(tmp_path: Path):
    settings = Settings.from_env(project_root=tmp_path)
    settings.ensure_directories()

    pd.DataFrame(
        {
            "incident_id": ["a", "b", "c", "d"],
            "reported_at": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "offense": ["Assault", "Burglary", "Theft", "Assault"],
            "offense_family": ["violent", "property", "property", "violent"],
            "zip": ["75201", "75201", "75202", "75202"],
            "latitude": [32.78, 32.78, 32.79, 32.79],
            "longitude": [-96.80, -96.80, -96.79, -96.79],
        }
    ).to_csv(settings.raw_dir / "crime_records.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "home_value": [600000, 420000],
            "as_of_date": ["2026-01-31", "2026-01-31"],
            "source": ["firecrawl", "firecrawl"],
        }
    ).to_csv(settings.raw_dir / "housing_market.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "population": [12000, 24000],
            "median_household_income": [98000, 76000],
            "owner_occupied_share": [0.42, 0.55],
            "poverty_rate": [0.08, 0.13],
            "median_gross_rent": [1900, 1450],
        }
    ).to_csv(settings.raw_dir / "acs_zcta.csv", index=False)

    outputs = build_all(settings)

    for output in outputs.values():
        assert Path(output).exists()
    model_df = pd.read_csv(outputs["model_dataset"])
    assert model_df["zip"].astype(str).tolist() == ["75201", "75202"]
