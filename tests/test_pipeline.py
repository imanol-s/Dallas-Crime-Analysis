import json
import math
from pathlib import Path

import pandas as pd
import pytest

from dallas_crime.config import Settings
from dallas_crime.pipeline.analyze import _build_vif_artifacts, _select_expanded_controls, run_zip_regression
from dallas_crime.pipeline.build import (
    aggregate_crime_data,
    build_all,
    build_model_dataset,
    normalize_zip,
    normalize_zip_series,
    prepare_census_controls,
    prepare_housing_features,
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
            "latitude": [32.780, 32.781, 32.782, 32.790, 32.791, 32.800],
            "longitude": [-96.800, -96.801, -96.802, -96.790, -96.791, -96.780],
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
    assert result.loc[0, "centroid_latitude"] == pytest.approx(32.781)
    assert result.loc[0, "centroid_longitude"] == pytest.approx(-96.801)
    assert str(result.loc[0, "period_start"].date()) == "2025-01-01"
    assert str(result.loc[0, "period_end"].date()) == "2025-01-05"


def test_build_model_dataset_merges_crime_housing_and_controls():
    crime_zip_df = pd.DataFrame(
        {
            "zip": ["75201", "75201", "75202"],
            "total_incidents": [120, 130, 80],
            "total_rate_per_1000": [12.0, 13.0, 8.0],
            "violent_rate_per_1000": [4.0, 5.0, 2.0],
            "property_rate_per_1000": [6.0, 7.0, 5.0],
        }
    )
    housing_df = pd.DataFrame(
        {
            "zip": ["75201-1234", "75201", "75202"],
            "home_value": [500000, 510000, 350000],
            "as_of_date": ["2025-12-01", "2026-01-01", "2026-01-01"],
            "source": ["zillow", "zillow", "zillow"],
            "realtor_active_listing_count": [40, 41, 22],
            "realtor_median_days_on_market": [55.0, 50.0, 43.0],
        }
    )
    controls_df = pd.DataFrame(
        {
            "zip": ["75201", "75201", "75202"],
            "median_household_income": [90000, 90500, 70000],
            "poverty_rate": [0.08, 0.07, 0.12],
            "owner_occupied_share": [0.42, 0.43, 0.55],
            "median_gross_rent": [1800, 1820, 1450],
            "renter_occupied_share": [0.58, 0.57, 0.45],
            "rent_burden": [0.24, 0.23, 0.25],
            "vacancy_proxy": [None, 0.08, 0.1],
            "educational_attainment": [0.6, 0.62, 0.5],
            "housing_tenure_mix": [-0.16, -0.14, 0.1],
        }
    )

    result = build_model_dataset(crime_zip_df, housing_df, controls_df)

    assert result["zip"].is_unique
    assert result["zip"].tolist() == ["75201", "75202"]
    assert result.loc[0, "total_rate_per_1000"] == pytest.approx(13.0)
    assert result.loc[0, "home_value"] == 510000
    assert result.loc[0, "source"] == "zillow"
    assert result.loc[0, "realtor_active_listing_count"] == 41
    assert result.loc[0, "realtor_median_days_on_market"] == pytest.approx(50.0)
    assert result.loc[0, "median_household_income"] == 90500
    assert result.loc[0, "vacancy_proxy"] == pytest.approx(0.08)
    assert result.loc[0, "log_home_value"] == pytest.approx(math.log(510000))


def test_prepare_census_controls_builds_phase3_controls_with_fallbacks():
    census_df = pd.DataFrame(
        {
            "zip": ["75201", "75202", "75261"],
            "population": [10000, 20000, 0],
            "median_household_income": [100000, 50000, -666666666],
            "poverty_rate": [8.0, 0.12, -1],
            "owner_occupied_share": [42.0, 0.55, -1],
            "median_gross_rent": [2000, 1400, -666666666],
            "occupied_housing_units": [4000, 8000, 0],
            "renter_occupied_units": [2320, 3600, 0],
            "total_housing_units": [4500, 9000, 0],
            "bachelors_or_higher_count": [3000, 2000, -1],
            "education_population_25_plus": [5000, 6000, 0],
        }
    )

    result = prepare_census_controls(census_df)

    assert result["zip"].tolist() == ["75201", "75202"]
    assert result.loc[0, "poverty_rate"] == pytest.approx(0.08)
    assert result.loc[0, "owner_occupied_share"] == pytest.approx(0.42)
    assert result.loc[0, "renter_occupied_share"] == pytest.approx(0.58)
    assert result.loc[0, "rent_burden"] == pytest.approx(0.24)
    assert result.loc[0, "vacancy_proxy"] == pytest.approx((4500 - 4000) / 4500)
    assert result.loc[0, "educational_attainment"] == pytest.approx(0.6)
    assert result.loc[0, "housing_tenure_mix"] == pytest.approx(-0.16)


def test_prepare_housing_features_nulls_implausible_annual_change_values():
    housing_df = pd.DataFrame(
        {
            "zip": ["75247", "75210"],
            "home_value": [515000, 180000],
            "annual_change_pct": [296.2, 8.4],
            "as_of_date": ["2026-01-31", "2026-01-31"],
            "source": ["redfin", "zillow"],
        }
    )

    result = prepare_housing_features(housing_df)

    assert pd.isna(result.loc[result["zip"] == "75247", "annual_change_pct"]).all()
    assert result.loc[result["zip"] == "75210", "annual_change_pct"].iloc[0] == pytest.approx(8.4)
    assert "population" not in result.columns


def test_build_model_dataset_requires_valid_acs_controls_membership():
    crime_zip_df = pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "total_incidents": [12, 9],
            "total_rate_per_1000": [12.0, 9.0],
            "violent_rate_per_1000": [4.0, 3.0],
            "property_rate_per_1000": [6.0, 5.0],
        }
    )
    housing_df = pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "home_value": [500000, 350000],
            "as_of_date": ["2026-01-01", "2026-01-01"],
            "source": ["zillow", "zillow"],
        }
    )
    controls_df = pd.DataFrame(
        {
            "zip": ["75201"],
            "population": [10000],
            "median_household_income": [90000],
            "poverty_rate": [0.08],
            "owner_occupied_share": [0.42],
            "median_gross_rent": [1800],
        }
    )

    result = build_model_dataset(crime_zip_df, housing_df, controls_df)

    assert result["zip"].tolist() == ["75201"]


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


def test_select_expanded_controls_prefers_population_acs_without_duplicate_population_signal():
    model_df = pd.DataFrame(
        {
            "zip": [f"75{i:03d}" for i in range(100, 112)],
            "home_value": [250000 + (i * 10000) for i in range(12)],
            "violent_rate_per_1000": [9.5 - (i * 0.3) for i in range(12)],
            "property_rate_per_1000": [21.0 - (i * 0.4) for i in range(12)],
            "median_household_income": [55000 + (i * 2500) for i in range(12)],
            "poverty_rate": [0.20 - (i * 0.006) for i in range(12)],
            "owner_occupied_share": [0.32 + (i * 0.015) for i in range(12)],
            "median_gross_rent": [1200 + (i * 40) for i in range(12)],
            "population": [12000 + (i * 600) for i in range(12)],
            "population_acs": [12000 + (i * 600) for i in range(12)],
            "median_rent": [1180 + (i * 35) for i in range(12)],
            "annual_change_pct": [2.1 + (i * 0.1) for i in range(12)],
        }
    )

    controls = _select_expanded_controls(
        model_df,
        dependent="log_home_value",
        predictors=("violent_rate_per_1000", "property_rate_per_1000"),
        baseline_controls=(
            "median_household_income",
            "poverty_rate",
            "owner_occupied_share",
            "median_gross_rent",
        ),
    )

    assert "population_acs" in controls
    assert "population" not in controls


def test_build_vif_artifacts_notes_infinite_vif_terms():
    model_df = pd.DataFrame(
        {
            "zip": [f"75{i:03d}" for i in range(100, 112)],
            "home_value": [260000 + (i * 12000) for i in range(12)],
            "violent_rate_per_1000": [8.8 - (i * 0.25) for i in range(12)],
            "property_rate_per_1000": [19.5 - (i * 0.35) for i in range(12)],
            "median_household_income": [58000 + (i * 2800) for i in range(12)],
            "poverty_rate": [0.19 - (i * 0.005) for i in range(12)],
            "owner_occupied_share": [0.34 + (i * 0.012) for i in range(12)],
            "median_gross_rent": [1250 + (i * 45) for i in range(12)],
            "population_acs": [14000 + (i * 500) for i in range(12)],
            "population_clone": [14000 + (i * 500) for i in range(12)],
        }
    )

    result = run_zip_regression(
        model_df,
        controls=(
            "median_household_income",
            "poverty_rate",
            "owner_occupied_share",
            "median_gross_rent",
            "population_acs",
            "population_clone",
        ),
        model_label="collinear_model",
    )

    vif_table, notes = _build_vif_artifacts([result])

    assert not vif_table.empty
    combined_notes = "\n".join(notes)
    assert "infinite VIF" in combined_notes
    assert "population_acs" in combined_notes
    assert "population_clone" in combined_notes


def test_build_all_creates_processed_outputs(tmp_path: Path):
    settings = Settings.from_env(project_root=tmp_path)
    settings.ensure_directories()

    pd.DataFrame(
        {
            "incident_id": ["a", "b", "c", "d", "e", "f"],
            "reported_at": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
            ],
            "offense": ["Assault", "Burglary", "Theft", "Assault", "Theft", "Burglary"],
            "offense_family": ["violent", "property", "property", "violent", "property", "property"],
            "zip": ["75201", "75201", "75202", "75203", "75204", "75204"],
            "latitude": [32.78, 32.78, 32.79, 32.80, 32.81, 32.81],
            "longitude": [-96.80, -96.80, -96.79, -96.78, -96.77, -96.77],
        }
    ).to_csv(settings.raw_dir / "crime_records.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75201", "75202", "75203", "75204"],
            "home_value": [580000, 600000, 420000, 450000, 3000000],
            "median_rent": [1800, 1820, 1450, 1500, 2500],
            "annual_change_pct": [3.1, 3.5, 2.8, 2.6, 7.0],
            "as_of_date": ["2025-12-31", "2026-01-31", "2026-01-31", "2026-01-31", "2026-01-31"],
            "source": ["firecrawl", "firecrawl", "firecrawl", "firecrawl", "firecrawl"],
        }
    ).to_csv(settings.raw_dir / "housing_market.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75201", "75202"],
            "period_start": ["2000-01-01", "2025-01-01", "2001-01-01"],
            "period_end": ["2000-12-31", "2025-01-31", "2001-12-31"],
            "period_year": [2000, 2025, 2001],
            "period_month": [None, 1, None],
            "frequency": ["annual", "monthly", "annual"],
            "source": ["fhfa_zip5", "realtor_history", "fhfa_zip5"],
            "source_url": ["https://example.com/fhfa", "https://example.com/realtor", "https://example.com/fhfa"],
            "metric_label": ["fhfa_hpi", "median_listing_price", "fhfa_hpi"],
            "price_signal_value": [100.0, 610000.0, 103.0],
            "price_signal_unit": ["index_2000_base", "usd_median_listing_price", "index_2000_base"],
        }
    ).to_csv(settings.raw_dir / "housing_market_history.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75201", "75202", "75203", "75204"],
            "population": [12000, 12500, 24000, 18000, 16000],
            "median_household_income": [98000, 99000, 76000, 68000, -500],
            "owner_occupied_share": [0.42, 0.43, 0.55, 0.5, 0.47],
            "poverty_rate": [0.08, 0.07, 0.13, 0.16, 0.1],
            "median_gross_rent": [1900, 1920, 1450, 1380, 1500],
            "occupied_housing_units": [5000, 5100, 9000, 7000, 6000],
            "renter_occupied_units": [2900, 2850, 4050, 3500, 3180],
            "total_housing_units": [5600, 5700, 9700, 7600, 7100],
            "bachelors_or_higher_count": [3200, 3300, 4100, 2500, 2300],
            "education_population_25_plus": [5200, 5300, 7800, 5200, 4900],
        }
    ).to_csv(settings.raw_dir / "acs_zcta.csv", index=False)

    outputs = build_all(settings)

    expected_keys = {
        "crime_zip",
        "housing_zip",
        "acs_controls",
        "model_dataset",
        "housing_history_panel",
        "target_zip_universe",
        "qa_duplicate_zip_report",
        "qa_missingness_report",
        "qa_impossible_values_report",
        "qa_outlier_markers",
        "qa_summary",
    }
    assert expected_keys <= set(outputs)
    for key in expected_keys:
        assert Path(outputs[key]).exists()

    model_df = pd.read_csv(outputs["model_dataset"])
    assert model_df["zip"].astype(str).tolist() == ["75201", "75204"]
    assert model_df["zip"].is_unique
    assert {"centroid_latitude", "centroid_longitude"} <= set(model_df.columns)

    history_df = pd.read_csv(outputs["housing_history_panel"])
    assert history_df["period_year"].min() == 2000
    assert set(history_df["source"]) == {"fhfa_zip5", "realtor_history"}

    target_universe = pd.read_csv(outputs["target_zip_universe"])
    assert target_universe["zip"].astype(str).is_unique
    assert {"provenance_tables", "in_target_universe", "in_model_dataset"} <= set(
        target_universe.columns
    )
    assert target_universe["in_target_universe"].sum() == 2
    assert set(target_universe.loc[target_universe["in_target_universe"] == 1, "zip"].astype(str)) == {
        "75201",
        "75204",
    }

    duplicate_report = pd.read_csv(outputs["qa_duplicate_zip_report"])
    assert ((duplicate_report["dataset"] == "housing_raw") & (duplicate_report["zip"] == 75201)).any()
    assert not (duplicate_report["dataset"] == "model_dataset").any()

    impossible_report = pd.read_csv(outputs["qa_impossible_values_report"])
    assert {"dataset", "column", "check", "value"} <= set(impossible_report.columns)
    if not impossible_report.empty:
        assert (
            (impossible_report["column"] == "median_household_income")
            & (impossible_report["check"] == "negative_value")
        ).any()

    outlier_report = pd.read_csv(outputs["qa_outlier_markers"])
    assert ((outlier_report["dataset"] == "housing_zip") & (outlier_report["column"] == "home_value")).any()

    qa_summary = json.loads(Path(outputs["qa_summary"]).read_text())
    assert qa_summary["target_zip_rows_for_modeling"] == 2
    assert qa_summary["duplicate_zip_rows"] >= 1
    assert qa_summary["dataset_row_counts"]["housing_history_panel"] == 3
