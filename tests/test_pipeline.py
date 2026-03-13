import json
import math
from pathlib import Path

import pandas as pd
import pytest

from dallas_crime.config import Settings
from dallas_crime.pipeline.analyze import (
    _build_cluster_stability_artifacts,
    _build_forecast_artifacts,
    _build_temporal_holdout_artifacts,
    _prepare_temporal_analysis_inputs,
    _build_vif_artifacts,
    _select_expanded_controls,
    run_zip_regression,
)
from dallas_crime.pipeline.build import (
    aggregate_crime_data,
    build_acs_snapshot_features,
    build_crime_history_features,
    build_housing_history_features,
    build_interaction_features,
    build_source_completeness_scores,
    build_all,
    build_model_dataset,
    normalize_zip,
    normalize_zip_series,
    prepare_crime_history_panel,
    prepare_census_snapshot_panel,
    prepare_census_controls,
    prepare_housing_features,
    prepare_housing_history_panel,
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


def test_build_forecast_artifacts_uses_pre_holdout_model_selection():
    values = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 5.0, 5.0, 5.0, 5.0]
    periods = pd.period_range("2023Q1", periods=len(values), freq="Q")
    crime_history_panel = pd.DataFrame(
        {
            "zip": ["75000"] * len(values),
            "period_start": [period.start_time for period in periods],
            "total_rate_per_1000": values,
        }
    )

    temporal_summary, temporal_series, _ = _prepare_temporal_analysis_inputs(
        crime_history_panel,
        modeled_zips={"75000"},
    )
    forecast_metrics, forecasts, _, _ = _build_forecast_artifacts(temporal_summary, temporal_series)

    assert not forecasts.empty
    assert forecasts["selected_model"].astype(str).nunique() == 1
    assert forecasts["selected_model"].astype(str).iloc[0] == "seasonal_naive_4q"
    seasonal_row = forecast_metrics.loc[
        forecast_metrics["model_name"].astype(str) == "seasonal_naive_4q"
    ].iloc[0]
    moving_average_row = forecast_metrics.loc[
        forecast_metrics["model_name"].astype(str) == "moving_average_4"
    ].iloc[0]
    assert seasonal_row["selected_zip_count"] == 1
    assert moving_average_row["selected_zip_count"] == 0


def test_build_forecast_artifacts_emits_lower_history_tiers_and_restricts_intervals():
    high_confidence_values = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 5.0, 5.0, 5.0, 5.0]
    limited_history_values = [40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 9.0, 8.0, 7.0]
    carry_forward_values = [3.0, 4.0, 5.0]
    crime_history_rows: list[dict[str, object]] = []
    for zip_code, values in (
        ("75000", high_confidence_values),
        ("75001", limited_history_values),
        ("75002", carry_forward_values),
    ):
        periods = pd.period_range("2023Q1", periods=len(values), freq="Q")
        for period, value in zip(periods, values):
            crime_history_rows.append(
                {
                    "zip": zip_code,
                    "period_start": period.start_time,
                    "total_rate_per_1000": value,
                }
            )

    crime_history_panel = pd.DataFrame(crime_history_rows)
    temporal_summary, temporal_series, _ = _prepare_temporal_analysis_inputs(
        crime_history_panel,
        modeled_zips={"75000", "75001", "75002"},
    )
    temporal_holdout, _, _ = _build_temporal_holdout_artifacts(temporal_summary, temporal_series)
    forecast_metrics, forecasts, intervals, _ = _build_forecast_artifacts(
        temporal_summary,
        temporal_series,
        temporal_holdout=temporal_holdout,
    )

    assert not forecasts.empty
    assert set(forecasts["forecast_tier"]) == {
        "high_confidence",
        "limited_history",
        "carry_forward_only",
    }
    assert forecasts.loc[forecasts["zip"].astype(str) == "75000", "selected_model"].astype(str).nunique() == 1
    assert (
        forecasts.loc[forecasts["zip"].astype(str) == "75001", "selected_model"].astype(str).iloc[0]
        == "moving_average_4"
    )
    assert (
        forecasts.loc[forecasts["zip"].astype(str) == "75002", "selected_model"].astype(str).iloc[0]
        == "naive_last"
    )
    assert forecasts.loc[
        forecasts["zip"].astype(str) == "75002", "policy_eligible"
    ].astype(int).eq(0).all()
    assert set(intervals["zip"].astype(str)) == {"75000"}
    assert forecast_metrics["selected_zip_count"].sum() == 1


def test_build_cluster_stability_artifacts_prefers_balanced_k2_solution():
    model_df = pd.DataFrame(
        {
            "zip": [f"7500{i}" for i in range(10)],
            "median_household_income": [45000, 46000, 47000, 48000, 49000, 90000, 91000, 92000, 93000, 94000],
            "poverty_rate": [0.24, 0.23, 0.22, 0.21, 0.20, 0.08, 0.07, 0.06, 0.05, 0.04],
            "owner_occupied_share": [0.28, 0.29, 0.30, 0.31, 0.32, 0.62, 0.63, 0.64, 0.65, 0.66],
            "median_gross_rent": [1100, 1125, 1150, 1175, 1200, 1850, 1875, 1900, 1925, 1950],
        }
    )

    diagnostics = _build_cluster_stability_artifacts(model_df)

    socioeconomic = diagnostics.loc[diagnostics["domain"].astype(str) == "socioeconomic"].iloc[0]
    assert socioeconomic["selected_k"] == 2
    assert socioeconomic["cluster_count"] == 2
    assert socioeconomic["feature_count"] >= 2
    assert "median_household_income" in socioeconomic["selected_feature_set"]
    assert socioeconomic["preprocessing_mode"] in {"raw", "winsor_5_95", "winsor_10_90"}
    assert socioeconomic["size_pass"] == 1
    assert socioeconomic["practical_utility_pass"] == 1


def test_build_cluster_stability_artifacts_uses_winsorized_solution_for_outlier_skew():
    model_df = pd.DataFrame(
        {
            "zip": [f"750{i:02d}" for i in range(12)],
            "total_rate_per_1000": [10, 11, 12, 13, 14, 15, 40, 42, 44, 46, 48, 500],
            "violent_rate_per_1000": [1, 1.1, 1.2, 1.3, 1.4, 1.5, 4.0, 4.2, 4.4, 4.6, 4.8, 50],
            "property_rate_per_1000": [9, 9.9, 10.8, 11.7, 12.6, 13.5, 36, 37.8, 39.6, 41.4, 43.2, 450],
        }
    )

    diagnostics = _build_cluster_stability_artifacts(model_df)

    crime = diagnostics.loc[diagnostics["domain"].astype(str) == "crime"].iloc[0]
    assert crime["selected_k"] == 2
    assert crime["cluster_count"] == 2
    assert crime["preprocessing_mode"] == "winsor_10_90"
    assert crime["size_pass"] == 1
    assert crime["practical_utility_pass"] == 1


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


def test_build_housing_history_features_derives_coverage_and_change_metrics():
    housing_history_df = pd.DataFrame(
        {
            "zip": ["75201", "75201", "75201", "75201", "75202"],
            "period_start": [
                "2020-01-01",
                "2024-01-01",
                "2024-06-01",
                "2025-06-01",
                "2023-01-01",
            ],
            "period_end": [
                "2020-12-31",
                "2024-12-31",
                "2024-06-30",
                "2025-06-30",
                "2023-12-31",
            ],
            "period_year": [2020, 2024, 2024, 2025, 2023],
            "source": [
                "fhfa_zip5",
                "fhfa_zip5",
                "realtor_history",
                "realtor_history",
                "fhfa_zip5",
            ],
            "price_signal_value": [100.0, 108.0, 580000.0, 610000.0, 95.0],
            "price_signal_unit": [
                "index_2000_base",
                "index_2000_base",
                "usd_median_listing_price",
                "usd_median_listing_price",
                "index_2000_base",
            ],
        }
    )

    panel = prepare_housing_history_panel(housing_history_df)
    result = build_housing_history_features(panel)

    assert result["zip"].tolist() == ["75201", "75202"]
    zip_75201 = result.loc[result["zip"] == "75201"].iloc[0]
    assert zip_75201["housing_history_observations_total"] == 4
    assert zip_75201["housing_history_source_count"] == 2
    assert zip_75201["housing_history_first_period_year"] == 2020
    assert zip_75201["housing_history_last_period_year"] == 2025
    assert zip_75201["housing_history_years_covered"] == 6
    assert zip_75201["housing_history_latest_source"] == "realtor_history"
    assert zip_75201["housing_history_latest_price_signal_value"] == pytest.approx(610000.0)
    assert zip_75201["fhfa_history_observations"] == 2
    assert zip_75201["fhfa_history_full_change_pct"] == pytest.approx(8.0)
    assert zip_75201["realtor_history_observations"] == 2
    assert zip_75201["realtor_history_full_change_pct"] == pytest.approx(
        ((610000.0 / 580000.0) - 1.0) * 100.0
    )


def test_build_crime_history_features_derives_quarterly_coverage_and_trend_metrics():
    crime_df = pd.DataFrame(
        {
            "zip": ["75201", "75201", "75201", "75201", "75202"],
            "reported_at": [
                "2024-01-15",
                "2024-03-10",
                "2024-04-12",
                "2024-07-01",
                "2024-02-01",
            ],
            "offense_family": ["violent", "property", "property", "violent", "violent"],
        }
    )
    population_df = pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "population": [1000, 2000],
        }
    )

    panel = prepare_crime_history_panel(crime_df, population_df)
    result = build_crime_history_features(panel)

    assert panel["zip"].tolist() == ["75201", "75201", "75201", "75202"]
    zip_75201 = result.loc[result["zip"] == "75201"].iloc[0]
    assert zip_75201["crime_history_period_count"] == 3
    assert str(pd.to_datetime(zip_75201["crime_history_first_period_start"]).date()) == "2024-01-01"
    assert str(pd.to_datetime(zip_75201["crime_history_last_period_start"]).date()) == "2024-07-01"
    assert zip_75201["crime_history_latest_total_incidents"] == pytest.approx(1.0)
    assert zip_75201["crime_history_latest_total_rate_per_1000"] == pytest.approx(1.0)
    assert zip_75201["crime_history_lag1_total_rate_per_1000"] == pytest.approx(1.0)
    assert zip_75201["crime_history_lag2_total_rate_per_1000"] == pytest.approx(2.0)
    assert pd.isna(zip_75201["crime_history_lag4_total_rate_per_1000"])
    assert zip_75201["crime_history_latest_quarter_number"] == 3
    assert zip_75201["crime_history_latest_is_q3"] == 1
    assert zip_75201["crime_history_q1_mean_rate_per_1000"] == pytest.approx(2.0)
    assert zip_75201["crime_history_q2_mean_rate_per_1000"] == pytest.approx(1.0)
    assert zip_75201["crime_history_q3_mean_rate_per_1000"] == pytest.approx(1.0)
    assert zip_75201["crime_history_mean_total_incidents"] == pytest.approx(4 / 3)
    assert zip_75201["crime_history_mean_total_rate_per_1000"] == pytest.approx(4 / 3)
    assert zip_75201["crime_history_total_rate_change_pct"] == pytest.approx(-50.0)
    assert zip_75201["crime_history_total_rate_trend_per_period"] == pytest.approx(-0.5)
    assert zip_75201["crime_history_latest_vs_lag1_rate_change"] == pytest.approx(0.0)
    assert pd.isna(zip_75201["crime_history_latest_vs_lag4_rate_change"])
    assert zip_75201["crime_history_rate_momentum_2q"] == pytest.approx(-1.0)
    assert zip_75201["crime_history_rate_acceleration"] == pytest.approx(1.0)


def test_build_acs_snapshot_features_derives_latest_change_and_trend_metrics():
    snapshot_df = pd.DataFrame(
        {
            "zip": ["75201", "75201", "75201", "75202"],
            "snapshot_year": [2021, 2022, 2023, 2023],
            "population": [10000, 10200, 10400, 8000],
            "median_household_income": [70000, 73500, 77000, 60000],
            "occupied_housing_units": [4000, 4100, 4200, 3200],
            "owner_occupied_units": [1600, 1681, 1764, 1408],
            "renter_occupied_units": [2400, 2419, 2436, 1792],
            "median_gross_rent": [1500, 1550, 1600, 1350],
            "poverty_universe": [9000, 9200, 9400, 7000],
            "poverty_count": [900, 874, 846, 910],
        }
    )

    panel = prepare_census_snapshot_panel(snapshot_df)
    result = build_acs_snapshot_features(panel)

    assert result["zip"].tolist() == ["75201", "75202"]
    zip_75201 = result.loc[result["zip"] == "75201"].iloc[0]
    assert zip_75201["acs_snapshot_observation_count"] == 3
    assert zip_75201["acs_snapshot_first_year"] == 2021
    assert zip_75201["acs_snapshot_last_year"] == 2023
    assert zip_75201["acs_snapshot_years_covered"] == 3
    assert zip_75201["acs_snapshot_latest_population"] == pytest.approx(10400.0)
    assert zip_75201["acs_snapshot_population_change"] == pytest.approx(400.0)
    assert zip_75201["acs_snapshot_latest_median_household_income"] == pytest.approx(77000.0)
    assert zip_75201["acs_snapshot_median_household_income_change_pct"] == pytest.approx(10.0)
    assert zip_75201["acs_snapshot_median_household_income_trend_per_year"] == pytest.approx(
        3500.0
    )
    assert zip_75201["acs_snapshot_latest_poverty_rate"] == pytest.approx(846 / 9400)
    assert zip_75201["acs_snapshot_poverty_rate_trend_per_year"] == pytest.approx(
        ((846 / 9400) - (900 / 9000)) / 2,
        rel=1e-3,
    )
    assert zip_75201["acs_snapshot_latest_owner_occupied_share"] == pytest.approx(1764 / 4200)
    assert zip_75201["acs_snapshot_latest_housing_tenure_mix"] == pytest.approx(
        (1764 / 4200) - (2436 / 4200)
    )


def test_build_source_completeness_scores_tracks_core_and_optional_categories():
    crime_zip = pd.DataFrame({"zip": ["75201", "75202"], "period_start": ["2025-01-01", "2025-01-01"]})
    crime_history_panel = pd.DataFrame(
        {
            "zip": ["75201", "75201", "75202"],
            "period_start": ["2024-01-01", "2024-04-01", "2024-01-01"],
        }
    )
    housing_zip = pd.DataFrame({"zip": ["75201"], "as_of_date": ["2026-01-31"]})
    housing_history_panel = pd.DataFrame(
        {
            "zip": ["75201", "75201", "75203"],
            "period_start": ["2024-01-01", "2025-01-01", "2025-01-01"],
        }
    )
    controls = pd.DataFrame({"zip": ["75201", "75202", "75203"]})
    census_snapshot_panel = pd.DataFrame(
        {
            "zip": ["75201", "75201", "75202"],
            "snapshot_year": [2022, 2023, 2023],
        }
    )
    sidecar_frames = {"economic": pd.DataFrame({"zip": ["75201", "75203"], "economic_index": [1, 2]})}

    result = build_source_completeness_scores(
        target_zips=["75201", "75202", "75203"],
        crime_zip=crime_zip,
        crime_history_panel=crime_history_panel,
        housing_zip=housing_zip,
        housing_history_panel=housing_history_panel,
        controls=controls,
        census_snapshot_panel=census_snapshot_panel,
        sidecar_frames=sidecar_frames,
    )

    assert {"zip", "category", "expected_periods", "observed_periods", "completeness_ratio"} <= set(
        result.columns
    )
    assert set(result["category"]) == {
        "crime_current",
        "crime_history",
        "housing_current",
        "housing_history",
        "acs_current",
        "acs_snapshots",
        "economic",
    }
    zip_75202_history = result.loc[
        (result["zip"] == "75202") & (result["category"] == "crime_history")
    ].iloc[0]
    assert zip_75202_history["expected_periods"] == 2
    assert zip_75202_history["observed_periods"] == 1
    assert zip_75202_history["completeness_ratio"] == pytest.approx(0.5)
    zip_75203_housing = result.loc[
        (result["zip"] == "75203") & (result["category"] == "housing_current")
    ].iloc[0]
    assert zip_75203_housing["completeness_ratio"] == pytest.approx(0.0)


def test_build_interaction_features_derives_expected_additive_terms():
    model_df = pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "total_rate_per_1000": [10.0, 20.0],
            "median_household_income": [100000.0, 50000.0],
            "rent_burden": [0.25, 0.35],
            "vacancy_proxy": [0.08, 0.12],
            "poverty_rate": [0.07, 0.15],
            "unemployment_rate": [0.06, 0.09],
            "pop_density": [12000.0, 8000.0],
            "crime_history_rate_momentum_2q": [1.5, -0.5],
            "home_value": [500000.0, 250000.0],
            "log_home_value": [math.log(500000.0), math.log(250000.0)],
            "source_completeness_overall_score": [0.95, 0.75],
            "annual_change_pct": [3.0, -1.0],
            "median_rent": [2100.0, 1450.0],
            "acs_snapshot_median_household_income_trend_per_year": [1500.0, -500.0],
        }
    )

    result = build_interaction_features(model_df)

    assert result["zip"].tolist() == ["75201", "75202"]
    zip_75201 = result.loc[result["zip"] == "75201"].iloc[0]
    assert zip_75201["crime_income_interaction"] == pytest.approx(10.0 * math.log1p(100000.0))
    assert zip_75201["crime_rent_burden_interaction"] == pytest.approx(10.0 * 0.25)
    assert zip_75201["crime_poverty_interaction"] == pytest.approx(10.0 * 0.07)
    assert zip_75201["crime_unemployment_interaction"] == pytest.approx(10.0 * 0.06)
    assert zip_75201["crime_density_interaction"] == pytest.approx(10.0 * math.log1p(12000.0))
    assert zip_75201["vacancy_poverty_interaction"] == pytest.approx(0.08 * 0.07)
    assert zip_75201["momentum_home_value_pressure_interaction"] == pytest.approx(
        1.5 * math.log(500000.0)
    )
    assert zip_75201["market_momentum_interaction"] == pytest.approx(3.0 * 1.5)
    assert zip_75201["rent_income_stress_interaction"] == pytest.approx(
        0.25 * (2100.0 / 100000.0)
    )
    assert zip_75201["completeness_weighted_crime_risk"] == pytest.approx(10.0 * 0.95)
    assert pd.notna(zip_75201["aggregate_distress_index"])
    assert pd.notna(zip_75201["aggregate_market_pressure_index"])


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
            "educational_attainment": [0.42, 0.44, 0.45, 0.47, 0.49, 0.52, 0.54, 0.57, 0.60, 0.63],
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
            "offense_family": [
                "violent",
                "property",
                "property",
                "violent",
                "property",
                "property",
            ],
            "zip": ["75201", "75201", "75202", "75203", "75204", "75204"],
            "latitude": [32.78, 32.78, 32.79, 32.80, 32.81, 32.81],
            "longitude": [-96.80, -96.80, -96.79, -96.78, -96.77, -96.77],
        }
    ).to_csv(settings.raw_dir / "crime_records.csv", index=False)
    pd.DataFrame(
        {
            "incident_id": ["ha", "hb", "hc", "hd", "he", "hf", "hg"],
            "reported_at": [
                "2024-01-15",
                "2024-04-10",
                "2025-01-01",
                "2024-02-05",
                "2024-07-12",
                "2024-03-01",
                "2025-01-06",
            ],
            "offense": ["Assault", "Burglary", "Theft", "Assault", "Burglary", "Theft", "Burglary"],
            "offense_family": [
                "violent",
                "property",
                "property",
                "violent",
                "property",
                "property",
                "property",
            ],
            "zip": ["75201", "75201", "75201", "75202", "75203", "75204", "75204"],
            "latitude": [32.78, 32.78, 32.78, 32.79, 32.80, 32.81, 32.81],
            "longitude": [-96.80, -96.80, -96.80, -96.79, -96.78, -96.77, -96.77],
        }
    ).to_csv(settings.raw_dir / "crime_history_records.csv", index=False)
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
            "source_url": [
                "https://example.com/fhfa",
                "https://example.com/realtor",
                "https://example.com/fhfa",
            ],
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
    pd.DataFrame(
        {
            "zip": ["75201", "75201", "75204", "75204"],
            "snapshot_year": [2022, 2023, 2022, 2023],
            "population": [12100, 12300, 15900, 16000],
            "median_household_income": [97000, 99500, 61000, 62500],
            "occupied_housing_units": [5050, 5125, 5980, 6020],
            "owner_occupied_units": [2121, 2204, 2811, 2889],
            "renter_occupied_units": [2929, 2921, 3169, 3131],
            "median_gross_rent": [1880, 1920, 1460, 1490],
            "poverty_universe": [11100, 11250, 14400, 14500],
            "poverty_count": [900, 855, 1512, 1450],
        }
    ).to_csv(settings.raw_dir / "acs_zcta_snapshots.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75204"],
            "economic_index": [1.2, 0.9],
            "median_wage": [78000, 52000],
        }
    ).to_csv(settings.raw_dir / "dfw_zip_economic_sidecar.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75204"],
            "investor_purchase_share": [0.08, 0.14],
        }
    ).to_csv(settings.raw_dir / "dfw_zip_real_estate_sidecar.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75204"],
            "law_staffing_score": [0.93, 0.74],
        }
    ).to_csv(settings.raw_dir / "dfw_zip_law_enforcement_sidecar.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75204"],
            "clinic_access_score": [0.88, 0.67],
        }
    ).to_csv(settings.raw_dir / "dfw_zip_social_services_sidecar.csv", index=False)
    pd.DataFrame(
        {
            "zip": ["75201", "75204"],
            "park_access_score": [0.62, 0.71],
        }
    ).to_csv(settings.raw_dir / "dfw_zip_infrastructure_sidecar.csv", index=False)

    outputs = build_all(settings)

    expected_keys = {
        "crime_zip",
        "crime_history_panel",
        "crime_history_features",
        "housing_zip",
        "acs_controls",
        "acs_snapshot_features",
        "model_dataset",
        "source_completeness_scores",
        "interaction_features",
        "housing_history_panel",
        "housing_history_features",
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
    assert {
        "crime_history_period_count",
        "crime_history_latest_total_rate_per_1000",
        "crime_history_lag1_total_rate_per_1000",
        "crime_history_latest_quarter_number",
        "housing_history_observations_total",
        "housing_history_latest_source",
        "acs_snapshot_observation_count",
        "acs_snapshot_latest_median_household_income",
        "economic_index",
        "median_wage",
        "law_staffing_score",
        "clinic_access_score",
        "park_access_score",
        "source_completeness_overall_score",
        "crime_income_interaction",
        "crime_rent_burden_interaction",
        "crime_poverty_interaction",
        "crime_unemployment_interaction",
        "crime_density_interaction",
        "market_momentum_interaction",
        "rent_income_stress_interaction",
        "aggregate_distress_index",
        "aggregate_market_pressure_index",
    } <= set(model_df.columns)
    model_crime_history_row = model_df.loc[model_df["zip"] == 75201].iloc[0]
    assert model_crime_history_row["crime_history_period_count"] == 3
    assert model_crime_history_row["crime_history_latest_total_rate_per_1000"] == pytest.approx(0.08)
    assert model_crime_history_row["crime_history_lag1_total_rate_per_1000"] == pytest.approx(0.08)
    assert model_crime_history_row["crime_history_latest_quarter_number"] == 1
    model_history_row = model_df.loc[model_df["zip"] == 75201].iloc[0]
    assert model_history_row["housing_history_observations_total"] == 2
    assert model_history_row["housing_history_latest_source"] == "realtor_history"
    assert model_history_row["acs_snapshot_observation_count"] == 2
    assert model_history_row["acs_snapshot_latest_median_household_income"] == pytest.approx(99500.0)
    assert model_history_row["economic_index"] == pytest.approx(1.2)
    assert 0.9 < model_history_row["source_completeness_overall_score"] <= 1.0

    crime_history_df = pd.read_csv(outputs["crime_history_panel"])
    assert {"zip", "period_start", "period_quarter", "total_incidents"} <= set(crime_history_df.columns)
    assert crime_history_df["zip"].astype(str).nunique() == 4
    assert sorted(crime_history_df["zip"].astype(str).unique().tolist()) == [
        "75201",
        "75202",
        "75203",
        "75204",
    ]
    assert len(crime_history_df) == 7

    crime_history_features_df = pd.read_csv(outputs["crime_history_features"])
    assert {
        "zip",
        "crime_history_period_count",
        "crime_history_latest_total_rate_per_1000",
        "crime_history_lag1_total_rate_per_1000",
        "crime_history_latest_quarter_number",
    } <= set(crime_history_features_df.columns)
    assert crime_history_features_df["zip"].astype(str).tolist() == ["75201", "75202", "75203", "75204"]

    acs_snapshot_features_df = pd.read_csv(outputs["acs_snapshot_features"])
    assert {
        "zip",
        "acs_snapshot_observation_count",
        "acs_snapshot_latest_median_household_income",
    } <= set(acs_snapshot_features_df.columns)
    assert acs_snapshot_features_df["zip"].astype(str).tolist() == ["75201", "75204"]

    history_df = pd.read_csv(outputs["housing_history_panel"])
    assert history_df["period_year"].min() == 2000
    assert set(history_df["source"]) == {"fhfa_zip5", "realtor_history"}

    history_features_df = pd.read_csv(outputs["housing_history_features"])
    assert {"zip", "housing_history_observations_total", "housing_history_latest_source"} <= set(
        history_features_df.columns
    )
    assert history_features_df["zip"].astype(str).tolist() == ["75201", "75202"]
    history_feature_row = history_features_df.loc[history_features_df["zip"] == 75201].iloc[0]
    assert history_feature_row["housing_history_observations_total"] == 2
    assert history_feature_row["housing_history_latest_source"] == "realtor_history"

    target_universe = pd.read_csv(outputs["target_zip_universe"])
    assert target_universe["zip"].astype(str).is_unique
    assert {"provenance_tables", "in_target_universe", "in_model_dataset"} <= set(
        target_universe.columns
    )
    assert target_universe["in_target_universe"].sum() == 2
    assert set(
        target_universe.loc[target_universe["in_target_universe"] == 1, "zip"].astype(str)
    ) == {"75201", "75204"}

    completeness_df = pd.read_csv(outputs["source_completeness_scores"])
    assert {
        "zip",
        "category",
        "expected_periods",
        "observed_periods",
        "completeness_ratio",
    } <= set(completeness_df.columns)
    assert {
        "crime_current",
        "crime_history",
        "housing_current",
        "housing_history",
        "acs_current",
        "acs_snapshots",
        "economic",
        "real_estate",
        "law_enforcement",
        "social_services",
        "infrastructure",
    } <= set(completeness_df["category"])

    interaction_df = pd.read_csv(outputs["interaction_features"])
    assert {
        "zip",
        "crime_income_interaction",
        "crime_rent_burden_interaction",
        "crime_poverty_interaction",
        "crime_unemployment_interaction",
        "crime_density_interaction",
        "vacancy_poverty_interaction",
        "momentum_home_value_pressure_interaction",
        "market_momentum_interaction",
        "rent_income_stress_interaction",
        "completeness_weighted_crime_risk",
        "aggregate_distress_index",
        "aggregate_market_pressure_index",
    } <= set(interaction_df.columns)
    assert interaction_df["zip"].astype(str).tolist() == ["75201", "75204"]

    duplicate_report = pd.read_csv(outputs["qa_duplicate_zip_report"])
    assert (
        (duplicate_report["dataset"] == "housing_raw") & (duplicate_report["zip"] == 75201)
    ).any()
    assert not (duplicate_report["dataset"] == "model_dataset").any()

    impossible_report = pd.read_csv(outputs["qa_impossible_values_report"])
    assert {"dataset", "column", "check", "value"} <= set(impossible_report.columns)
    if not impossible_report.empty:
        assert (
            (impossible_report["column"] == "median_household_income")
            & (impossible_report["check"] == "negative_value")
        ).any()

    outlier_report = pd.read_csv(outputs["qa_outlier_markers"])
    assert (
        (outlier_report["dataset"] == "housing_zip")
        & (outlier_report["column"] == "home_value")
    ).any()

    qa_summary = json.loads(Path(outputs["qa_summary"]).read_text())
    assert qa_summary["target_zip_rows_for_modeling"] == 2
    assert qa_summary["duplicate_zip_rows"] >= 1
    assert qa_summary["dataset_row_counts"]["crime_history_panel"] == 7
    assert qa_summary["dataset_row_counts"]["crime_history_features"] == 4
    assert qa_summary["dataset_row_counts"]["acs_snapshot_features"] == 2
    assert qa_summary["dataset_row_counts"]["source_completeness_scores"] == 44
    assert qa_summary["dataset_row_counts"]["interaction_features"] == 2
    assert qa_summary["dataset_row_counts"]["housing_history_panel"] == 3
    assert qa_summary["dataset_row_counts"]["housing_history_features"] == 2
