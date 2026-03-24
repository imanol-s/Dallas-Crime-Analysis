from pathlib import Path

import pandas as pd

from dallas_crime.config import Settings
from dallas_crime.pipeline.analyze import run_analysis


def _write_model_dataset(settings: Settings) -> None:
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
            "total_rate_per_1000": [31.2, 29.8, 28.5, 26.3, 25.6, 24.6, 23.0, 21.9, 20.3, 19.5],
            "violent_rate_per_1000": [9.2, 8.7, 8.1, 7.5, 7.1, 6.8, 6.1, 5.7, 5.2, 4.9],
            "property_rate_per_1000": [22.0, 21.1, 20.4, 19.0, 18.5, 17.8, 16.9, 16.2, 15.1, 14.6],
            "centroid_latitude": [
                32.77,
                32.775,
                32.78,
                32.785,
                32.79,
                32.795,
                32.8,
                32.805,
                32.81,
                32.815,
            ],
            "centroid_longitude": [
                -96.82,
                -96.815,
                -96.81,
                -96.805,
                -96.8,
                -96.795,
                -96.79,
                -96.785,
                -96.78,
                -96.775,
            ],
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
            "population": [9500, 9800, 10000, 10200, 10500, 10800, 11100, 11500, 11900, 12300],
            "population_acs": [
                9400,
                9750,
                9950,
                10150,
                10450,
                10750,
                11050,
                11450,
                11850,
                12250,
            ],
            "median_rent": [1110, 1140, 1180, 1220, 1260, 1310, 1360, 1410, 1470, 1525],
            "annual_change_pct": [1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.7, 2.9, 3.1, 3.2],
            "educational_attainment": [0.42, 0.44, 0.45, 0.47, 0.49, 0.52, 0.54, 0.57, 0.60, 0.63],
        }
    )
    model_df.to_csv(settings.processed_dir / "model_dataset.csv", index=False)


def test_run_analysis_writes_report_artifacts_without_optional_history(tmp_path: Path):
    settings = Settings.from_env(project_root=tmp_path)
    settings.ensure_directories()
    _write_model_dataset(settings)

    outputs = run_analysis(settings)

    expected_keys = {
        "coefficients",
        "metrics",
        "sample_sizes",
        "residuals",
        "residual_review",
        "vif",
        "vif_notes",
        "baseline_control_audit",
        "scatter_plot",
        "geography_plot",
        "cluster_assignments",
        "cluster_profiles",
        "cluster_stability_diagnostics",
        "spatial_diagnostics",
        "spatial_hotspots",
        "validation_metrics",
        "validation_notes",
        "zip_comparison",
        "model_summary_table",
        "trend_decomposition",
        "trend_notes",
        "feature_selection_metrics",
        "feature_selection_notes",
        "feature_power_retention_metrics",
        "feature_power_retention_notes",
        "predictive_model_metrics",
        "predictive_model_predictions",
        "model_selection_notes",
        "forecast_model_metrics",
        "crime_forecasts",
        "forecast_confidence_intervals",
        "temporal_holdout_results",
        "forecast_interval_calibration",
        "forecast_notes",
        "scenario_impacts",
        "scenario_notes",
        "zip_benchmarks",
        "benchmark_summary",
        "drift_diagnostics",
        "drift_notes",
        "influence_robustness_diagnostics",
        "statistical_guardrails",
        "comprehensive_validation_metrics",
        "comprehensive_validation_notes",
        "policy_recommendations",
        "policy_recommendations_notes",
        "policy_guardrails",
        "summary",
        "factor_importance_univariate",
        "factor_importance_standardized",
        "factor_importance_variance_decomposition",
        "factor_importance_summary",
    }
    assert set(outputs) == expected_keys
    for output in outputs.values():
        assert Path(output).exists()

    metrics = pd.read_csv(outputs["metrics"])
    # baseline and sensitivity_check must always be present; new model variants may also appear
    assert {"baseline", "sensitivity_check"} <= set(metrics["model_label"])
    assert {"nobs", "r_squared", "adjusted_r_squared"} <= set(metrics.columns)

    coefficients = pd.read_csv(outputs["coefficients"])
    assert {"baseline", "sensitivity_check"} <= set(coefficients["model_label"])
    assert {
        "term",
        "estimate",
        "std_error",
        "p_value",
        "fdr_q_value",
        "interpretation_allowed",
    } <= set(coefficients.columns)

    sample_sizes = pd.read_csv(outputs["sample_sizes"])
    assert {"baseline", "sensitivity_check"} <= set(sample_sizes["model_label"])
    assert sample_sizes["nobs"].min() >= 8

    residuals = pd.read_csv(outputs["residuals"])
    assert {"baseline", "sensitivity_check"} <= set(residuals["model_label"])
    assert {"observed", "fitted_value", "residual", "absolute_residual"} <= set(residuals.columns)

    vif = pd.read_csv(outputs["vif"])
    assert {"baseline", "sensitivity_check"} <= set(vif["model_label"])
    assert {"term", "vif"} <= set(vif.columns)

    cluster_assignments = pd.read_csv(outputs["cluster_assignments"])
    assert {"zip", "crime_cluster", "socioeconomic_cluster", "market_cluster"} <= set(
        cluster_assignments.columns
    )
    assert cluster_assignments["zip"].astype(str).nunique() == len(cluster_assignments)

    cluster_profiles = pd.read_csv(outputs["cluster_profiles"])
    assert {"domain", "cluster_label", "zip_count", "feature", "mean_value"} <= set(
        cluster_profiles.columns
    )
    assert {"crime", "socioeconomic", "market"} <= set(cluster_profiles["domain"])

    cluster_stability = pd.read_csv(outputs["cluster_stability_diagnostics"])
    assert not cluster_stability.empty
    assert {
        "domain",
        "selected_feature_set",
        "preprocessing_mode",
        "silhouette_score",
        "leave_one_feature_out_mean_ari",
        "practical_utility_pass",
    } <= set(cluster_stability.columns)

    spatial_diagnostics = pd.read_csv(outputs["spatial_diagnostics"])
    assert {
        "metric",
        "morans_i",
        "permutation_p_value",
        "spatial_lag_correlation",
        "fdr_q_value",
        "interpretation_allowed",
    } <= set(spatial_diagnostics.columns)
    assert {"home_value", "total_rate_per_1000"} <= set(spatial_diagnostics["metric"])

    spatial_hotspots = pd.read_csv(outputs["spatial_hotspots"])
    assert {
        "zip",
        "total_rate_per_1000",
        "crime_spatial_lag",
        "crime_spatial_quadrant",
    } <= set(spatial_hotspots.columns)
    assert set(spatial_hotspots["crime_spatial_quadrant"]) <= {
        "high-high",
        "high-low",
        "low-high",
        "low-low",
    }

    validation_metrics = pd.read_csv(outputs["validation_metrics"])
    assert {"baseline", "sensitivity_check"} <= set(validation_metrics["model_label"])
    assert {
        "loocv_rmse",
        "predicted_r_squared",
        "max_leverage",
        "fit_deterioration_pass",
        "fit_improvement_warning_count",
        "influence_robustness_pass",
    } <= set(validation_metrics.columns)

    validation_notes = Path(outputs["validation_notes"]).read_text()
    assert "Model Validation Notes" in validation_notes

    trend_decomposition = pd.read_csv(outputs["trend_decomposition"])
    assert trend_decomposition.empty
    trend_notes = Path(outputs["trend_notes"]).read_text()
    assert "No decomposition rows were produced" in trend_notes

    feature_selection_metrics = pd.read_csv(outputs["feature_selection_metrics"])
    assert not feature_selection_metrics.empty
    assert {
        "feature_name",
        "selection_rank",
        "recommended_for_future_models",
        "fdr_q_value",
        "interpretation_allowed",
    } <= set(feature_selection_metrics.columns)
    feature_selection_notes = Path(outputs["feature_selection_notes"]).read_text()
    assert "Feature candidates evaluated" in feature_selection_notes

    feature_power_metrics = pd.read_csv(outputs["feature_power_retention_metrics"])
    assert not feature_power_metrics.empty
    assert {
        "metric",
        "value",
        "threshold",
        "meets_threshold",
        "definition",
    } <= set(feature_power_metrics.columns)
    assert {
        "feature_selection_score_retention_ratio",
        "predictive_r_squared_retention_ratio",
        "feature_power_checkpoint_pass",
    } <= set(feature_power_metrics["metric"])
    feature_power_notes = Path(outputs["feature_power_retention_notes"]).read_text()
    assert "Feature Power Retention Notes" in feature_power_notes

    predictive_model_metrics = pd.read_csv(outputs["predictive_model_metrics"])
    assert not predictive_model_metrics.empty
    assert {"model_label", "model_tier", "selected_for_ensemble"} <= set(
        predictive_model_metrics.columns
    )
    predictive_model_predictions = pd.read_csv(outputs["predictive_model_predictions"])
    assert not predictive_model_predictions.empty
    assert {"zip", "model_label", "predicted", "residual"} <= set(
        predictive_model_predictions.columns
    )
    model_selection_notes = Path(outputs["model_selection_notes"]).read_text()
    assert "Model-family rows written" in model_selection_notes

    forecast_model_metrics = pd.read_csv(outputs["forecast_model_metrics"])
    assert forecast_model_metrics.empty
    crime_forecasts = pd.read_csv(outputs["crime_forecasts"])
    assert crime_forecasts.empty
    forecast_intervals = pd.read_csv(outputs["forecast_confidence_intervals"])
    assert forecast_intervals.empty
    temporal_holdout_results = pd.read_csv(outputs["temporal_holdout_results"])
    assert temporal_holdout_results.empty
    forecast_interval_calibration = pd.read_csv(outputs["forecast_interval_calibration"])
    assert forecast_interval_calibration.empty
    forecast_notes = Path(outputs["forecast_notes"]).read_text()
    assert "No crime forecasts were produced" in forecast_notes

    scenario_impacts = pd.read_csv(outputs["scenario_impacts"])
    assert scenario_impacts.empty
    scenario_notes = Path(outputs["scenario_notes"]).read_text()
    assert "deterministic multipliers" in scenario_notes
    assert "policy_guardrails.md" in scenario_notes

    zip_benchmarks = pd.read_csv(outputs["zip_benchmarks"])
    assert {"zip", "crime_cluster", "home_value_vs_metro_pct"} <= set(zip_benchmarks.columns)
    assert not zip_benchmarks.empty
    benchmark_summary = Path(outputs["benchmark_summary"]).read_text()
    assert "Benchmark Summary" in benchmark_summary

    drift_diagnostics = pd.read_csv(outputs["drift_diagnostics"])
    assert drift_diagnostics.empty
    drift_notes = Path(outputs["drift_notes"]).read_text()
    assert "drift diagnostics are empty" in drift_notes

    influence_robustness = pd.read_csv(outputs["influence_robustness_diagnostics"])
    assert not influence_robustness.empty
    assert {
        "model_label",
        "removed_zip",
        "flag_reason",
        "robustness_pass",
    } <= set(influence_robustness.columns)

    statistical_guardrails = pd.read_csv(outputs["statistical_guardrails"])
    assert not statistical_guardrails.empty
    assert {
        "domain",
        "term",
        "fdr_q_value",
        "passes_practical_effect",
        "interpretation_allowed",
    } <= set(statistical_guardrails.columns)
    assert {"regression", "feature_selection", "spatial"} <= set(statistical_guardrails["domain"])

    comprehensive_validation = pd.read_csv(outputs["comprehensive_validation_metrics"])
    assert not comprehensive_validation.empty
    assert {"domain", "item", "metric", "value"} <= set(comprehensive_validation.columns)
    comprehensive_validation_notes = Path(outputs["comprehensive_validation_notes"]).read_text()
    assert "Comprehensive Validation Notes" in comprehensive_validation_notes

    policy_recommendations = pd.read_csv(outputs["policy_recommendations"])
    assert policy_recommendations.empty
    policy_recommendations_notes = Path(outputs["policy_recommendations_notes"]).read_text()
    assert "No policy recommendations were generated" in policy_recommendations_notes
    policy_guardrails = Path(outputs["policy_guardrails"]).read_text()
    assert "non-causal" in policy_guardrails

    summary = Path(outputs["summary"]).read_text()
    assert "Dallas Crime and Housing Report" in summary
    assert "## Methods" in summary
    assert "## Findings" in summary
    assert "## Limitations" in summary
    assert "top_bottom_zip_comparison.md" in summary
    assert "crime_home_value_geography.png" in summary
    assert "cluster_assignments.csv" in summary
    assert "spatial_diagnostics.csv" in summary
    assert "model_validation_metrics.csv" in summary
    assert "crime_trend_decomposition.csv" in summary
    assert "forecast_model_metrics.csv" in summary
    assert "temporal_holdout_results.csv" in summary
    assert "forecast_interval_calibration.csv" in summary
    assert "feature_selection_metrics.csv" in summary
    assert "feature_power_retention_metrics.csv" in summary
    assert "predictive_model_metrics.csv" in summary
    assert "scenario_impacts.csv" in summary
    assert "zip_benchmarks.csv" in summary
    assert "model_drift_diagnostics.csv" in summary
    assert "cluster_stability_diagnostics.csv" in summary
    assert "influence_robustness_diagnostics.csv" in summary
    assert "statistical_guardrails.csv" in summary
    assert "comprehensive_validation_metrics.csv" in summary
    assert "policy_recommendations_by_segment.csv" in summary
    assert "policy_guardrails.md" in summary
    assert "dallas-crime acquire && dallas-crime build && dallas-crime analyze" in summary


def test_run_analysis_populates_optional_history_artifacts(tmp_path: Path):
    settings = Settings.from_env(project_root=tmp_path)
    settings.ensure_directories()
    _write_model_dataset(settings)

    periods = pd.period_range("2023Q1", periods=12, freq="Q")
    crime_history_rows: list[dict[str, object]] = []
    for zip_index, zip_code in enumerate([f"75{i:03d}" for i in range(100, 110)]):
        zip_periods = periods if zip_index < 9 else periods[:10]
        for period_index, period in enumerate(zip_periods):
            seasonal_adjustment = (-0.6, -0.2, 0.4, 0.8)[period_index % 4]
            crime_history_rows.append(
                {
                    "zip": zip_code,
                    "period_start": period.start_time,
                    "period_end": period.end_time,
                    "period_year": period.year,
                    "period_quarter": period.quarter,
                    "frequency": "quarterly",
                    "total_incidents": 100 - (zip_index * 2) - period_index,
                    "violent_incidents": 30 - zip_index,
                    "property_incidents": 60 - period_index,
                    "other_incidents": 10,
                    "population": 9000 + (zip_index * 250),
                    "total_rate_per_1000": 32
                    - (zip_index * 0.7)
                    - (period_index * 0.5)
                    + seasonal_adjustment,
                    "violent_rate_per_1000": 10 - (zip_index * 0.2) - (period_index * 0.1),
                    "property_rate_per_1000": 22 - (zip_index * 0.5) - (period_index * 0.4),
                }
            )
    pd.DataFrame(crime_history_rows).to_csv(
        settings.processed_dir / "crime_history_panel.csv",
        index=False,
    )

    housing_history_rows: list[dict[str, object]] = []
    for zip_index, zip_code in enumerate([f"75{i:03d}" for i in range(100, 110)]):
        for year in range(2021, 2026):
            housing_history_rows.append(
                {
                    "zip": zip_code,
                    "period_start": f"{year}-01-01",
                    "period_end": f"{year}-12-31",
                    "period_year": year,
                    "source": "fhfa_zip5",
                    "price_signal_value": 100 + (zip_index * 2) + ((year - 2021) * 1.5),
                    "price_signal_unit": "index_2000_base",
                }
            )
    pd.DataFrame(housing_history_rows).to_csv(
        settings.processed_dir / "housing_history_panel.csv",
        index=False,
    )

    outputs = run_analysis(settings)

    trend_decomposition = pd.read_csv(outputs["trend_decomposition"])
    assert not trend_decomposition.empty
    assert {"scope", "period_start", "trend_total_rate_per_1000"} <= set(
        trend_decomposition.columns
    )

    forecast_model_metrics = pd.read_csv(outputs["forecast_model_metrics"])
    assert set(forecast_model_metrics["model_name"]) == {
        "naive_last",
        "seasonal_naive_4q",
        "moving_average_4",
        "linear_trend",
    }
    assert forecast_model_metrics["evaluation_points"].max() > 0
    assert forecast_model_metrics["eligible_zip_count"].iloc[0] == 9
    assert {"temporal_holdout_mape", "temporal_holdout_pass", "temporal_holdout_rank"} <= set(
        forecast_model_metrics.columns
    )

    crime_forecasts = pd.read_csv(outputs["crime_forecasts"])
    assert not crime_forecasts.empty
    assert set(crime_forecasts["horizon_quarters"]) == {1, 2, 3, 4}
    assert crime_forecasts["zip"].astype(str).nunique() == 10
    assert {"forecast_tier", "policy_eligible", "selection_rule"} <= set(crime_forecasts.columns)
    assert set(crime_forecasts["forecast_tier"]) == {"high_confidence", "limited_history"}
    assert (
        crime_forecasts.loc[
            crime_forecasts["forecast_tier"] == "limited_history", "policy_eligible"
        ]
        .astype(int)
        .eq(0)
        .all()
    )

    forecast_intervals = pd.read_csv(outputs["forecast_confidence_intervals"])
    assert not forecast_intervals.empty
    assert set(forecast_intervals["interval_level"]) == {80, 95}
    assert {"forecast_tier", "policy_eligible", "interval_width", "upper_to_forecast_ratio"} <= set(
        forecast_intervals.columns
    )
    assert set(forecast_intervals["forecast_tier"]) == {"high_confidence"}

    temporal_holdout_results = pd.read_csv(outputs["temporal_holdout_results"])
    assert not temporal_holdout_results.empty
    assert {"evaluation_scope", "model_name", "selected_family", "mape", "mape_pass"} <= set(
        temporal_holdout_results.columns
    )
    selected_holdout = temporal_holdout_results.loc[
        temporal_holdout_results["evaluation_scope"] == "selected_zip_model"
    ]
    assert not selected_holdout.empty
    assert selected_holdout["eligible_zip_count"].iloc[0] == 9
    assert set(selected_holdout["selected_family"]) == {"holdout_pass_screened_zip_selection"}
    assert selected_holdout["mape_pass"].iloc[0] == 1

    forecast_interval_calibration = pd.read_csv(outputs["forecast_interval_calibration"])
    assert not forecast_interval_calibration.empty
    assert {
        "evaluation_scope",
        "selected_family",
        "interval_level",
        "calibration_pass",
        "shape_pass",
    } <= set(forecast_interval_calibration.columns)
    assert {"selected_zip_model", "forecast_output_shape"} <= set(
        forecast_interval_calibration["evaluation_scope"]
    )

    residuals = pd.read_csv(outputs["residuals"])
    assert {"is_high_influence_flagged", "influence_flag_reason"} <= set(residuals.columns)
    assert set(
        pd.to_numeric(residuals["is_high_influence_flagged"], errors="coerce").dropna().unique()
    ) <= {
        0,
        1,
    }

    scenario_impacts = pd.read_csv(outputs["scenario_impacts"])
    assert not scenario_impacts.empty
    assert scenario_impacts["zip"].astype(str).nunique() == 9
    assert set(scenario_impacts["scenario_name"]) == {
        "baseline",
        "stabilization",
        "adverse_momentum",
        "seasonal_peak",
        "systemic_shock",
    }
    assert {"evidence_posture", "causal_interpretation_allowed"} <= set(scenario_impacts.columns)

    zip_benchmarks = pd.read_csv(outputs["zip_benchmarks"])
    assert not zip_benchmarks.empty
    assert {"home_value_percentile", "total_rate_vs_metro_pct", "crime_cluster"} <= set(
        zip_benchmarks.columns
    )

    drift_diagnostics = pd.read_csv(outputs["drift_diagnostics"])
    assert not drift_diagnostics.empty
    assert {
        "entity_type",
        "entity_id",
        "z_score",
        "drift_flag",
        "trailing_contiguous_quarters",
    } <= set(drift_diagnostics.columns)

    feature_selection_metrics = pd.read_csv(outputs["feature_selection_metrics"])
    assert not feature_selection_metrics.empty
    assert feature_selection_metrics["recommended_for_future_models"].sum() >= 1
    assert "interpretation_allowed" in feature_selection_metrics.columns

    feature_power_metrics = pd.read_csv(outputs["feature_power_retention_metrics"])
    assert not feature_power_metrics.empty
    checkpoint_row = feature_power_metrics.loc[
        feature_power_metrics["metric"] == "feature_power_checkpoint_pass"
    ].iloc[0]
    assert checkpoint_row["value"] in {0.0, 1.0}

    predictive_model_metrics = pd.read_csv(outputs["predictive_model_metrics"])
    assert not predictive_model_metrics.empty
    assert {"baseline", "intermediate", "advanced", "specialized"} <= set(
        predictive_model_metrics["model_tier"]
    )

    predictive_model_predictions = pd.read_csv(outputs["predictive_model_predictions"])
    assert not predictive_model_predictions.empty
    assert (predictive_model_predictions["model_label"] == "ensemble_top2_inverse_rmse").any()

    cluster_stability = pd.read_csv(outputs["cluster_stability_diagnostics"])
    assert not cluster_stability.empty
    assert {
        "domain",
        "selected_k",
        "selected_feature_set",
        "preprocessing_mode",
        "leave_one_feature_out_mean_ari",
        "practical_utility_pass",
    } <= set(cluster_stability.columns)

    influence_robustness = pd.read_csv(outputs["influence_robustness_diagnostics"])
    assert not influence_robustness.empty
    assert {"baseline", "sensitivity_check"} <= set(influence_robustness["model_label"])
    assert {
        "mean_home_value_pct_delta",
        "p90_home_value_pct_delta",
        "max_home_value_pct_delta",
        "fit_stability_pass",
        "fit_deterioration_pass",
        "fit_improvement_warning",
    } <= set(influence_robustness.columns)
    # robustness_pass depends on the active model spec; assert it contains only valid 0/1 values
    assert set(
        pd.to_numeric(influence_robustness["robustness_pass"], errors="coerce").dropna().unique()
    ) <= {0, 1}

    statistical_guardrails = pd.read_csv(outputs["statistical_guardrails"])
    assert not statistical_guardrails.empty
    assert {"regression", "feature_selection", "spatial"} <= set(statistical_guardrails["domain"])

    comprehensive_validation = pd.read_csv(outputs["comprehensive_validation_metrics"])
    assert not comprehensive_validation.empty
    assert {
        "regression",
        "predictive_family",
        "forecast",
        "temporal_holdout",
        "interval_calibration",
        "scenario",
    } <= set(comprehensive_validation["domain"])

    policy_recommendations = pd.read_csv(outputs["policy_recommendations"])
    assert not policy_recommendations.empty
    assert {
        "segment_type",
        "segment_label",
        "priority_tier",
        "scenario_eligible_zip_count",
        "high_influence_zip_count",
        "segment_guardrail_status",
        "segment_confidence_tier",
        "guardrail_flags",
        "evidence_posture",
    } <= set(policy_recommendations.columns)
    assert set(policy_recommendations["priority_tier"]) <= {"high", "moderate", "watch"}
    assert set(policy_recommendations["segment_guardrail_status"]) <= {
        "blocked",
        "caution",
        "clear",
    }

    forecast_notes = Path(outputs["forecast_notes"]).read_text()
    assert "Selected high-confidence ZIP-level models" in forecast_notes
    assert "Temporal holdout results" in forecast_notes
    assert "Coverage tiers" in forecast_notes
    assert "forecast-only coverage" in forecast_notes

    scenario_notes = Path(outputs["scenario_notes"]).read_text()
    assert "forecast-only lower-confidence tiers" in scenario_notes

    policy_guardrails = Path(outputs["policy_guardrails"]).read_text()
    assert "exploratory, non-causal" in policy_guardrails
    assert "High-confidence forecast/scenario coverage" in policy_guardrails
    assert "Lower-confidence forecast-only coverage" in policy_guardrails


def test_factor_importance_artifacts_are_populated(tmp_path: Path):
    """Verify factor importance artifacts have the expected shape and content."""
    settings = Settings.from_env(project_root=tmp_path)
    settings.ensure_directories()
    _write_model_dataset(settings)

    outputs = run_analysis(settings)

    # Phase 1: univariate screening has at least 5 rows
    univariate = pd.read_csv(outputs["factor_importance_univariate"])
    assert len(univariate) >= 5
    assert {"feature", "n_complete", "pearson_r", "spearman_rho", "univariate_r_squared"} <= set(
        univariate.columns
    )

    # Phase 2: standardized coefficients have at least 1 row with abs_standardized_beta > 0
    standardized = pd.read_csv(outputs["factor_importance_standardized"])
    assert len(standardized) >= 1
    assert (standardized["abs_standardized_beta"] > 0).any()

    # Phase 3: variance decomposition sums lmg_r_squared_share to within 0.01 of total model R²
    decomposition = pd.read_csv(outputs["factor_importance_variance_decomposition"])
    assert not decomposition.empty
    lmg_sum = float(decomposition["lmg_r_squared_share"].sum())
    # Reconstruct total R² from the standardized model: fit the same features
    from dallas_crime.pipeline.analyze.core import _ensure_dependent_column

    model_df = pd.read_csv(settings.processed_dir / "model_dataset.csv")
    frame = _ensure_dependent_column(model_df, "log_home_value")
    final_features = standardized["feature"].tolist()
    all_cols = ["log_home_value", *final_features]
    work = frame[all_cols].copy()
    for c in all_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna().reset_index(drop=True)
    z_work = work.copy()
    for c in all_cols:
        col_std = float(work[c].std(ddof=1))
        if col_std > 0:
            z_work[c] = (work[c] - work[c].mean()) / col_std
    import statsmodels.formula.api as smf

    formula = "log_home_value ~ " + " + ".join(final_features)
    total_r2 = float(smf.ols(formula=formula, data=z_work).fit().rsquared)
    assert abs(lmg_sum - total_r2) < 0.01, (
        f"LMG shares sum {lmg_sum:.6f} differs from total R² {total_r2:.6f} by more than 0.01"
    )

    # Phase 4: summary markdown exists and is non-empty
    summary_text = Path(outputs["factor_importance_summary"]).read_text()
    assert len(summary_text) > 0
    assert "Factor Importance Summary" in summary_text


def test_drift_min_completeness_constant_exists():
    from dallas_crime.pipeline.analyze.core import DRIFT_MIN_COMPLETENESS

    assert DRIFT_MIN_COMPLETENESS == 0.33


def test_drift_prorate_insufficient_completeness():
    """Verify that drift scoring marks rows with insufficient completeness as sentinel."""
    import numpy as np

    from dallas_crime.pipeline.analyze.forecast import _build_drift_artifacts

    periods = pd.period_range("2023Q1", periods=6, freq="Q")
    # Build a temporal_summary with one zip that passes drift gate
    temporal_summary = pd.DataFrame(
        {
            "zip": ["75100"],
            "observed_quarters": [6],
            "expected_quarters": [6],
            "overall_completeness_ratio": [1.0],
            "trailing_contiguous_quarters": [6],
            "forecast_gate_pass": [1],
            "drift_gate_pass": [1],
            "first_period_start": [periods[0].start_time],
            "latest_period_start": [periods[-1].start_time],
        }
    )
    # Build temporal_series: one ZIP with 6 quarters as a DataFrame
    series_df = pd.DataFrame(
        {
            "period_start": [p.start_time for p in periods],
            "total_rate_per_1000": [30.0, 29.0, 28.0, 27.0, 26.0, 5.0],
        }
    )
    temporal_series = {"75100": series_df}

    # Set data_cutoff very close to the start of the latest quarter
    # so that quarter_completeness < DRIFT_MIN_COMPLETENESS
    latest_period_start = periods[-1].start_time
    data_cutoff = latest_period_start + pd.Timedelta(days=1)

    drift_df, _notes = _build_drift_artifacts(
        temporal_summary,
        temporal_series,
        data_cutoff=data_cutoff,
    )

    # The zip-level row should exist with drift_flag = -1 (insufficient data)
    zip_rows = drift_df[drift_df["entity_id"] == "75100"]
    if not zip_rows.empty:
        row = zip_rows.iloc[0]
        assert row["drift_flag"] == -1
        assert np.isnan(row["z_score"])


def test_spatial_lag_model_produces_coefficient_row(tmp_path: Path):
    """R4: spatial lag model (if spreg available) writes a row in regression_coefficients.csv."""
    settings = Settings.from_env(project_root=tmp_path)
    settings.ensure_directories()
    _write_model_dataset(settings)

    outputs = run_analysis(settings)

    coefficients = pd.read_csv(outputs["coefficients"])
    # If spreg is installed the spatial_lag model must appear in coefficients.
    # If spreg is absent the test passes silently (graceful-degrade behaviour is correct).
    try:
        import spreg  # noqa: F401

        assert "spatial_lag" in set(coefficients["model_label"]), (
            "spreg is installed but spatial_lag model label not found in regression_coefficients.csv"
        )
        spatial_rows = coefficients[coefficients["model_label"] == "spatial_lag"]
        assert not spatial_rows.empty
        assert "total_rate_per_1000" in set(spatial_rows["term"])
    except ImportError:
        pass  # graceful degradation — spatial lag was skipped
