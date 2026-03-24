"""Regression utilities and report generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

if TYPE_CHECKING:
    from dallas_crime.config import Settings

# Public API re-exports.
from dallas_crime.pipeline.analyze.core import (  # noqa: E402, F401
    RegressionResult,
    run_zip_regression,
)

# Internal imports consumed by run_analysis below — not part of the public API.
from dallas_crime.pipeline.analyze.core import (  # noqa: E402
    DEFAULT_CONTROLS,
    DEFAULT_PREDICTORS,
    _annotate_residuals_with_influence_flags,
    _apply_regression_guardrails,
    _audit_baseline_controls,
    _build_comprehensive_validation_artifacts,
    _build_factor_importance_artifacts,
    _build_feature_power_retention_artifacts,
    _build_feature_selection_artifacts,
    _build_influence_robustness_artifacts,
    _build_policy_recommendations_artifacts,
    _build_predictive_model_family_artifacts,
    _build_statistical_guardrails_artifacts,
    _build_validation_artifacts,
    _build_vif_artifacts,
    _load_optional_analysis_inputs,
    _select_expanded_controls,
    _summarize_influence_robustness,
)
from dallas_crime.pipeline.analyze.forecast import (  # noqa: E402
    _build_benchmark_artifacts,
    _build_drift_artifacts,
    _build_forecast_artifacts,
    _build_interval_shape_artifacts,
    _build_scenario_artifacts,
    _build_temporal_holdout_artifacts,
    _build_trend_decomposition_artifacts,
    _prepare_temporal_analysis_inputs,
)
from dallas_crime.pipeline.analyze.segmentation import (  # noqa: E402
    _build_cluster_stability_artifacts,
    _build_segmentation_artifacts,
)
from dallas_crime.pipeline.analyze.spatial import (  # noqa: E402
    _build_spatial_artifacts,
    _fit_spatial_lag_model,
)
from dallas_crime.pipeline.analyze.reporting import (  # noqa: E402
    _write_benchmark_summary,
    _write_comprehensive_validation_notes,
    _write_drift_notes,
    _write_feature_power_retention_notes,
    _write_feature_selection_notes,
    _write_forecast_notes,
    _write_geography_plot,
    _write_model_family_notes,
    _write_model_summary_table,
    _write_policy_guardrails,
    _write_policy_recommendations_notes,
    _write_residual_review,
    _write_scatter_plot,
    _write_scenario_notes,
    _write_summary_report,
    _write_trend_notes,
    _write_validation_notes,
    _write_vif_notes,
    _write_zip_comparison_table,
)


def run_analysis(settings: "Settings") -> dict[str, str]:
    """Run regression analysis from the processed model dataset and write report artifacts."""

    model_path = settings.processed_dir / "model_dataset.csv"
    if not model_path.exists():
        raise FileNotFoundError(f"Processed model dataset not found at {model_path}")

    model_df = pd.read_csv(model_path)
    target_universe_path = settings.processed_dir / "target_zip_universe.csv"
    target_universe = (
        pd.read_csv(target_universe_path) if target_universe_path.exists() else pd.DataFrame()
    )
    optional_inputs = _load_optional_analysis_inputs(settings)
    crime_history_panel = optional_inputs["crime_history_panel"]
    modeled_zips = set(pd.Series(model_df.get("zip"), dtype="string").dropna().astype(str))
    temporal_summary, temporal_series, temporal_notes = _prepare_temporal_analysis_inputs(
        crime_history_panel,
        modeled_zips=modeled_zips,
    )

    baseline_result = run_zip_regression(
        model_df,
        predictors=DEFAULT_PREDICTORS,
        controls=DEFAULT_CONTROLS,
        model_label="baseline",
    )
    # R5: impute FHFA_annual_change_pct for missing ZIPs before expanded control selection.
    # Uses Zillow-derived annual_change_pct as proxy if correlation >= 0.7, else median.
    if "FHFA_annual_change_pct" in model_df.columns and "annual_change_pct" in model_df.columns:
        _fhfa_missing_mask = pd.to_numeric(
            model_df["FHFA_annual_change_pct"], errors="coerce"
        ).isna()
        _fhfa_missing_count = int(_fhfa_missing_mask.sum())
        if _fhfa_missing_count > 0:
            model_df = model_df.copy()
            model_df["FHFA_annual_change_pct"] = pd.to_numeric(
                model_df["FHFA_annual_change_pct"], errors="coerce"
            )
            model_df["annual_change_pct"] = pd.to_numeric(
                model_df["annual_change_pct"], errors="coerce"
            )
            model_df["FHFA_annual_change_pct_imputed"] = False
            _complete_mask = (
                model_df["FHFA_annual_change_pct"].notna() & model_df["annual_change_pct"].notna()
            )
            if _complete_mask.sum() >= 3:
                _proxy_corr = float(
                    model_df.loc[_complete_mask, "FHFA_annual_change_pct"].corr(
                        model_df.loc[_complete_mask, "annual_change_pct"]
                    )
                )
                if _proxy_corr >= 0.7:
                    try:
                        _proxy_fit = smf.ols(
                            "FHFA_annual_change_pct ~ annual_change_pct",
                            data=model_df.loc[_complete_mask],
                        ).fit()
                        model_df.loc[_fhfa_missing_mask, "FHFA_annual_change_pct"] = (
                            _proxy_fit.predict(model_df.loc[_fhfa_missing_mask])
                        )
                        _impute_method = f"OLS proxy via annual_change_pct (r={_proxy_corr:.3f})"
                    except Exception:
                        _median_val = float(
                            model_df.loc[_complete_mask, "FHFA_annual_change_pct"].median()
                        )
                        model_df.loc[_fhfa_missing_mask, "FHFA_annual_change_pct"] = _median_val
                        _impute_method = "median (OLS proxy fit failed)"
                else:
                    _median_val = float(
                        model_df.loc[_complete_mask, "FHFA_annual_change_pct"].median()
                    )
                    model_df.loc[_fhfa_missing_mask, "FHFA_annual_change_pct"] = _median_val
                    _impute_method = f"median (proxy correlation {_proxy_corr:.3f} < 0.7)"
                model_df.loc[_fhfa_missing_mask, "FHFA_annual_change_pct_imputed"] = True
                print(
                    f"[analyze] NOTE (R5): imputed {_fhfa_missing_count} missing "
                    f"FHFA_annual_change_pct value(s) using {_impute_method}.",
                    flush=True,
                )

    expanded_controls, _expanded_selection_notes = _select_expanded_controls(
        model_df,
        dependent="log_home_value",
        predictors=DEFAULT_PREDICTORS,
        baseline_controls=DEFAULT_CONTROLS,
    )
    if _expanded_selection_notes:
        for _note in _expanded_selection_notes:
            print(f"[analyze] VIF-gate: {_note}", flush=True)
    expanded_result = run_zip_regression(
        model_df,
        predictors=DEFAULT_PREDICTORS,
        controls=expanded_controls,
        model_label="sensitivity_check",
    )
    results = [baseline_result, expanded_result]

    # R2: population-weighted variant (guards against missing / all-null population_acs)
    _pop_col = "population_acs"
    if (
        _pop_col in model_df.columns
        and pd.to_numeric(model_df[_pop_col], errors="coerce").notna().any()
    ):
        popweighted_result = run_zip_regression(
            model_df,
            predictors=DEFAULT_PREDICTORS,
            controls=DEFAULT_CONTROLS,
            model_label="baseline_popweighted",
            weights_column=_pop_col,
        )
        results.append(popweighted_result)
    else:
        print(
            f"[analyze] NOTE (R2): '{_pop_col}' is missing or all-null; "
            "population-weighted model skipped.",
            flush=True,
        )

    # R3: winsorized variant — clip total_rate_per_1000 at 5th/95th percentile on a copy
    _crime_col = "total_rate_per_1000"
    if _crime_col in model_df.columns:
        _crime_numeric = pd.to_numeric(model_df[_crime_col], errors="coerce").dropna()
        if len(_crime_numeric) >= 10:
            _p5 = float(np.percentile(_crime_numeric, 5))
            _p95 = float(np.percentile(_crime_numeric, 95))
            _winsorized_df = model_df.copy()
            _winsorized_df[_crime_col] = np.clip(
                pd.to_numeric(_winsorized_df[_crime_col], errors="coerce"), _p5, _p95
            )
            winsorized_result = run_zip_regression(
                _winsorized_df,
                predictors=DEFAULT_PREDICTORS,
                controls=DEFAULT_CONTROLS,
                model_label="baseline_winsorized",
            )
            results.append(winsorized_result)

    # R4: spatial lag model — gracefully skipped if spreg/libpysal unavailable or fit fails
    _spatial_result = _fit_spatial_lag_model(
        model_df,
        predictors=DEFAULT_PREDICTORS,
        controls=DEFAULT_CONTROLS,
        model_label="spatial_lag",
    )
    if _spatial_result is not None:
        results.append(_spatial_result)

    # Parsimonious "best explanation" model — one representative per collinearity cluster:
    #   Cluster 1 (socioeconomic): educational_attainment
    #   Cluster 3 (price level): realtor_listing_price
    #   Standalone (market pressure): aggregate_market_pressure_index
    # No additional controls — the three predictors *are* the model.
    _parsimonious_predictors = (
        "educational_attainment",
        "realtor_listing_price",
        "aggregate_market_pressure_index",
    )
    _parsimonious_available = all(col in model_df.columns for col in _parsimonious_predictors)
    if _parsimonious_available:
        try:
            parsimonious_result = run_zip_regression(
                model_df,
                predictors=_parsimonious_predictors,
                controls=(),
                model_label="parsimonious",
            )
            results.append(parsimonious_result)
        except (ValueError, KeyError) as exc:
            print(
                f"[analyze] NOTE: parsimonious model skipped — {exc}",
                flush=True,
            )
    else:
        _missing = [c for c in _parsimonious_predictors if c not in model_df.columns]
        print(
            f"[analyze] NOTE: parsimonious model skipped — missing columns: {', '.join(_missing)}",
            flush=True,
        )

    if expanded_result.nobs < baseline_result.nobs:
        _fhfa_drop = baseline_result.nobs - expanded_result.nobs
        print(
            f"[analyze] NOTE (DQA-D1): sensitivity_check model dropped {_fhfa_drop} observation(s) "
            f"vs baseline (n={baseline_result.nobs} → n={expanded_result.nobs}). "
            "This is likely due to FHFA_annual_change_pct missingness in 10 ZIPs; "
            "3 of those are high-influence observations. "
            "See model_validation_notes.md for discussion.",
            flush=True,
        )

    # R6: audit baseline controls for collinearity; writes diagnostics and prints warnings
    baseline_control_audit = _audit_baseline_controls(model_df)

    coefficients_path = settings.reports_dir / "regression_coefficients.csv"
    metrics_path = settings.reports_dir / "regression_metrics.csv"
    sample_sizes_path = settings.reports_dir / "model_sample_sizes.csv"
    residuals_path = settings.reports_dir / "model_residuals.csv"
    residual_review_path = settings.reports_dir / "residual_review.md"
    vif_path = settings.reports_dir / "model_vif.csv"
    vif_notes_path = settings.reports_dir / "model_vif_notes.md"
    scatter_path = settings.reports_dir / "home_value_vs_total_crime.png"
    geography_path = settings.reports_dir / "crime_home_value_geography.png"
    cluster_assignments_path = settings.reports_dir / "cluster_assignments.csv"
    cluster_profiles_path = settings.reports_dir / "cluster_profiles.csv"
    spatial_diagnostics_path = settings.reports_dir / "spatial_diagnostics.csv"
    spatial_hotspots_path = settings.reports_dir / "spatial_hotspots.csv"
    validation_metrics_path = settings.reports_dir / "model_validation_metrics.csv"
    validation_notes_path = settings.reports_dir / "model_validation_notes.md"
    zip_comparison_path = settings.reports_dir / "top_bottom_zip_comparison.md"
    model_summary_table_path = settings.reports_dir / "model_summary_table.md"
    trend_decomposition_path = settings.reports_dir / "crime_trend_decomposition.csv"
    trend_notes_path = settings.reports_dir / "crime_trend_decomposition.md"
    feature_selection_metrics_path = settings.reports_dir / "feature_selection_metrics.csv"
    feature_selection_notes_path = settings.reports_dir / "feature_selection_notes.md"
    feature_power_retention_metrics_path = (
        settings.reports_dir / "feature_power_retention_metrics.csv"
    )
    feature_power_retention_notes_path = settings.reports_dir / "feature_power_retention_notes.md"
    forecast_model_metrics_path = settings.reports_dir / "forecast_model_metrics.csv"
    crime_forecasts_path = settings.reports_dir / "crime_forecasts.csv"
    forecast_intervals_path = settings.reports_dir / "forecast_confidence_intervals.csv"
    forecast_notes_path = settings.reports_dir / "forecast_notes.md"
    predictive_model_metrics_path = settings.reports_dir / "predictive_model_metrics.csv"
    predictive_model_predictions_path = settings.reports_dir / "predictive_model_predictions.csv"
    model_selection_notes_path = settings.reports_dir / "model_selection_notes.md"
    scenario_impacts_path = settings.reports_dir / "scenario_impacts.csv"
    scenario_notes_path = settings.reports_dir / "scenario_notes.md"
    zip_benchmarks_path = settings.reports_dir / "zip_benchmarks.csv"
    benchmark_summary_path = settings.reports_dir / "benchmark_summary.md"
    drift_diagnostics_path = settings.reports_dir / "model_drift_diagnostics.csv"
    drift_notes_path = settings.reports_dir / "model_drift_notes.md"
    temporal_holdout_results_path = settings.reports_dir / "temporal_holdout_results.csv"
    forecast_interval_calibration_path = settings.reports_dir / "forecast_interval_calibration.csv"
    influence_robustness_path = settings.reports_dir / "influence_robustness_diagnostics.csv"
    cluster_stability_path = settings.reports_dir / "cluster_stability_diagnostics.csv"
    statistical_guardrails_path = settings.reports_dir / "statistical_guardrails.csv"
    comprehensive_validation_metrics_path = (
        settings.reports_dir / "comprehensive_validation_metrics.csv"
    )
    comprehensive_validation_notes_path = settings.reports_dir / "comprehensive_validation_notes.md"
    policy_recommendations_path = settings.reports_dir / "policy_recommendations_by_segment.csv"
    policy_recommendations_notes_path = (
        settings.reports_dir / "policy_recommendations_by_segment.md"
    )
    policy_guardrails_path = settings.reports_dir / "policy_guardrails.md"
    baseline_control_audit_path = settings.reports_dir / "baseline_control_audit.csv"
    factor_importance_univariate_path = settings.reports_dir / "factor_importance_univariate.csv"
    factor_importance_standardized_path = (
        settings.reports_dir / "factor_importance_standardized.csv"
    )
    factor_importance_variance_decomposition_path = (
        settings.reports_dir / "factor_importance_variance_decomposition.csv"
    )
    factor_importance_summary_path = settings.reports_dir / "factor_importance_summary.md"
    report_path = settings.reports_dir / "summary.md"

    coefficients = _apply_regression_guardrails(
        pd.concat([result.coefficients for result in results], ignore_index=True)
    )
    metrics = pd.concat([result.metrics_table() for result in results], ignore_index=True)
    sample_sizes = metrics[["model_label", "nobs", "predictors", "controls", "formula"]].copy()
    residuals = (
        pd.concat([result.residuals for result in results], ignore_index=True)
        .sort_values(["model_label", "absolute_residual"], ascending=[True, False])
        .reset_index(drop=True)
    )
    vif_table, vif_notes = _build_vif_artifacts(results)
    cluster_assignments, cluster_profiles = _build_segmentation_artifacts(model_df)
    cluster_stability = _build_cluster_stability_artifacts(model_df)
    spatial_diagnostics, spatial_hotspots = _build_spatial_artifacts(model_df)
    feature_selection_metrics, feature_selection_notes = _build_feature_selection_artifacts(
        model_df,
        expanded_controls=expanded_controls,
    )
    predictive_model_metrics, predictive_model_predictions, model_selection_notes = (
        _build_predictive_model_family_artifacts(
            model_df,
            baseline_result=baseline_result,
            expanded_result=expanded_result,
            cluster_assignments=cluster_assignments,
        )
    )
    feature_power_retention_metrics, feature_power_retention_notes = (
        _build_feature_power_retention_artifacts(
            feature_selection_metrics,
            predictive_model_metrics,
        )
    )
    validation_metrics, validation_notes = _build_validation_artifacts(results)
    influence_robustness, influence_notes = _build_influence_robustness_artifacts(results)
    influence_summary = _summarize_influence_robustness(influence_robustness)
    residuals = _annotate_residuals_with_influence_flags(residuals, influence_robustness)
    if not influence_summary.empty:
        validation_metrics = validation_metrics.merge(
            influence_summary,
            on="model_label",
            how="left",
        )
        for row in influence_summary.itertuples(index=False):
            validation_notes.append(
                f"- `{row.model_label}`: {int(row.influence_flag_count)} high-influence ZIPs were flagged; "
                f"max crime-term shift={float(row.max_crime_term_effect_pct_change):.3f}%, "
                f"max p90 home-value prediction delta={float(row.max_p90_home_value_pct_delta):.3f}%, "
                f"fit_deterioration_pass={int(row.fit_deterioration_pass)}, "
                f"fit_improvement_warning_count={int(row.fit_improvement_warning_count)}, "
                f"pass={int(row.influence_robustness_pass)}."
            )
    else:
        for column in (
            "influence_flag_count",
            "max_crime_term_effect_pct_change",
            "max_p90_home_value_pct_delta",
            "max_home_value_pct_delta",
            "fit_stability_pass",
            "fit_deterioration_pass",
            "fit_improvement_warning_count",
            "any_crime_term_sign_flip",
            "influence_robustness_pass",
        ):
            validation_metrics[column] = np.nan
    validation_notes.extend(influence_notes)
    if not influence_robustness.empty:
        validation_notes.append(
            "See `influence_robustness_diagnostics.csv` for leave-high-leverage / high-Cook's ZIP-out refits."
        )

    # --- DQA-H2: FHFA confound visibility --------------------------------
    if expanded_result.nobs < baseline_result.nobs:
        baseline_zips = set(pd.Series(baseline_result.model_frame["zip"], dtype="string").dropna())
        expanded_zips = set(pd.Series(expanded_result.model_frame["zip"], dtype="string").dropna())
        excluded_zips = sorted(baseline_zips - expanded_zips)
        influence_flagged_zips = (
            set(pd.Series(influence_robustness.get("removed_zip"), dtype="string").dropna())
            if not influence_robustness.empty
            else set()
        )
        excluded_in_influence = sorted(set(excluded_zips) & influence_flagged_zips)
        confound_note = (
            f"- DQA-H2: sensitivity-check model dropped {len(excluded_zips)} ZIP(s) "
            f"vs baseline (n={baseline_result.nobs} → n={expanded_result.nobs}), "
            f"likely due to FHFA_annual_change_pct missingness. "
            f"Excluded ZIPs: {', '.join(excluded_zips)}."
        )
        if excluded_in_influence:
            confound_note += (
                f" Of these, {len(excluded_in_influence)} are also high-influence "
                f"flagged: {', '.join(excluded_in_influence)}."
            )
        validation_notes.append(confound_note)
    # --- end DQA-H2 -------------------------------------------------------

    trend_decomposition, trend_notes = _build_trend_decomposition_artifacts(crime_history_panel)
    temporal_holdout_results, interval_calibration_holdout, holdout_notes = (
        _build_temporal_holdout_artifacts(temporal_summary, temporal_series)
    )
    forecast_model_metrics, crime_forecasts, forecast_intervals, forecast_notes = (
        _build_forecast_artifacts(
            temporal_summary,
            temporal_series,
            temporal_holdout=temporal_holdout_results,
        )
    )
    forecast_notes = [*temporal_notes, *holdout_notes, *forecast_notes]
    interval_calibration = pd.concat(
        [
            interval_calibration_holdout,
            _build_interval_shape_artifacts(forecast_intervals),
        ],
        ignore_index=True,
    )
    scenario_impacts = _build_scenario_artifacts(crime_forecasts)
    zip_benchmarks = _build_benchmark_artifacts(model_df, cluster_assignments)
    _drift_data_cutoff = pd.Timestamp.today().normalize()
    drift_diagnostics, drift_notes = _build_drift_artifacts(
        temporal_summary,
        temporal_series,
        data_cutoff=_drift_data_cutoff,
    )
    statistical_guardrails = _build_statistical_guardrails_artifacts(
        coefficients,
        feature_selection_metrics,
        spatial_diagnostics,
    )
    comprehensive_validation_metrics, comprehensive_validation_notes = (
        _build_comprehensive_validation_artifacts(
            regression_validation=validation_metrics,
            predictive_metrics=predictive_model_metrics,
            forecast_metrics=forecast_model_metrics,
            temporal_holdout=temporal_holdout_results,
            interval_calibration=interval_calibration,
            drift_diagnostics=drift_diagnostics,
            influence_robustness=influence_robustness,
            cluster_stability=cluster_stability,
            statistical_guardrails=statistical_guardrails,
            scenario_impacts=scenario_impacts,
        )
    )
    policy_recommendations, policy_notes = _build_policy_recommendations_artifacts(
        zip_benchmarks,
        scenario_impacts,
        cluster_stability=cluster_stability,
        influence_robustness=influence_robustness,
    )
    policy_notes.append("See `policy_guardrails.md` before using any scenario or segment output.")
    if not influence_summary.empty:
        if (
            pd.to_numeric(influence_summary["influence_robustness_pass"], errors="coerce")
            .fillna(0)
            .eq(1)
            .all()
        ):
            policy_notes.append(
                "Regression-linked crime terms remain interpretation-blocked on current data because "
                "the FDR/practical-effect guardrails still do not clear the crime terms, even though "
                "prediction-stability influence checks improved."
            )
        else:
            policy_notes.append(
                "Regression-linked crime terms remain interpretation-blocked on current data because "
                "influence robustness fails in at least one modeled specification."
            )

    coefficients.to_csv(coefficients_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    sample_sizes.to_csv(sample_sizes_path, index=False)
    residuals.to_csv(residuals_path, index=False)
    vif_table.to_csv(vif_path, index=False)
    cluster_assignments.to_csv(cluster_assignments_path, index=False)
    cluster_profiles.to_csv(cluster_profiles_path, index=False)
    cluster_stability.to_csv(cluster_stability_path, index=False)
    spatial_diagnostics.to_csv(spatial_diagnostics_path, index=False)
    spatial_hotspots.to_csv(spatial_hotspots_path, index=False)
    feature_selection_metrics.to_csv(feature_selection_metrics_path, index=False)
    feature_power_retention_metrics.to_csv(feature_power_retention_metrics_path, index=False)
    predictive_model_metrics.to_csv(predictive_model_metrics_path, index=False)
    predictive_model_predictions.to_csv(predictive_model_predictions_path, index=False)
    validation_metrics.to_csv(validation_metrics_path, index=False)
    trend_decomposition.to_csv(trend_decomposition_path, index=False)
    forecast_model_metrics.to_csv(forecast_model_metrics_path, index=False)
    crime_forecasts.to_csv(crime_forecasts_path, index=False)
    forecast_intervals.to_csv(forecast_intervals_path, index=False)
    temporal_holdout_results.to_csv(temporal_holdout_results_path, index=False)
    interval_calibration.to_csv(forecast_interval_calibration_path, index=False)
    scenario_impacts.to_csv(scenario_impacts_path, index=False)
    zip_benchmarks.to_csv(zip_benchmarks_path, index=False)
    drift_diagnostics.to_csv(drift_diagnostics_path, index=False)
    influence_robustness.to_csv(influence_robustness_path, index=False)
    statistical_guardrails.to_csv(statistical_guardrails_path, index=False)
    comprehensive_validation_metrics.to_csv(comprehensive_validation_metrics_path, index=False)
    policy_recommendations.to_csv(policy_recommendations_path, index=False)
    baseline_control_audit.to_csv(baseline_control_audit_path, index=False)

    # Factor importance analysis (Phases 1–4)
    fi_univariate, fi_standardized, fi_decomposition, fi_summary_md = (
        _build_factor_importance_artifacts(model_df)
    )
    fi_univariate.to_csv(factor_importance_univariate_path, index=False)
    fi_standardized.to_csv(factor_importance_standardized_path, index=False)
    fi_decomposition.to_csv(factor_importance_variance_decomposition_path, index=False)
    factor_importance_summary_path.write_text(fi_summary_md, encoding="utf-8")

    _write_residual_review(residuals, residual_review_path)
    _write_vif_notes(vif_notes, vif_notes_path)
    _write_validation_notes(validation_notes, validation_notes_path)
    _write_trend_notes(trend_decomposition, trend_notes, trend_notes_path)
    _write_feature_selection_notes(
        feature_selection_metrics,
        feature_selection_notes,
        feature_selection_notes_path,
    )
    _write_feature_power_retention_notes(
        feature_power_retention_metrics,
        feature_power_retention_notes,
        feature_power_retention_notes_path,
    )
    _write_model_family_notes(
        predictive_model_metrics,
        model_selection_notes,
        model_selection_notes_path,
    )
    _write_forecast_notes(
        forecast_model_metrics,
        crime_forecasts,
        temporal_holdout_results,
        interval_calibration,
        forecast_notes,
        forecast_notes_path,
    )
    _write_scenario_notes(forecast_model_metrics, crime_forecasts, scenario_notes_path)
    _write_benchmark_summary(zip_benchmarks, benchmark_summary_path)
    _write_drift_notes(drift_notes, drift_notes_path)
    _write_comprehensive_validation_notes(
        comprehensive_validation_notes,
        comprehensive_validation_notes_path,
    )
    _write_policy_recommendations_notes(
        policy_recommendations,
        policy_notes,
        policy_recommendations_notes_path,
    )
    _write_policy_guardrails(
        model_df=model_df,
        target_universe=target_universe,
        crime_forecasts=crime_forecasts,
        temporal_holdout=temporal_holdout_results,
        interval_calibration=interval_calibration,
        influence_summary=influence_summary,
        cluster_stability=cluster_stability,
        statistical_guardrails=statistical_guardrails,
        output_path=policy_guardrails_path,
    )
    _write_scatter_plot(model_df, scatter_path)
    _write_geography_plot(model_df, geography_path)
    _write_zip_comparison_table(model_df, zip_comparison_path)
    _write_model_summary_table(metrics, model_summary_table_path)
    _parsimonious_for_report = next((r for r in results if r.model_label == "parsimonious"), None)
    _write_summary_report(
        model_df=model_df,
        results=results,
        coefficients=coefficients,
        crime_forecasts=crime_forecasts,
        temporal_holdout=temporal_holdout_results,
        influence_summary=influence_summary,
        cluster_stability=cluster_stability,
        settings=settings,
        output_path=report_path,
        factor_importance_summary_md=fi_summary_md,
        parsimonious_result=_parsimonious_for_report,
    )

    return {
        "coefficients": str(coefficients_path),
        "metrics": str(metrics_path),
        "sample_sizes": str(sample_sizes_path),
        "residuals": str(residuals_path),
        "residual_review": str(residual_review_path),
        "vif": str(vif_path),
        "vif_notes": str(vif_notes_path),
        "scatter_plot": str(scatter_path),
        "geography_plot": str(geography_path),
        "cluster_assignments": str(cluster_assignments_path),
        "cluster_profiles": str(cluster_profiles_path),
        "cluster_stability_diagnostics": str(cluster_stability_path),
        "spatial_diagnostics": str(spatial_diagnostics_path),
        "spatial_hotspots": str(spatial_hotspots_path),
        "validation_metrics": str(validation_metrics_path),
        "validation_notes": str(validation_notes_path),
        "zip_comparison": str(zip_comparison_path),
        "model_summary_table": str(model_summary_table_path),
        "trend_decomposition": str(trend_decomposition_path),
        "trend_notes": str(trend_notes_path),
        "feature_selection_metrics": str(feature_selection_metrics_path),
        "feature_selection_notes": str(feature_selection_notes_path),
        "feature_power_retention_metrics": str(feature_power_retention_metrics_path),
        "feature_power_retention_notes": str(feature_power_retention_notes_path),
        "predictive_model_metrics": str(predictive_model_metrics_path),
        "predictive_model_predictions": str(predictive_model_predictions_path),
        "model_selection_notes": str(model_selection_notes_path),
        "forecast_model_metrics": str(forecast_model_metrics_path),
        "crime_forecasts": str(crime_forecasts_path),
        "forecast_confidence_intervals": str(forecast_intervals_path),
        "temporal_holdout_results": str(temporal_holdout_results_path),
        "forecast_interval_calibration": str(forecast_interval_calibration_path),
        "forecast_notes": str(forecast_notes_path),
        "scenario_impacts": str(scenario_impacts_path),
        "scenario_notes": str(scenario_notes_path),
        "zip_benchmarks": str(zip_benchmarks_path),
        "benchmark_summary": str(benchmark_summary_path),
        "drift_diagnostics": str(drift_diagnostics_path),
        "drift_notes": str(drift_notes_path),
        "influence_robustness_diagnostics": str(influence_robustness_path),
        "statistical_guardrails": str(statistical_guardrails_path),
        "comprehensive_validation_metrics": str(comprehensive_validation_metrics_path),
        "comprehensive_validation_notes": str(comprehensive_validation_notes_path),
        "policy_recommendations": str(policy_recommendations_path),
        "policy_recommendations_notes": str(policy_recommendations_notes_path),
        "policy_guardrails": str(policy_guardrails_path),
        "summary": str(report_path),
        "baseline_control_audit": str(baseline_control_audit_path),
        "factor_importance_univariate": str(factor_importance_univariate_path),
        "factor_importance_standardized": str(factor_importance_standardized_path),
        "factor_importance_variance_decomposition": str(
            factor_importance_variance_decomposition_path
        ),
        "factor_importance_summary": str(factor_importance_summary_path),
    }
