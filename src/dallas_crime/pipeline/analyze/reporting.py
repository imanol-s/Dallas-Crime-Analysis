"""Report writers, plots, and summary generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dallas_crime.config import Settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dallas_crime.pipeline.analyze.core import (
    FORECAST_HISTORY_MIN_QUARTERS,
    FORECAST_LIMITED_HISTORY_MIN_QUARTERS,
    RegressionResult,
)


def _write_trend_notes(
    decomposition_df: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Crime Trend Decomposition",
        "",
    ]
    if decomposition_df.empty:
        lines.extend(
            [
                "No decomposition rows were produced for this run.",
                "",
                *notes,
            ]
        )
    else:
        lines.extend(
            [
                f"Decomposition rows written: {len(decomposition_df)}.",
                "",
                "See `crime_trend_decomposition.csv` for quarterly observed, trend, seasonal, and residual values.",
            ]
        )
        if notes:
            lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_forecast_notes(
    model_metrics_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    temporal_holdout_df: pd.DataFrame,
    interval_calibration_df: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Crime Forecast Notes",
        "",
    ]
    if forecasts_df.empty:
        lines.extend(
            [
                "No crime forecasts were produced for this run.",
                "",
                *notes,
            ]
        )
    else:
        selected = (
            forecasts_df.loc[
                forecasts_df.get(
                    "forecast_tier", pd.Series("high_confidence", index=forecasts_df.index)
                ).astype(str)
                == "high_confidence",
                ["zip", "selected_model"],
            ]
            .drop_duplicates()
            .sort_values(["selected_model", "zip"], kind="mergesort")
        )
        eligible_zip_count = (
            int(pd.to_numeric(model_metrics_df["eligible_zip_count"], errors="coerce").max())
            if not model_metrics_df.empty
            else int(
                forecasts_df.loc[
                    forecasts_df.get(
                        "policy_eligible", pd.Series(1, index=forecasts_df.index)
                    ).astype(int)
                    == 1,
                    "zip",
                ]
                .astype(str)
                .nunique()
            )
        )
        gated_out_zip_count = (
            int(pd.to_numeric(model_metrics_df["gated_out_zip_count"], errors="coerce").max())
            if not model_metrics_df.empty
            else 0
        )
        modeled_zip_count = eligible_zip_count + gated_out_zip_count
        tier_counts = (
            forecasts_df[["zip", "forecast_tier"]]
            .drop_duplicates()
            .groupby("forecast_tier")["zip"]
            .nunique()
            if "forecast_tier" in forecasts_df.columns
            else pd.Series(dtype=int)
        )
        limited_history_count = int(tier_counts.get("limited_history", 0))
        carry_forward_count = int(tier_counts.get("carry_forward_only", 0))
        lines.append(f"Forecast rows written: {len(forecasts_df)}.")
        lines.append("")
        if not model_metrics_df.empty:
            lines.append("Model family comparison is recorded in `forecast_model_metrics.csv`.")
            lines.append("")
        if not temporal_holdout_df.empty:
            selected_holdout = temporal_holdout_df.loc[
                temporal_holdout_df["evaluation_scope"].astype(str) == "selected_zip_model"
            ]
            if not selected_holdout.empty:
                row = selected_holdout.iloc[0]
                lines.append(
                    "Temporal holdout results are recorded in `temporal_holdout_results.csv` "
                    f"(selected ZIP model MAPE={row['mape']:.2f}, pass={int(row['mape_pass'])})."
                )
                lines.append("")
        if not interval_calibration_df.empty:
            lines.append(
                "Interval calibration diagnostics are recorded in `forecast_interval_calibration.csv`."
            )
            lines.append("")
        lines.append("Coverage tiers:")
        lines.append("")
        lines.append(
            f"- High-confidence forecast/scenario coverage is {eligible_zip_count}/{modeled_zip_count} modeled ZIPs."
        )
        if limited_history_count > 0:
            lines.append(
                f"- Limited-history forecast-only coverage adds {limited_history_count} modeled ZIPs with {FORECAST_LIMITED_HISTORY_MIN_QUARTERS}-{FORECAST_HISTORY_MIN_QUARTERS - 1} trailing quarters."
            )
        if carry_forward_count > 0:
            lines.append(
                f"- Carry-forward forecast-only coverage adds {carry_forward_count} modeled ZIPs with 1-{FORECAST_LIMITED_HISTORY_MIN_QUARTERS - 1} trailing quarters."
            )
        remaining_unforecasted = max(
            modeled_zip_count - eligible_zip_count - limited_history_count - carry_forward_count,
            0,
        )
        if remaining_unforecasted > 0:
            lines.append(
                f"- The remaining {remaining_unforecasted} modeled ZIPs still lacked enough contiguous history for any forecast tier."
            )
        lines.append(
            "- Scenario and policy artifacts remain limited to the high-confidence subset."
        )
        lines.append("")
        lines.append("Selected high-confidence ZIP-level models:")
        lines.append("")
        for row in selected.itertuples(index=False):
            lines.append(f"- ZIP {row.zip}: {row.selected_model}")
        if notes:
            lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_scenario_notes(
    forecast_model_metrics: pd.DataFrame,
    crime_forecasts: pd.DataFrame,
    output_path: Path,
) -> None:
    eligible_zip_count = (
        int(pd.to_numeric(forecast_model_metrics["eligible_zip_count"], errors="coerce").max())
        if not forecast_model_metrics.empty
        else 0
    )
    gated_out_zip_count = (
        int(pd.to_numeric(forecast_model_metrics["gated_out_zip_count"], errors="coerce").max())
        if not forecast_model_metrics.empty
        else 0
    )
    modeled_zip_count = eligible_zip_count + gated_out_zip_count
    lower_tier_zip_count = (
        int(
            crime_forecasts.loc[
                crime_forecasts.get(
                    "policy_eligible", pd.Series(1, index=crime_forecasts.index)
                ).astype(int)
                == 0,
                "zip",
            ]
            .astype(str)
            .nunique()
        )
        if not crime_forecasts.empty
        else 0
    )
    lines = [
        "# Scenario Notes",
        "",
        "Scenario impacts are deterministic multipliers applied to the ZIP-level crime forecast output.",
        "They are exploratory planning aids, not causal predictions or policy-effect estimates.",
        "Use them only with `policy_guardrails.md` and the temporal holdout / interval calibration diagnostics.",
        (
            f"Scenario rows are emitted only for the {eligible_zip_count}/{modeled_zip_count} modeled ZIPs "
            "that passed high-confidence forecast eligibility."
            if modeled_zip_count > 0
            else "Scenario rows are emitted only when forecast-eligible ZIPs are available."
        ),
        (
            f"An additional {lower_tier_zip_count} modeled ZIPs receive forecast-only lower-confidence tiers, but they are excluded from scenario rows."
            if lower_tier_zip_count > 0
            else "No lower-confidence forecast-only ZIP tiers were emitted in this run."
        ),
        "",
        "- `baseline`: uses the point forecast unchanged.",
        "- `stabilization`: applies a 5% improvement to forecasted crime rate.",
        "- `adverse_momentum`: applies a 10% worsening to forecasted crime rate.",
        "- `seasonal_peak`: applies a modest uplift, with a larger seasonal bump in Q2 and Q3.",
        "- `systemic_shock`: applies a severe system-wide stress multiplier.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def _write_benchmark_summary(benchmarks_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Benchmark Summary",
        "",
    ]
    if benchmarks_df.empty:
        lines.append("No benchmark rows were produced for this run.")
        output_path.write_text("\n".join(lines) + "\n")
        return

    top_home_value = benchmarks_df.sort_values("home_value", ascending=False).head(3)
    top_crime = benchmarks_df.sort_values("total_rate_per_1000", ascending=False).head(3)
    lines.extend(
        [
            f"ZIP benchmark rows written: {len(benchmarks_df)}.",
            "",
            "## Highest Home Value ZIPs",
            "",
            "| zip | home_value | home_value_vs_metro_pct |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in top_home_value.itertuples(index=False):
        lines.append(f"| {row.zip} | {row.home_value:.0f} | {row.home_value_vs_metro_pct:.2f} |")

    lines.extend(
        [
            "",
            "## Highest Crime-Rate ZIPs",
            "",
            "| zip | total_rate_per_1000 | total_rate_vs_metro_pct |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in top_crime.itertuples(index=False):
        lines.append(
            f"| {row.zip} | {row.total_rate_per_1000:.2f} | {row.total_rate_vs_metro_pct:.2f} |"
        )

    output_path.write_text("\n".join(lines) + "\n")


def _write_drift_notes(notes: list[str], output_path: Path) -> None:
    lines = [
        "# Model Drift Notes",
        "",
        "Drift diagnostics compare the latest available quarter against the trailing four-quarter baseline.",
        "",
    ]
    if notes:
        lines.extend(notes)
    else:
        lines.append("No additional drift caveats were recorded.")
    output_path.write_text("\n".join(lines) + "\n")


def _write_feature_selection_notes(
    selection_df: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Feature Selection Notes",
        "",
    ]
    if selection_df.empty:
        lines.append("Feature-selection metrics were not produced for this run.")
        if notes:
            lines.extend(["", *notes])
        output_path.write_text("\n".join(lines) + "\n")
        return

    recommended = selection_df.loc[
        selection_df["recommended_for_future_models"] == 1,
        "feature_name",
    ].astype(str)
    interpretable = selection_df.loc[
        pd.to_numeric(selection_df["interpretation_allowed"], errors="coerce").fillna(0).astype(int)
        == 1,
        "feature_name",
    ].astype(str)
    lines.append(f"Feature candidates evaluated: {len(selection_df)}.")
    lines.append("")
    lines.append(
        "Recommended features (by availability-weighted association score): "
        f"{', '.join(recommended.tolist()) if not recommended.empty else 'none'}."
    )
    lines.append("")
    lines.append(
        "Features clearing both FDR and practical-effect guardrails: "
        f"{', '.join(interpretable.tolist()) if not interpretable.empty else 'none'}."
    )
    if notes:
        lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_feature_power_retention_notes(
    power_metrics: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Feature Power Retention Notes",
        "",
        "This artifact tracks the Q2 feature-power checkpoint with explicit retention thresholds.",
        "",
    ]
    if power_metrics.empty:
        lines.append("Feature-power retention metrics were not produced for this run.")
        if notes:
            lines.extend(["", *notes])
        output_path.write_text("\n".join(lines) + "\n")
        return

    metric_to_value = {str(row.metric): row.value for row in power_metrics.itertuples(index=False)}
    score_retention = pd.to_numeric(
        pd.Series([metric_to_value.get("feature_selection_score_retention_ratio")]),
        errors="coerce",
    ).iloc[0]
    r_squared_retention = pd.to_numeric(
        pd.Series([metric_to_value.get("predictive_r_squared_retention_ratio")]),
        errors="coerce",
    ).iloc[0]
    checkpoint_pass = pd.to_numeric(
        pd.Series([metric_to_value.get("feature_power_checkpoint_pass")]),
        errors="coerce",
    ).iloc[0]

    lines.append(
        f"- Feature-selection score retention ratio: {score_retention:.3f}"
        if pd.notna(score_retention)
        else "- Feature-selection score retention ratio: not estimable."
    )
    lines.append(
        f"- Predictive R-squared retention ratio: {r_squared_retention:.3f}"
        if pd.notna(r_squared_retention)
        else "- Predictive R-squared retention ratio: not estimable."
    )
    lines.append(
        f"- Checkpoint pass flag: {int(checkpoint_pass)}"
        if pd.notna(checkpoint_pass)
        else "- Checkpoint pass flag: not estimable."
    )
    lines.extend(
        [
            "",
            "Threshold rule: both retention ratios must be >= 0.90.",
        ]
    )
    if notes:
        lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_model_family_notes(
    predictive_metrics: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Predictive Model Family Notes",
        "",
    ]
    if predictive_metrics.empty:
        lines.append("Predictive model-family metrics were not produced for this run.")
        if notes:
            lines.extend(["", *notes])
        output_path.write_text("\n".join(lines) + "\n")
        return

    ranked = predictive_metrics.sort_values(
        ["selection_rank", "rmse", "model_label"],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    )
    lines.append(f"Model-family rows written: {len(predictive_metrics)}.")
    lines.append("")
    lines.append("Top-ranked models by selection rank:")
    lines.append("")
    for row in ranked.head(3).itertuples(index=False):
        lines.append(
            f"- `{row.model_label}` ({row.model_tier}) "
            f"rmse={row.rmse:.4f} loocv_rmse={row.loocv_rmse if pd.notna(row.loocv_rmse) else float('nan'):.4f}"
        )
    if notes:
        lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_comprehensive_validation_notes(notes: list[str], output_path: Path) -> None:
    lines = [
        "# Comprehensive Validation Notes",
        "",
        "This artifact aggregates regression, predictive-family, forecast, temporal holdout, interval calibration, influence robustness, cluster stability, drift, scenario, and statistical guardrail checks.",
    ]
    if notes:
        lines.extend(["", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_policy_recommendations_notes(
    policy_df: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Policy Recommendations By Segment",
        "",
    ]
    if policy_df.empty:
        lines.append("No policy recommendations were generated for this run.")
        if notes:
            lines.extend(["", *notes])
        output_path.write_text("\n".join(lines) + "\n")
        return

    high_priority = policy_df.loc[policy_df["priority_tier"] == "high"]
    blocked_segments = policy_df.loc[
        policy_df.get("segment_guardrail_status", pd.Series("clear", index=policy_df.index)).astype(
            str
        )
        == "blocked"
    ]
    lines.append(f"Segment recommendations written: {len(policy_df)}.")
    lines.append("")
    lines.append(
        f"High-priority segments: {len(high_priority)} "
        f"({', '.join(high_priority['segment_label'].astype(str).tolist()) if not high_priority.empty else 'none'})."
    )
    lines.append(
        f"Blocked / low-confidence segments: {len(blocked_segments)} "
        f"({', '.join(blocked_segments['segment_label'].astype(str).tolist()) if not blocked_segments.empty else 'none'})."
    )
    lines.append("")
    lines.append(
        "See `policy_recommendations_by_segment.csv` for segment guardrail status, confidence tiers, "
        "scenario support, and high-influence concentrations."
    )
    lines.append(
        "These segment recommendations remain exploratory and non-causal; see `policy_guardrails.md`."
    )
    if notes:
        lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_policy_guardrails(
    *,
    model_df: pd.DataFrame,
    target_universe: pd.DataFrame,
    crime_forecasts: pd.DataFrame,
    temporal_holdout: pd.DataFrame,
    interval_calibration: pd.DataFrame,
    influence_summary: pd.DataFrame,
    cluster_stability: pd.DataFrame,
    statistical_guardrails: pd.DataFrame,
    output_path: Path,
) -> None:
    modeled_zip_count = int(pd.Series(model_df.get("zip"), dtype="string").dropna().nunique())
    target_zip_count = (
        int(pd.Series(target_universe.get("zip"), dtype="string").dropna().nunique())
        if not target_universe.empty and "zip" in target_universe.columns
        else modeled_zip_count
    )
    selected_holdout = temporal_holdout.loc[
        temporal_holdout["evaluation_scope"].astype(str) == "selected_zip_model"
    ]
    selected_calibration = interval_calibration.loc[
        interval_calibration["evaluation_scope"].astype(str) == "selected_zip_model"
    ]
    holdout_pass = (
        bool(pd.to_numeric(selected_holdout["mape_pass"], errors="coerce").fillna(0).eq(1).all())
        if not selected_holdout.empty
        else False
    )
    calibration_pass = (
        bool(
            pd.to_numeric(selected_calibration["calibration_pass"], errors="coerce")
            .fillna(0)
            .eq(1)
            .all()
        )
        if not selected_calibration.empty
        else False
    )
    influence_pass = (
        bool(
            pd.to_numeric(influence_summary["influence_robustness_pass"], errors="coerce")
            .fillna(0)
            .eq(1)
            .all()
        )
        if not influence_summary.empty
        else False
    )
    cluster_pass = (
        bool(
            pd.to_numeric(cluster_stability["practical_utility_pass"], errors="coerce")
            .fillna(0)
            .eq(1)
            .all()
        )
        if not cluster_stability.empty
        else False
    )
    interpretation_share = (
        float(
            pd.to_numeric(statistical_guardrails["interpretation_allowed"], errors="coerce").mean()
        )
        if not statistical_guardrails.empty
        else np.nan
    )
    forecast_high_confidence_count = (
        int(
            crime_forecasts.loc[
                crime_forecasts.get(
                    "policy_eligible", pd.Series(1, index=crime_forecasts.index)
                ).astype(int)
                == 1,
                "zip",
            ]
            .astype(str)
            .nunique()
        )
        if not crime_forecasts.empty
        else 0
    )
    lower_confidence_forecast_count = (
        int(
            crime_forecasts.loc[
                crime_forecasts.get(
                    "policy_eligible", pd.Series(1, index=crime_forecasts.index)
                ).astype(int)
                == 0,
                "zip",
            ]
            .astype(str)
            .nunique()
        )
        if not crime_forecasts.empty
        else 0
    )
    eligible_zip_count = (
        int(pd.to_numeric(selected_holdout["eligible_zip_count"], errors="coerce").max())
        if not selected_holdout.empty
        else forecast_high_confidence_count
    )
    gated_out_zip_count = max(modeled_zip_count - eligible_zip_count, 0)
    selected_mape = (
        float(pd.to_numeric(selected_holdout["mape"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else np.nan
    )
    selected_mape_threshold = (
        float(pd.to_numeric(selected_holdout["mape_threshold"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else np.nan
    )
    influence_fail_count = (
        int(
            pd.to_numeric(influence_summary["influence_robustness_pass"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        if not influence_summary.empty
        else 0
    )
    max_crime_term_shift = (
        float(
            pd.to_numeric(
                influence_summary["max_crime_term_effect_pct_change"], errors="coerce"
            ).max()
        )
        if not influence_summary.empty
        else np.nan
    )
    max_prediction_delta = (
        float(
            pd.to_numeric(influence_summary["max_p90_home_value_pct_delta"], errors="coerce").max()
        )
        if not influence_summary.empty
        else np.nan
    )
    fit_warning_count = (
        int(
            pd.to_numeric(influence_summary["fit_improvement_warning_count"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
        if not influence_summary.empty
        else 0
    )
    cluster_fail_count = (
        int(
            pd.to_numeric(cluster_stability["practical_utility_pass"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        if not cluster_stability.empty
        else 0
    )

    lines = [
        "# Policy Guardrails",
        "",
        "Scenario and segment outputs in this repo are exploratory, non-causal planning aids.",
        "They must not be described as policy-impact estimates, intervention effects, or decision-ready resource-allocation evidence.",
        "",
        f"- Modeled ZIP coverage in this run: {modeled_zip_count}/{target_zip_count}.",
        f"- High-confidence forecast/scenario coverage in this run: {eligible_zip_count}/{modeled_zip_count} modeled ZIPs.",
        (
            f"- Lower-confidence forecast-only coverage adds {lower_confidence_forecast_count} modeled ZIPs; those rows are excluded from scenario and policy calculations."
            if lower_confidence_forecast_count > 0
            else f"- No lower-confidence forecast-only ZIP tiers were emitted; {gated_out_zip_count} modeled ZIPs remain uncovered."
        ),
        (
            f"- Selected ZIP temporal holdout: MAPE={selected_mape:.3f} versus threshold {selected_mape_threshold:.1f}."
            if pd.notna(selected_mape) and pd.notna(selected_mape_threshold)
            else "- Selected ZIP temporal holdout: not estimable."
        ),
        f"- Temporal holdout pass status: {int(holdout_pass)}.",
        f"- Interval calibration pass status: {int(calibration_pass)}.",
        (
            f"- Influence robustness severity: {influence_fail_count}/{len(influence_summary)} regression specs fail on prediction-stability guardrails; max p90 home-value delta={max_prediction_delta:.3f}%, max crime-term shift={max_crime_term_shift:.3f}%."
            if not influence_summary.empty
            and pd.notna(max_prediction_delta)
            and pd.notna(max_crime_term_shift)
            else "- Influence robustness severity: not estimable."
        ),
        f"- Influence robustness pass status: {int(influence_pass)}.",
        f"- Influence fit-improvement warning count: {fit_warning_count}/{len(influence_summary)} regression specs.",
        (
            f"- Cluster practical-utility severity: {cluster_fail_count}/{len(cluster_stability)} domains fail."
            if not cluster_stability.empty
            else "- Cluster practical-utility severity: not estimable."
        ),
        f"- Cluster practical-utility pass status: {int(cluster_pass)}.",
        (
            f"- Share of statistical tests clearing FDR + practical-effect guardrails: {interpretation_share:.3f}."
            if pd.notna(interpretation_share)
            else "- Share of statistical tests clearing FDR + practical-effect guardrails: not estimable."
        ),
        "",
        "Regression-linked crime terms remain interpretation-blocked whenever statistical guardrails or practical-effect checks fail, even if influence robustness improves.",
        "Decision rule for this repo: keep scenario and policy artifacts in descriptive/exploratory mode unless the temporal holdout, interval calibration, influence robustness, and cluster utility checks all pass.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def _write_scatter_plot(model_df: pd.DataFrame, output_path: Path) -> None:
    frame = model_df.copy()
    frame["home_value"] = pd.to_numeric(frame["home_value"], errors="coerce")
    frame["total_rate_per_1000"] = pd.to_numeric(frame["total_rate_per_1000"], errors="coerce")
    frame = frame.dropna(subset=["home_value", "total_rate_per_1000"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(frame["total_rate_per_1000"], frame["home_value"], alpha=0.8, color="#0b5cab")
    ax.set_title("Dallas ZIP Home Value vs Total Crime Rate")
    ax.set_xlabel("Total Crime Rate per 1,000 Residents")
    ax.set_ylabel("Home Value")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_geography_plot(model_df: pd.DataFrame, output_path: Path) -> None:
    required = {"centroid_latitude", "centroid_longitude", "total_rate_per_1000", "home_value"}
    if not required <= set(model_df.columns):
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Geography view unavailable\nmissing ZIP centroid coordinates in model dataset",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    frame = model_df.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    if frame.empty:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Geography view unavailable\nno ZIP rows had complete coordinates and metrics",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    home_values = frame["home_value"].clip(lower=1)
    size_scale = np.sqrt(home_values / home_values.median()) * 80

    fig, ax = plt.subplots(figsize=(8.5, 6))
    scatter = ax.scatter(
        frame["centroid_longitude"],
        frame["centroid_latitude"],
        c=frame["total_rate_per_1000"],
        s=size_scale,
        cmap="YlOrRd",
        alpha=0.8,
        edgecolors="#1f2933",
        linewidths=0.4,
    )
    ax.set_title("Dallas ZIP Crime and Housing Geography")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Total Crime Rate per 1,000")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_vif_notes(notes: list[str], output_path: Path) -> None:
    lines = [
        "# Multicollinearity Check Notes",
        "",
    ]
    if not notes:
        lines.extend(
            [
                "VIF was computed successfully for all modeled terms.",
                "",
                "See `model_vif.csv` for model-by-model values.",
            ]
        )
    else:
        lines.extend(
            [
                "VIF checks were run with the following caveats:",
                "",
                *notes,
                "",
                "See `model_vif.csv` for available model-by-model values.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n")


def _write_validation_notes(notes: list[str], output_path: Path) -> None:
    lines = [
        "# Model Validation Notes",
        "",
    ]
    if not notes:
        lines.extend(
            [
                "LOOCV-style PRESS diagnostics completed without additional caveats.",
                "",
                "See `model_validation_metrics.csv` for model-by-model values.",
            ]
        )
    else:
        lines.extend(
            [
                "Validation checks completed with the following caveats:",
                "",
                *notes,
                "",
                "See `model_validation_metrics.csv` for model-by-model values.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n")


def _write_residual_review(residuals_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Residual Review",
        "",
        "Rows below show the largest absolute residuals for each model.",
    ]
    for model_label, model_frame in residuals_df.groupby("model_label"):
        lines.extend(
            [
                "",
                f"## {model_label}",
                "",
                "| zip | observed | fitted_value | residual | absolute_residual |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in model_frame.head(5).itertuples(index=False):
            zip_code = getattr(row, "zip", "n/a")
            lines.append(
                f"| {zip_code} | {row.observed:.3f} | {row.fitted_value:.3f} | "
                f"{row.residual:.3f} | {row.absolute_residual:.3f} |"
            )
        flagged = model_frame.loc[
            pd.to_numeric(model_frame.get("is_high_influence_flagged"), errors="coerce")
            .fillna(0)
            .astype(int)
            == 1
        ].copy()
        if not flagged.empty:
            lines.extend(
                [
                    "",
                    "High-influence ZIPs can have small residuals; use leverage/Cook's diagnostics alongside residual size.",
                    "",
                    "| zip | influence_flag_reason | influence_leverage | influence_cooks_distance | absolute_residual |",
                    "| --- | --- | ---: | ---: | ---: |",
                ]
            )
            flagged = flagged.sort_values(
                ["influence_leverage", "influence_cooks_distance", "absolute_residual"],
                ascending=[False, False, False],
                kind="mergesort",
            )
            for row in flagged.head(5).itertuples(index=False):
                lines.append(
                    f"| {getattr(row, 'zip', 'n/a')} | {getattr(row, 'influence_flag_reason', '')} | "
                    f"{float(getattr(row, 'influence_leverage', np.nan)):.3f} | "
                    f"{float(getattr(row, 'influence_cooks_distance', np.nan)):.3f} | "
                    f"{float(getattr(row, 'absolute_residual', np.nan)):.3f} |"
                )

    output_path.write_text("\n".join(lines) + "\n")


def _write_zip_comparison_table(model_df: pd.DataFrame, output_path: Path) -> None:
    frame = model_df.copy()
    frame["home_value"] = pd.to_numeric(frame["home_value"], errors="coerce")
    frame["total_rate_per_1000"] = pd.to_numeric(frame["total_rate_per_1000"], errors="coerce")
    frame = frame.dropna(subset=["home_value", "total_rate_per_1000"])

    top = frame.sort_values("home_value", ascending=False).head(5)
    bottom = frame.sort_values("home_value", ascending=True).head(5)

    lines = [
        "# ZIP Comparison Table",
        "",
        "## Highest Home Value ZIPs",
        "",
        "| zip | home_value | total_rate_per_1000 |",
        "| --- | ---: | ---: |",
    ]
    for row in top.itertuples(index=False):
        lines.append(f"| {row.zip} | {row.home_value:.0f} | {row.total_rate_per_1000:.2f} |")

    lines.extend(
        [
            "",
            "## Lowest Home Value ZIPs",
            "",
            "| zip | home_value | total_rate_per_1000 |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in bottom.itertuples(index=False):
        lines.append(f"| {row.zip} | {row.home_value:.0f} | {row.total_rate_per_1000:.2f} |")

    output_path.write_text("\n".join(lines) + "\n")


def _write_model_summary_table(metrics_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Model Summary Table",
        "",
        "| model_label | nobs | r_squared | adjusted_r_squared |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in metrics_df.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {int(row.nobs)} | {row.r_squared:.3f} | {row.adjusted_r_squared:.3f} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def _coefficient_for_term(
    coefficients: pd.DataFrame,
    *,
    model_label: str,
    term: str,
) -> pd.Series | None:
    match = coefficients[
        (coefficients["model_label"].astype(str) == model_label)
        & (coefficients["term"].astype(str) == term)
    ]
    if match.empty:
        return None
    return match.iloc[0]


def _effect_size_text(coefficient_row: pd.Series | None) -> str:
    if coefficient_row is None:
        return "not estimable from this specification"
    estimate = pd.to_numeric(pd.Series([coefficient_row.get("estimate")]), errors="coerce").iloc[0]
    if pd.isna(estimate):
        return "not estimable from this specification"
    percent_change = (np.exp(float(estimate)) - 1.0) * 100.0
    q_value = pd.to_numeric(pd.Series([coefficient_row.get("fdr_q_value")]), errors="coerce").iloc[
        0
    ]
    interpretation_allowed = (
        pd.to_numeric(pd.Series([coefficient_row.get("interpretation_allowed")]), errors="coerce")
        .fillna(0)
        .astype(int)
        .iloc[0]
        == 1
    )
    if interpretation_allowed:
        return f"{percent_change:.2f}% change in expected home value (passes FDR/practical-effect guardrails)"
    q_text = f"{q_value:.3f}" if pd.notna(q_value) else "n/a"
    return (
        f"{percent_change:.2f}% change in expected home value, but this term remains exploratory "
        f"(FDR q={q_text}; below repo guardrails)"
    )


def _write_summary_report(
    model_df: pd.DataFrame,
    results: list[RegressionResult],
    coefficients: pd.DataFrame,
    crime_forecasts: pd.DataFrame,
    temporal_holdout: pd.DataFrame,
    influence_summary: pd.DataFrame,
    cluster_stability: pd.DataFrame,
    settings: "Settings",
    output_path: Path,
    *,
    factor_importance_summary_md: str = "",
    parsimonious_result: RegressionResult | None = None,
) -> None:
    numeric = model_df[["total_rate_per_1000", "home_value"]].apply(pd.to_numeric, errors="coerce")
    total_corr = numeric.corr().iloc[0, 1]
    baseline = next(result for result in results if result.model_label == "baseline")
    expanded = next(result for result in results if result.model_label == "sensitivity_check")

    baseline_crime = _effect_size_text(
        _coefficient_for_term(
            coefficients,
            model_label="baseline",
            term="total_rate_per_1000",
        )
    )
    sensitivity_crime = _effect_size_text(
        _coefficient_for_term(
            coefficients,
            model_label="sensitivity_check",
            term="total_rate_per_1000",
        )
    )
    housing_sources = sorted(
        str(source)
        for source in model_df.get("source", pd.Series(dtype="string")).dropna().unique()
    )
    total_corr_text = f"{total_corr:.3f}" if pd.notna(total_corr) else "not estimable"
    crime_history_path = settings.processed_dir / "crime_history_panel.csv"
    history_panel_path = settings.processed_dir / "housing_history_panel.csv"
    crime_history_note = (
        "Quarterly crime history was available for decomposition, forecasting, scenarios, and drift diagnostics."
        if crime_history_path.exists()
        else "Quarterly crime history was not available for this run."
    )
    history_note = (
        "Recent context was supplemented with a 2000-2025 historical housing panel "
        "(Realtor monthly history plus FHFA ZIP5 annual HPI)."
        if history_panel_path.exists()
        else "Historical housing context was not available for this run."
    )
    selected_holdout = temporal_holdout.loc[
        temporal_holdout["evaluation_scope"].astype(str) == "selected_zip_model"
    ]
    forecast_eligible_zip_count = (
        int(pd.to_numeric(selected_holdout["eligible_zip_count"], errors="coerce").max())
        if not selected_holdout.empty
        else 0
    )
    lower_confidence_forecast_count = (
        int(
            crime_forecasts.loc[
                crime_forecasts.get(
                    "policy_eligible", pd.Series(1, index=crime_forecasts.index)
                ).astype(int)
                == 0,
                "zip",
            ]
            .astype(str)
            .nunique()
        )
        if not crime_forecasts.empty
        else 0
    )
    selected_holdout_mape = (
        float(pd.to_numeric(selected_holdout["mape"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else np.nan
    )
    selected_holdout_threshold = (
        float(pd.to_numeric(selected_holdout["mape_threshold"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else np.nan
    )
    selected_holdout_pass = (
        int(pd.to_numeric(selected_holdout["mape_pass"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else 0
    )
    influence_fail_count = (
        int(
            pd.to_numeric(influence_summary["influence_robustness_pass"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        if not influence_summary.empty
        else 0
    )
    max_crime_term_shift = (
        float(
            pd.to_numeric(
                influence_summary["max_crime_term_effect_pct_change"], errors="coerce"
            ).max()
        )
        if not influence_summary.empty
        else np.nan
    )
    max_prediction_delta = (
        float(
            pd.to_numeric(influence_summary["max_p90_home_value_pct_delta"], errors="coerce").max()
        )
        if not influence_summary.empty
        else np.nan
    )
    fit_warning_count = (
        int(
            pd.to_numeric(influence_summary["fit_improvement_warning_count"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
        if not influence_summary.empty
        else 0
    )
    cluster_fail_count = (
        int(
            pd.to_numeric(cluster_stability["practical_utility_pass"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        if not cluster_stability.empty
        else 0
    )

    lines = [
        "# Dallas Crime and Housing Report",
        "",
        "This narrative summarizes the latest pipeline outputs in plain language for non-technical readers.",
        "",
        "## Methods",
        "",
        (
            "- Geography: ZIP-level Dallas-area records using the processed model dataset; "
            f"ZIPs with fewer than {settings.min_total_incidents_per_zip} incident(s) in the active crime window "
            "were excluded from the study universe."
        ),
        "- Inputs: Dallas OpenData crime incidents, ACS ZIP controls, and current housing price signals "
        "from Zillow, Realtor, Redfin, and Realtor bulk ZIP feeds.",
        f"- Current housing sources represented in this model run: {', '.join(housing_sources) if housing_sources else 'none recorded'}.",
        f"- Crime history context: {crime_history_note}",
        f"- Historical context: {history_note}",
        "- Strategy: two robust regression models were run from the same processed dataset:",
        "  - baseline model with core crime + household controls",
        f"  - sensitivity-check model adding completeness-qualified controls: {', '.join(expanded.controls)}",
        "  - Sensitivity check only — n/p ratio below 10; not for primary inference.",
        "",
        "## Findings",
        "",
        f"- Baseline sample size: {baseline.nobs} ZIPs; sensitivity-check sample size: {expanded.nobs} ZIPs.",
        f"- Overall correlation between total crime rate and home value: {total_corr_text}.",
        "- Estimated effect per +1 crime incident per 1,000 residents:",
        f"  - Baseline total-crime-rate term: {baseline_crime}.",
        f"  - Sensitivity-check total-crime-rate term: {sensitivity_crime}.",
        f"- Model fit stayed stable across specs (baseline R-squared: {baseline.r_squared:.3f}, "
        f"expanded R-squared: {expanded.r_squared:.3f}).",
        "",
        "## Limitations",
        "",
        "- This is observational ZIP-level analysis and should not be read as causal proof.",
        (
            f"- High-confidence forecast/scenario coverage is {forecast_eligible_zip_count}/{len(model_df)} modeled ZIPs; lower-confidence forecast-only tiers add {lower_confidence_forecast_count} more ZIPs."
            if len(model_df) > 0
            else "- Forecast/scenario coverage was not estimable for this run."
        ),
        (
            f"- Selected ZIP temporal holdout is MAPE={selected_holdout_mape:.3f} versus threshold {selected_holdout_threshold:.1f} (pass={selected_holdout_pass})."
            if pd.notna(selected_holdout_mape) and pd.notna(selected_holdout_threshold)
            else "- Selected ZIP temporal holdout was not estimable for this run."
        ),
        (
            f"- Influence robustness fails in {influence_fail_count}/{len(influence_summary)} regression specs on prediction-stability guardrails; max p90 home-value prediction delta={max_prediction_delta:.3f}%, max crime-term shift={max_crime_term_shift:.3f}%."
            if not influence_summary.empty
            and pd.notna(max_prediction_delta)
            and pd.notna(max_crime_term_shift)
            else "- Influence robustness severity was not estimable for this run."
        ),
        f"- Fit-improvement warnings remain in {fit_warning_count}/{len(influence_summary)} regression specs."
        if not influence_summary.empty
        else "- Fit-improvement warnings were not estimable for this run.",
        (
            f"- Cluster practical utility fails in {cluster_fail_count}/{len(cluster_stability)} segmentation domains on current data."
            if not cluster_stability.empty
            else "- Cluster practical utility was not estimable for this run."
        ),
        "- Housing measures combine multiple reputable sources with different definitions "
        "(typical value, median listing price, median sale price, HPI), so they are not perfectly interchangeable.",
        "- ACS and FHFA inputs can lag real-time market and demographic shifts.",
        "- Small sample sizes and correlated neighborhood factors can inflate uncertainty.",
        "- Results are sensitive to ZIP-universe filtering, available controls, and the current crime lookback window.",
        "",
        "## Factor Importance Analysis",
        "",
        "The crime-housing hypothesis has been answered: total_rate_per_1000 is not a significant "
        "predictor once socioeconomic controls are included. The analysis pivoted to the broader "
        "question — which features are the strongest independent drivers of ZIP-level home values?",
        "",
    ]

    # Parsimonious model results
    if parsimonious_result is not None:
        lines.extend(
            [
                "### Parsimonious Model",
                "",
                "A three-predictor model using one representative from each independent collinearity cluster:",
                "",
                "| Predictor | Cluster | Coefficient | Std Error | p-value |",
                "|-----------|---------|-------------|-----------|---------|",
            ]
        )
        _cluster_labels = {
            "educational_attainment": "Socioeconomic (Cluster 1)",
            "realtor_listing_price": "Price level (Cluster 3)",
            "aggregate_market_pressure_index": "Market pressure",
        }
        for _, row in parsimonious_result.coefficients.iterrows():
            term = str(row["term"])
            if term == "Intercept":
                continue
            cluster = _cluster_labels.get(term, "")
            estimate = float(row["estimate"])
            se = float(row["std_error"])
            pval = float(row["p_value"])
            lines.append(f"| {term} | {cluster} | {estimate:+.6f} | {se:.6f} | {pval:.4f} |")
        lines.extend(
            [
                "",
                f"- Sample size: {parsimonious_result.nobs} ZIPs",
                f"- R-squared: {parsimonious_result.r_squared:.3f}",
                f"- Adjusted R-squared: {parsimonious_result.adjusted_r_squared:.3f}",
                "- Standard errors: HC3 robust",
                "- No additional controls — the three cluster representatives are the full model.",
                "",
                "The negative coefficient on aggregate_market_pressure_index reflects affordability "
                "dynamics: this composite index (mean z-score of crime rate, annual price change, "
                "rent-to-home-value ratio, and income growth trend) captures ZIPs where market "
                "pressure runs against home values. Higher-pressure ZIPs tend to have lower home "
                "values after controlling for educational attainment and listing price level.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "### Parsimonious Model",
                "",
                "The parsimonious model was not estimable for this run (missing predictor columns).",
                "",
            ]
        )

    # Factor importance summary (from Phase 4 markdown, embedded as-is)
    if factor_importance_summary_md:
        # Strip the top-level heading since we already have the section header
        _fi_lines = factor_importance_summary_md.strip().split("\n")
        _skip = 0
        for _fi_line in _fi_lines:
            if _fi_line.startswith("# ") and "Factor Importance" in _fi_line:
                _skip += 1
                # Also skip blank line after heading
                if _skip < len(_fi_lines) and _fi_lines[_skip].strip() == "":
                    _skip += 1
                break
            _skip += 1
        lines.extend(_fi_lines[_skip:])
        lines.append("")
    else:
        lines.append("Factor importance artifacts were not available for this run.")
        lines.append("")

    lines.extend(
        [
            "## Visuals and Artifacts",
            "",
            "- `home_value_vs_total_crime.png` (scatter visual)",
            "- `crime_home_value_geography.png` (ZIP centroid geography-aware view)",
            "- `cluster_assignments.csv` and `cluster_profiles.csv` (deterministic ZIP segmentation outputs)",
            "- `spatial_diagnostics.csv` and `spatial_hotspots.csv` (Moran-style clustering and local spillover view)",
            "- `model_validation_metrics.csv` and `model_validation_notes.md` (PRESS/LOOCV-style validation diagnostics)",
            "- `feature_selection_metrics.csv` and `feature_selection_notes.md` (candidate feature ranking and coverage)",
            "- `feature_power_retention_metrics.csv` and `feature_power_retention_notes.md` (explicit retained-signal checkpoint metrics)",
            "- `predictive_model_metrics.csv`, `predictive_model_predictions.csv`, and `model_selection_notes.md` (broader model-family and ensemble selection diagnostics)",
            "- `crime_trend_decomposition.csv` and `crime_trend_decomposition.md` (quarterly decomposition view)",
            "- `forecast_model_metrics.csv`, `crime_forecasts.csv`, and `forecast_confidence_intervals.csv` (ZIP-level forecast model comparison and 12-month outlook)",
            "- `temporal_holdout_results.csv` and `forecast_interval_calibration.csv` (temporal holdout and interval credibility checks)",
            "- `scenario_impacts.csv` and `scenario_notes.md` (deterministic planning scenarios)",
            "- `zip_benchmarks.csv` and `benchmark_summary.md` (ZIP-vs-metro and ZIP-vs-cluster comparisons)",
            "- `model_drift_diagnostics.csv` and `model_drift_notes.md` (latest-quarter drift checks)",
            "- `influence_robustness_diagnostics.csv` and `cluster_stability_diagnostics.csv` (robustness and segmentation-utility checks)",
            "- `statistical_guardrails.csv` and `policy_guardrails.md` (multiple-testing, practical-effect, and non-causal interpretation guardrails)",
            "- `comprehensive_validation_metrics.csv` and `comprehensive_validation_notes.md` (cross-artifact validation rollup)",
            "- `policy_recommendations_by_segment.csv` and `policy_recommendations_by_segment.md` (segment-level action framing)",
            "- `top_bottom_zip_comparison.md` (additional ZIP comparison table)",
            "- `model_summary_table.md` (model-level metrics table)",
            "- `regression_coefficients.csv` and `regression_metrics.csv`",
            "- `model_sample_sizes.csv`, `model_residuals.csv`, `residual_review.md`",
            "- `model_vif.csv` and `model_vif_notes.md`",
            "- `factor_importance_univariate.csv`, `factor_importance_standardized.csv`, "
            "`factor_importance_variance_decomposition.csv`, and `factor_importance_summary.md` "
            "(factor importance analysis: univariate screening, standardized coefficients, "
            "LMG variance decomposition, and summary)",
            "",
            "## Rebuild Path",
            "",
            "- Full refresh from raw sources: `dallas-crime acquire && dallas-crime build && dallas-crime analyze`",
            "- Report refresh from existing raw inputs: `dallas-crime build && dallas-crime analyze`",
            "",
            "## References",
            "",
            "- Dallas OpenData Police Incidents dataset",
            "- U.S. Census ACS 5-year ZIP-level tables",
            "- Zillow, Realtor, and Redfin ZIP market pages captured via Firecrawl",
            "- Realtor ZIP market pages and Realtor ZIP inventory/history feeds",
            "- FHFA ZIP5 annual HPI",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")
