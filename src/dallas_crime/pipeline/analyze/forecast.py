"""Forecasting, temporal analysis, holdout validation, scenarios, benchmarks, and drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from dallas_crime.pipeline.analyze.core import (
    DRIFT_HISTORY_MIN_QUARTERS,
    DRIFT_MIN_COMPLETENESS,
    FORECAST_HISTORY_MIN_QUARTERS,
    FORECAST_LIMITED_HISTORY_MIN_QUARTERS,
    FORECAST_MODEL_FALLBACK_ORDER,
    FORECAST_MODELS,
    SCENARIO_MULTIPLIERS,
    TEMPORAL_HOLDOUT_QUARTERS,
    _safe_ratio,
)


def _prepare_temporal_analysis_inputs(
    crime_history_panel: pd.DataFrame,
    *,
    modeled_zips: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    columns = [
        "zip",
        "observed_quarters",
        "expected_quarters",
        "overall_completeness_ratio",
        "trailing_contiguous_quarters",
        "forecast_gate_pass",
        "drift_gate_pass",
        "first_period_start",
        "latest_period_start",
    ]
    notes: list[str] = []
    required = {"zip", "period_start", "total_rate_per_1000"}
    if crime_history_panel.empty:
        notes.append("Crime history panel was not available for temporal credibility checks.")
        return pd.DataFrame(columns=columns), {}, notes
    if not required <= set(crime_history_panel.columns):
        notes.append(
            "Crime history panel was missing one or more required temporal columns "
            f"({', '.join(sorted(required))})."
        )
        return pd.DataFrame(columns=columns), {}, notes

    panel = crime_history_panel.copy()
    panel["zip"] = panel["zip"].astype("string")
    panel["period_start"] = pd.to_datetime(panel["period_start"], errors="coerce")
    panel["total_rate_per_1000"] = pd.to_numeric(panel["total_rate_per_1000"], errors="coerce")
    panel = panel.dropna(subset=["zip", "period_start", "total_rate_per_1000"]).copy()
    if modeled_zips is not None:
        panel = panel.loc[panel["zip"].isin(sorted(modeled_zips))].copy()
    if panel.empty:
        notes.append("No modeled ZIP had complete quarterly crime history rows after filtering.")
        return pd.DataFrame(columns=columns), {}, notes

    temporal_rows: list[dict[str, object]] = []
    trailing_series: dict[str, pd.DataFrame] = {}
    for zip_code, zip_frame in panel.groupby("zip", sort=True):
        ordered = (
            zip_frame.sort_values("period_start", kind="mergesort", ignore_index=True)
            .drop_duplicates(subset=["period_start"], keep="last")
            .reset_index(drop=True)
        )
        periods = ordered["period_start"].dt.to_period("Q")
        ordinals = periods.astype(int)
        if ordered.empty:
            continue

        diffs = ordinals.diff()
        trailing_count = 1
        for diff in diffs.iloc[1:][::-1]:
            if diff == 1:
                trailing_count += 1
            else:
                break
        trailing_frame = ordered.iloc[-trailing_count:].copy().reset_index(drop=True)
        observed_quarters = int(len(ordered))
        expected_quarters = int(ordinals.iloc[-1] - ordinals.iloc[0] + 1)
        temporal_rows.append(
            {
                "zip": str(zip_code),
                "observed_quarters": observed_quarters,
                "expected_quarters": expected_quarters,
                "overall_completeness_ratio": _safe_ratio(observed_quarters, expected_quarters),
                "trailing_contiguous_quarters": int(len(trailing_frame)),
                "forecast_gate_pass": int(len(trailing_frame) >= FORECAST_HISTORY_MIN_QUARTERS),
                "drift_gate_pass": int(len(trailing_frame) >= DRIFT_HISTORY_MIN_QUARTERS),
                "first_period_start": ordered["period_start"].iloc[0],
                "latest_period_start": ordered["period_start"].iloc[-1],
            }
        )
        trailing_series[str(zip_code)] = trailing_frame

    summary = pd.DataFrame.from_records(temporal_rows, columns=columns)
    if summary.empty:
        notes.append("No ZIP-level quarterly histories were usable after temporal filtering.")
        return summary, trailing_series, notes

    forecast_eligible = int(summary["forecast_gate_pass"].sum())
    drift_eligible = int(summary["drift_gate_pass"].sum())
    gated_out = int(len(summary) - forecast_eligible)
    notes.append(
        f"Forecast gating kept {forecast_eligible} modeled ZIPs with >= {FORECAST_HISTORY_MIN_QUARTERS} "
        f"trailing contiguous quarters and excluded {gated_out} ZIPs."
    )
    notes.append(
        f"Drift gating kept {drift_eligible} modeled ZIPs with >= {DRIFT_HISTORY_MIN_QUARTERS} "
        "trailing contiguous quarters."
    )
    return summary.sort_values("zip", ignore_index=True), trailing_series, notes


def _walk_forward_forecast_metrics(
    history: np.ndarray, *, model_name: str
) -> dict[str, float] | None:
    errors: list[float] = []
    absolute_errors: list[float] = []
    apes: list[float] = []
    for index in range(1, len(history)):
        prediction = _predict_with_model(history[:index], model_name, horizon=1)
        actual = float(history[index])
        if pd.isna(prediction) or pd.isna(actual):
            continue
        error = float(actual - prediction)
        errors.append(error)
        absolute_errors.append(abs(error))
        if actual > 0:
            apes.append(abs(error) / actual * 100.0)

    if not errors:
        return None

    abs_errors = np.asarray(absolute_errors, dtype=float)
    signed_errors = np.asarray(errors, dtype=float)
    return {
        "evaluation_points": float(len(errors)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(np.square(signed_errors)))),
        "mape": float(np.mean(apes)) if apes else np.nan,
        "error_std": float(np.std(signed_errors, ddof=0)),
        "p75_absolute_error": float(np.quantile(abs_errors, 0.75)),
    }


def _forecast_interval_bounds(
    *,
    prediction: float,
    interval_level: int,
    error_scale: float,
    history: np.ndarray,
    horizon: int,
) -> tuple[float, float]:
    z_score = 1.2816 if interval_level == 80 else 1.96
    clean_history = pd.Series(history, dtype=float).dropna().to_numpy(dtype=float)
    history_scale = (
        float(np.quantile(clean_history, 0.75)) if clean_history.size else abs(prediction)
    )
    history_max = float(np.max(clean_history)) if clean_history.size else abs(prediction)
    upper_cap = max(history_max * 2.0, abs(prediction) * 20.0, prediction + 0.01)
    base_scale = max(error_scale, history_scale * 0.05, abs(prediction) * 0.05, 0.01)
    raw_width = float(z_score * base_scale * np.sqrt(horizon))
    capped_width = min(raw_width, upper_cap - prediction)
    floor_width = min(
        max(history_scale * 0.02, abs(prediction) * 0.02, 0.01), upper_cap - prediction
    )
    width = max(capped_width, floor_width)
    lower_bound = max(float(prediction) - width, 0.0)
    upper_bound = min(float(prediction) + width, upper_cap)
    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 0.01
    return float(lower_bound), float(upper_bound)


def _predict_with_model(history: np.ndarray, model_name: str, *, horizon: int) -> float:
    if len(history) == 0:
        return np.nan

    if model_name == "naive_last":
        return float(history[-1])
    if model_name == "seasonal_naive_4q":
        if len(history) < 4:
            return np.nan
        seasonal_index = -4 + ((horizon - 1) % 4)
        return float(history[seasonal_index])
    if model_name == "moving_average_4":
        if len(history) < 4:
            return np.nan
        return float(np.mean(history[-4:]))
    if model_name == "linear_trend":
        if len(history) < 2:
            return np.nan
        slope, intercept = np.polyfit(np.arange(len(history), dtype=float), history, 1)
        return float(intercept + (slope * (len(history) + horizon - 1)))
    raise KeyError(f"unknown forecast model: {model_name}")


def _select_forecast_model(
    zip_metrics: pd.DataFrame,
    *,
    history: np.ndarray,
    allowed_models: tuple[str, ...] | None = None,
) -> str | None:
    eligible_models = tuple(
        model_name
        for model_name in (allowed_models or FORECAST_MODEL_FALLBACK_ORDER)
        if model_name in FORECAST_MODELS
    )
    if not zip_metrics.empty:
        ranked = zip_metrics.copy()
        if eligible_models:
            ranked = ranked.loc[ranked["model_name"].astype(str).isin(set(eligible_models))].copy()
        if ranked.empty:
            ranked = zip_metrics.copy()
        ranked["fallback_rank"] = ranked["model_name"].map(FORECAST_MODEL_FALLBACK_ORDER.index)
        ranked = ranked.sort_values(
            ["rmse", "mae", "fallback_rank"],
            ascending=[True, True, True],
            na_position="last",
            kind="mergesort",
        )
        return str(ranked.iloc[0]["model_name"])

    for model_name in eligible_models or FORECAST_MODEL_FALLBACK_ORDER:
        prediction = _predict_with_model(history, model_name, horizon=1)
        if pd.notna(prediction):
            return model_name
    return None


def _build_trend_decomposition_artifacts(
    crime_history_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "scope",
        "zip",
        "period_start",
        "observed_total_rate_per_1000",
        "trend_total_rate_per_1000",
        "seasonal_total_rate_per_1000",
        "residual_total_rate_per_1000",
        "history_points",
    ]
    required = {"zip", "period_start", "total_rate_per_1000"}
    notes: list[str] = []
    if crime_history_panel.empty:
        notes.append("Crime history panel was not available, so decomposition artifacts are empty.")
        return pd.DataFrame(columns=columns), notes
    if not required <= set(crime_history_panel.columns):
        notes.append(
            "Crime history panel was missing one or more required columns "
            f"({', '.join(sorted(required))}); decomposition artifacts are empty."
        )
        return pd.DataFrame(columns=columns), notes

    panel = crime_history_panel.copy()
    panel["zip"] = panel["zip"].astype("string")
    panel["period_start"] = pd.to_datetime(panel["period_start"], errors="coerce")
    panel["total_rate_per_1000"] = pd.to_numeric(panel["total_rate_per_1000"], errors="coerce")
    panel = panel.dropna(subset=["zip", "period_start", "total_rate_per_1000"]).copy()
    if panel.empty:
        notes.append("Crime history panel had no complete quarterly total-rate rows to decompose.")
        return pd.DataFrame(columns=columns), notes

    series_frames: list[tuple[str, str | None, pd.Series]] = []
    metro_series = (
        panel.groupby("period_start")["total_rate_per_1000"].mean().sort_index(kind="mergesort")
    )
    series_frames.append(("metro", None, metro_series))
    for zip_code, zip_frame in panel.groupby("zip", sort=True):
        zip_series = zip_frame.sort_values("period_start", kind="mergesort").set_index(
            "period_start"
        )["total_rate_per_1000"]
        series_frames.append(("zip", str(zip_code), zip_series))

    rows: list[dict[str, object]] = []
    for scope, zip_code, series in series_frames:
        history_points = int(series.notna().sum())
        label = "metro aggregate" if zip_code is None else f"ZIP {zip_code}"
        if history_points < 8:
            notes.append(f"{label}: fewer than 8 quarterly observations were available.")
            continue

        try:
            decomposition = seasonal_decompose(
                series,
                model="additive",
                period=4,
                extrapolate_trend="freq",
            )
        except ValueError as exc:
            notes.append(f"{label}: decomposition failed ({exc}).")
            continue

        for period_start, observed in series.items():
            rows.append(
                {
                    "scope": scope,
                    "zip": zip_code,
                    "period_start": period_start,
                    "observed_total_rate_per_1000": float(observed),
                    "trend_total_rate_per_1000": float(decomposition.trend.loc[period_start])
                    if pd.notna(decomposition.trend.loc[period_start])
                    else np.nan,
                    "seasonal_total_rate_per_1000": float(decomposition.seasonal.loc[period_start])
                    if pd.notna(decomposition.seasonal.loc[period_start])
                    else np.nan,
                    "residual_total_rate_per_1000": float(decomposition.resid.loc[period_start])
                    if pd.notna(decomposition.resid.loc[period_start])
                    else np.nan,
                    "history_points": history_points,
                }
            )

    decomposition_df = pd.DataFrame.from_records(rows, columns=columns)
    if decomposition_df.empty and not notes:
        notes.append("No crime-rate series met the minimum history requirement for decomposition.")
    if decomposition_df.empty:
        return decomposition_df, notes
    return decomposition_df.sort_values(
        ["scope", "zip", "period_start"],
        kind="mergesort",
        ignore_index=True,
    ), notes


def _build_forecast_artifacts(
    temporal_summary: pd.DataFrame,
    temporal_series: dict[str, pd.DataFrame],
    *,
    temporal_holdout: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    metric_columns = [
        "model_name",
        "evaluation_points",
        "zip_count",
        "mae",
        "rmse",
        "mape",
        "selected_zip_count",
        "eligible_zip_count",
        "gated_out_zip_count",
        "minimum_history_quarters",
        "temporal_holdout_mape",
        "temporal_holdout_pass",
        "temporal_holdout_rank",
    ]
    forecast_columns = [
        "zip",
        "forecast_period_start",
        "horizon_quarters",
        "selected_model",
        "forecast_tier",
        "policy_eligible",
        "selection_rule",
        "history_points",
        "trailing_contiguous_quarters",
        "overall_completeness_ratio",
        "predicted_total_rate_per_1000",
    ]
    interval_columns = [
        "zip",
        "forecast_period_start",
        "horizon_quarters",
        "selected_model",
        "forecast_tier",
        "policy_eligible",
        "interval_level",
        "forecast_value",
        "lower_bound",
        "upper_bound",
        "interval_width",
        "upper_to_forecast_ratio",
    ]
    notes: list[str] = []
    if temporal_summary.empty or not temporal_series:
        notes.append("Temporal summary inputs were unavailable, so forecast artifacts are empty.")
        return (
            pd.DataFrame(columns=metric_columns),
            pd.DataFrame(columns=forecast_columns),
            pd.DataFrame(columns=interval_columns),
            notes,
        )

    eligible = temporal_summary.loc[
        pd.to_numeric(temporal_summary["forecast_gate_pass"], errors="coerce").fillna(0).astype(int)
        == 1
    ].copy()
    limited_history = temporal_summary.loc[
        pd.to_numeric(temporal_summary["trailing_contiguous_quarters"], errors="coerce")
        .fillna(0)
        .between(FORECAST_LIMITED_HISTORY_MIN_QUARTERS, FORECAST_HISTORY_MIN_QUARTERS - 1)
    ].copy()
    carry_forward = temporal_summary.loc[
        pd.to_numeric(temporal_summary["trailing_contiguous_quarters"], errors="coerce")
        .fillna(0)
        .between(1, FORECAST_LIMITED_HISTORY_MIN_QUARTERS - 1)
    ].copy()
    if eligible.empty:
        notes.append(
            "No modeled ZIP met the trailing-contiguous quarterly history gate for forecasting "
            f"({FORECAST_HISTORY_MIN_QUARTERS} quarters)."
        )
        return (
            pd.DataFrame(columns=metric_columns),
            pd.DataFrame(columns=forecast_columns),
            pd.DataFrame(columns=interval_columns),
            notes,
        )
    notes.append(
        f"Forecasts use only modeled ZIPs with >= {FORECAST_HISTORY_MIN_QUARTERS} trailing contiguous quarters "
        f"({len(eligible)} of {len(temporal_summary)} modeled ZIPs passed)."
    )
    if not limited_history.empty or not carry_forward.empty:
        notes.append(
            "Additional forecast-only coverage is emitted for lower-history ZIPs using explicit "
            "reduced-confidence tiers; those rows are excluded from interval calibration and scenarios."
        )
    holdout_lookup: dict[str, dict[str, float]] = {}
    allowed_models: tuple[str, ...] = ()
    if temporal_holdout is not None and not temporal_holdout.empty:
        candidate_holdout = temporal_holdout.loc[
            temporal_holdout["evaluation_scope"].astype(str) == "candidate_model"
        ].copy()
        if not candidate_holdout.empty:
            candidate_holdout["temporal_holdout_rank"] = candidate_holdout["mape"].rank(
                method="dense",
                na_option="bottom",
            )
            holdout_lookup = {
                str(row.model_name): {
                    "temporal_holdout_mape": float(row.mape) if pd.notna(row.mape) else np.nan,
                    "temporal_holdout_pass": int(row.mape_pass) if pd.notna(row.mape_pass) else 0,
                    "temporal_holdout_rank": float(row.temporal_holdout_rank)
                    if pd.notna(row.temporal_holdout_rank)
                    else np.nan,
                }
                for row in candidate_holdout.itertuples(index=False)
            }
            allowed_models = tuple(
                model_name
                for model_name in FORECAST_MODELS
                if model_name
                in set(
                    candidate_holdout.loc[
                        pd.to_numeric(candidate_holdout["mape_pass"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        == 1,
                        "model_name",
                    ]
                    .astype(str)
                    .tolist()
                )
            )
    notes.append(
        "Selected ZIP-level forecast families are chosen from the pre-holdout training window used "
        "in temporal holdout validation, then projected forward on the full available history."
    )
    if allowed_models:
        notes.append(
            "ZIP-level model selection is restricted to candidate forecast families that pass aggregate "
            f"temporal holdout ({', '.join(allowed_models)})."
        )
    notes.append(
        "Scenario artifacts remain limited to high-confidence forecasts; lower-history tiers stay "
        "forecast-only and non-policy-eligible."
    )

    per_zip_metric_rows: list[dict[str, object]] = []
    for row in eligible.itertuples(index=False):
        zip_code = str(row.zip)
        ordered = temporal_series.get(zip_code)
        if ordered is None or ordered.empty:
            continue
        values = ordered["total_rate_per_1000"].to_numpy(dtype=float)
        for model_name in FORECAST_MODELS:
            metrics = _walk_forward_forecast_metrics(values, model_name=model_name)
            if metrics is None:
                continue
            per_zip_metric_rows.append(
                {
                    "zip": zip_code,
                    "model_name": model_name,
                    **metrics,
                }
            )

    per_zip_metrics = pd.DataFrame(per_zip_metric_rows)
    selected_models: dict[str, str] = {}
    zip_scales: dict[tuple[str, str], float] = {}
    if not per_zip_metrics.empty:
        per_zip_metrics = per_zip_metrics.sort_values(
            ["zip", "rmse", "mae", "model_name"],
            kind="mergesort",
            ignore_index=True,
        )
        for row in per_zip_metrics.itertuples(index=False):
            zip_scales[(str(row.zip), str(row.model_name))] = max(
                float(row.p75_absolute_error) if pd.notna(row.p75_absolute_error) else 0.0,
                float(row.mae) * 0.75 if pd.notna(row.mae) else 0.0,
                0.01,
            )

    overall_metric_rows: list[dict[str, object]] = []
    overall_scales: dict[str, float] = {}
    for model_name in FORECAST_MODELS:
        model_metrics = per_zip_metrics.loc[per_zip_metrics["model_name"] == model_name].copy()
        if model_metrics.empty:
            overall_metric_rows.append(
                {
                    "model_name": model_name,
                    "evaluation_points": 0,
                    "zip_count": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                    "selected_zip_count": 0,
                    "eligible_zip_count": int(len(eligible)),
                    "gated_out_zip_count": int(len(temporal_summary) - len(eligible)),
                    "minimum_history_quarters": FORECAST_HISTORY_MIN_QUARTERS,
                    "temporal_holdout_mape": np.nan,
                    "temporal_holdout_pass": 0,
                    "temporal_holdout_rank": np.nan,
                }
            )
            overall_scales[model_name] = 0.01
            continue

        overall_metric_rows.append(
            {
                "model_name": model_name,
                "evaluation_points": int(model_metrics["evaluation_points"].sum()),
                "zip_count": int(model_metrics["zip"].nunique()),
                "mae": float(model_metrics["mae"].mean()),
                "rmse": float(model_metrics["rmse"].mean()),
                "mape": float(model_metrics["mape"].mean())
                if model_metrics["mape"].notna().any()
                else np.nan,
                "selected_zip_count": 0,
                "eligible_zip_count": int(len(eligible)),
                "gated_out_zip_count": int(len(temporal_summary) - len(eligible)),
                "minimum_history_quarters": FORECAST_HISTORY_MIN_QUARTERS,
                "temporal_holdout_mape": holdout_lookup.get(model_name, {}).get(
                    "temporal_holdout_mape",
                    np.nan,
                ),
                "temporal_holdout_pass": holdout_lookup.get(model_name, {}).get(
                    "temporal_holdout_pass",
                    0,
                ),
                "temporal_holdout_rank": holdout_lookup.get(model_name, {}).get(
                    "temporal_holdout_rank",
                    np.nan,
                ),
            }
        )
        overall_scales[model_name] = max(
            float(model_metrics["p75_absolute_error"].mean())
            if model_metrics["p75_absolute_error"].notna().any()
            else 0.0,
            float(model_metrics["mae"].mean()) * 0.75
            if model_metrics["mae"].notna().any()
            else 0.0,
            0.01,
        )

    forecast_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    completeness_lookup = temporal_summary.set_index("zip")
    for zip_code in eligible["zip"].astype(str):
        ordered = temporal_series.get(zip_code)
        if ordered is None or ordered.empty:
            continue
        values = ordered["total_rate_per_1000"].to_numpy(dtype=float)
        zip_metrics = (
            per_zip_metrics.loc[per_zip_metrics["zip"] == zip_code].copy()
            if not per_zip_metrics.empty
            else pd.DataFrame()
        )
        selection_history = values[:-TEMPORAL_HOLDOUT_QUARTERS]
        selection_metric_rows: list[dict[str, object]] = []
        selection_scales: dict[str, float] = {}
        if len(selection_history) >= 1:
            for model_name in FORECAST_MODELS:
                metrics = _walk_forward_forecast_metrics(selection_history, model_name=model_name)
                if metrics is None:
                    continue
                selection_metric_rows.append(
                    {
                        "model_name": model_name,
                        "evaluation_points": metrics["evaluation_points"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "mape": metrics["mape"],
                    }
                )
                selection_scales[model_name] = max(
                    float(metrics["p75_absolute_error"]),
                    float(metrics["mae"]) * 0.75,
                    0.01,
                )
        selection_metrics = pd.DataFrame(selection_metric_rows)
        selected_model = _select_forecast_model(
            selection_metrics if not selection_metrics.empty else zip_metrics,
            history=selection_history if len(selection_history) >= 1 else values,
            allowed_models=allowed_models or None,
        )
        if selected_model is None:
            notes.append(
                f"ZIP {zip_code}: no forecast model was applicable for the available history."
            )
            continue

        selected_models[zip_code] = selected_model
        error_scale = selection_scales.get(
            selected_model,
            zip_scales.get((zip_code, selected_model), overall_scales.get(selected_model, 0.01)),
        )
        last_period = pd.Timestamp(ordered["period_start"].iloc[-1]).to_period("Q")
        for horizon in range(1, 5):
            prediction = _predict_with_model(values, selected_model, horizon=horizon)
            if pd.isna(prediction):
                continue
            forecast_period = (last_period + horizon).start_time
            forecast_rows.append(
                {
                    "zip": zip_code,
                    "forecast_period_start": forecast_period,
                    "horizon_quarters": horizon,
                    "selected_model": selected_model,
                    "forecast_tier": "high_confidence",
                    "policy_eligible": 1,
                    "selection_rule": (
                        "holdout_pass_screened_zip_selection"
                        if allowed_models
                        else "zip_level_history_selection"
                    ),
                    "history_points": int(len(values)),
                    "trailing_contiguous_quarters": int(
                        completeness_lookup.loc[zip_code, "trailing_contiguous_quarters"]
                    ),
                    "overall_completeness_ratio": float(
                        completeness_lookup.loc[zip_code, "overall_completeness_ratio"]
                    ),
                    "predicted_total_rate_per_1000": float(prediction),
                }
            )
            for interval_level in (80, 95):
                lower_bound, upper_bound = _forecast_interval_bounds(
                    prediction=float(prediction),
                    interval_level=interval_level,
                    error_scale=error_scale,
                    history=values,
                    horizon=horizon,
                )
                interval_rows.append(
                    {
                        "zip": zip_code,
                        "forecast_period_start": forecast_period,
                        "horizon_quarters": horizon,
                        "selected_model": selected_model,
                        "forecast_tier": "high_confidence",
                        "policy_eligible": 1,
                        "interval_level": interval_level,
                        "forecast_value": float(prediction),
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "interval_width": float(upper_bound - lower_bound),
                        "upper_to_forecast_ratio": (
                            float(upper_bound / prediction) if float(prediction) > 0 else np.nan
                        ),
                    }
                )

    for tier_name, tier_frame, tier_model, selection_rule in (
        (
            "limited_history",
            limited_history,
            "moving_average_4",
            "limited_history_fixed_moving_average_4",
        ),
        (
            "carry_forward_only",
            carry_forward,
            "naive_last",
            "carry_forward_last_observation",
        ),
    ):
        for row in tier_frame.itertuples(index=False):
            zip_code = str(row.zip)
            ordered = temporal_series.get(zip_code)
            if ordered is None or ordered.empty:
                continue
            values = ordered["total_rate_per_1000"].to_numpy(dtype=float)
            last_period = pd.Timestamp(ordered["period_start"].iloc[-1]).to_period("Q")
            for horizon in range(1, 5):
                prediction = _predict_with_model(values, tier_model, horizon=horizon)
                if pd.isna(prediction):
                    continue
                forecast_period = (last_period + horizon).start_time
                forecast_rows.append(
                    {
                        "zip": zip_code,
                        "forecast_period_start": forecast_period,
                        "horizon_quarters": horizon,
                        "selected_model": tier_model,
                        "forecast_tier": tier_name,
                        "policy_eligible": 0,
                        "selection_rule": selection_rule,
                        "history_points": int(len(values)),
                        "trailing_contiguous_quarters": int(row.trailing_contiguous_quarters),
                        "overall_completeness_ratio": float(row.overall_completeness_ratio),
                        "predicted_total_rate_per_1000": float(prediction),
                    }
                )

    metrics_df = pd.DataFrame.from_records(overall_metric_rows, columns=metric_columns)
    if selected_models:
        selected_counts = pd.Series(selected_models).value_counts()
        metrics_df["selected_zip_count"] = (
            metrics_df["model_name"].map(selected_counts).fillna(0).astype(int)
        )
    if metrics_df["rmse"].notna().any():
        metrics_df["rmse_rank"] = metrics_df["rmse"].rank(method="dense", na_option="bottom")
        metrics_df = metrics_df.sort_values(["rmse_rank", "model_name"], kind="mergesort").drop(
            columns=["rmse_rank"]
        )
    if not forecast_rows and not notes:
        notes.append("No ZIP had enough quarterly history to produce crime forecasts.")

    return (
        metrics_df.reset_index(drop=True),
        pd.DataFrame.from_records(forecast_rows, columns=forecast_columns),
        pd.DataFrame.from_records(interval_rows, columns=interval_columns),
        notes,
    )


def _build_temporal_holdout_artifacts(
    temporal_summary: pd.DataFrame,
    temporal_series: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    holdout_columns = [
        "evaluation_scope",
        "model_name",
        "selected_family",
        "eligible_zip_count",
        "zip_count",
        "holdout_quarters",
        "evaluation_points",
        "mae",
        "rmse",
        "mape",
        "mape_threshold",
        "mape_pass",
    ]
    calibration_columns = [
        "evaluation_scope",
        "model_name",
        "selected_family",
        "interval_level",
        "eligible_zip_count",
        "zip_count",
        "holdout_quarters",
        "evaluation_points",
        "empirical_coverage",
        "target_coverage",
        "coverage_gap",
        "mean_interval_width",
        "median_interval_width",
        "equal_bound_count",
        "equal_bound_share",
        "extreme_upper_ratio_count",
        "extreme_upper_ratio_share",
        "max_upper_to_forecast_ratio",
        "coverage_pass",
        "shape_pass",
        "calibration_pass",
    ]
    notes: list[str] = []
    if temporal_summary.empty or not temporal_series:
        notes.append("Temporal holdout inputs were unavailable, so holdout artifacts are empty.")
        return (
            pd.DataFrame(columns=holdout_columns),
            pd.DataFrame(columns=calibration_columns),
            notes,
        )

    eligible = temporal_summary.loc[
        pd.to_numeric(temporal_summary["forecast_gate_pass"], errors="coerce").fillna(0).astype(int)
        == 1
    ].copy()
    if eligible.empty:
        notes.append(
            "No modeled ZIP met the quarterly history gate required for temporal holdout evaluation."
        )
        return (
            pd.DataFrame(columns=holdout_columns),
            pd.DataFrame(columns=calibration_columns),
            notes,
        )

    holdout_stats: dict[tuple[str, str], dict[str, object]] = {}
    calibration_stats: dict[tuple[str, str, int], dict[str, object]] = {}

    def _update_holdout(
        scope: str, model_name: str, zip_code: str, error: float, actual: float
    ) -> None:
        stats = holdout_stats.setdefault(
            (scope, model_name),
            {"zips": set(), "errors": [], "abs_errors": [], "apes": []},
        )
        stats["zips"].add(zip_code)
        stats["errors"].append(float(error))
        stats["abs_errors"].append(abs(float(error)))
        if actual > 0:
            stats["apes"].append(abs(float(error)) / float(actual) * 100.0)

    def _update_calibration(
        scope: str,
        model_name: str,
        zip_code: str,
        interval_level: int,
        *,
        prediction: float,
        actual: float,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        stats = calibration_stats.setdefault(
            (scope, model_name, interval_level),
            {
                "zips": set(),
                "count": 0,
                "hits": 0,
                "widths": [],
                "equal_bound_count": 0,
                "extreme_upper_ratio_count": 0,
                "max_upper_to_forecast_ratio": np.nan,
            },
        )
        ratio = float(upper_bound / prediction) if prediction > 0 else np.nan
        stats["zips"].add(zip_code)
        stats["count"] += 1
        stats["hits"] += int(lower_bound <= actual <= upper_bound)
        stats["widths"].append(float(upper_bound - lower_bound))
        stats["equal_bound_count"] += int(np.isclose(lower_bound, upper_bound))
        stats["extreme_upper_ratio_count"] += int(pd.notna(ratio) and ratio > 100.0)
        if pd.notna(ratio):
            if pd.isna(stats["max_upper_to_forecast_ratio"]):
                stats["max_upper_to_forecast_ratio"] = ratio
            else:
                stats["max_upper_to_forecast_ratio"] = max(
                    float(stats["max_upper_to_forecast_ratio"]),
                    ratio,
                )

    zip_evaluations: list[dict[str, object]] = []
    for zip_code in eligible["zip"].astype(str):
        ordered = temporal_series.get(zip_code)
        if ordered is None or ordered.empty:
            continue
        values = ordered["total_rate_per_1000"].to_numpy(dtype=float)
        if len(values) < TEMPORAL_HOLDOUT_QUARTERS + 1:
            continue
        training_history = values[:-TEMPORAL_HOLDOUT_QUARTERS]
        holdout_actuals = values[-TEMPORAL_HOLDOUT_QUARTERS:]
        training_metric_rows: list[dict[str, object]] = []
        training_scales: dict[str, float] = {}
        for model_name in FORECAST_MODELS:
            metrics = _walk_forward_forecast_metrics(training_history, model_name=model_name)
            if metrics is None:
                continue
            training_metric_rows.append(
                {
                    "model_name": model_name,
                    "evaluation_points": metrics["evaluation_points"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                }
            )
            training_scales[model_name] = max(
                float(metrics["p75_absolute_error"]),
                float(metrics["mae"]) * 0.75,
                0.01,
            )
        training_metrics = pd.DataFrame(training_metric_rows)
        if training_metrics.empty:
            continue
        zip_evaluations.append(
            {
                "zip": zip_code,
                "training_history": training_history,
                "holdout_actuals": holdout_actuals,
                "training_metrics": training_metrics,
                "training_scales": training_scales,
            }
        )
        for model_name in FORECAST_MODELS:
            if model_name not in training_scales:
                continue
            for horizon, actual in enumerate(holdout_actuals, start=1):
                prediction = _predict_with_model(training_history, model_name, horizon=horizon)
                if pd.isna(prediction) or pd.isna(actual):
                    continue
                error = float(actual - prediction)
                _update_holdout("candidate_model", model_name, zip_code, error, float(actual))
                for interval_level in (80, 95):
                    lower_bound, upper_bound = _forecast_interval_bounds(
                        prediction=float(prediction),
                        interval_level=interval_level,
                        error_scale=training_scales[model_name],
                        history=training_history,
                        horizon=horizon,
                    )
                    _update_calibration(
                        "candidate_model",
                        model_name,
                        zip_code,
                        interval_level,
                        prediction=float(prediction),
                        actual=float(actual),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )

    mape_threshold = 20.0

    def _holdout_rows_for_scope(
        *,
        scope: str,
        selected_family: str | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (stats_scope, model_name), stats in holdout_stats.items():
            if stats_scope != scope:
                continue
            abs_errors = np.asarray(stats["abs_errors"], dtype=float)
            errors = np.asarray(stats["errors"], dtype=float)
            apes = np.asarray(stats["apes"], dtype=float)
            mean_mape = float(np.mean(apes)) if apes.size else np.nan
            rows.append(
                {
                    "evaluation_scope": stats_scope,
                    "model_name": model_name,
                    "selected_family": (
                        model_name if stats_scope == "candidate_model" else selected_family
                    ),
                    "eligible_zip_count": int(len(eligible)),
                    "zip_count": int(len(stats["zips"])),
                    "holdout_quarters": TEMPORAL_HOLDOUT_QUARTERS,
                    "evaluation_points": int(len(abs_errors)),
                    "mae": float(np.mean(abs_errors)),
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "mape": mean_mape,
                    "mape_threshold": mape_threshold,
                    "mape_pass": int(pd.notna(mean_mape) and mean_mape <= mape_threshold),
                }
            )
        return rows

    candidate_holdout_rows = _holdout_rows_for_scope(scope="candidate_model")
    candidate_holdout_df = pd.DataFrame.from_records(
        candidate_holdout_rows, columns=holdout_columns
    )
    allowed_models = tuple(
        model_name
        for model_name in FORECAST_MODELS
        if model_name
        in set(
            candidate_holdout_df.loc[
                pd.to_numeric(candidate_holdout_df["mape_pass"], errors="coerce")
                .fillna(0)
                .astype(int)
                == 1,
                "model_name",
            ]
            .astype(str)
            .tolist()
        )
    )
    selected_family = (
        "holdout_pass_screened_zip_selection" if allowed_models else "zip_level_holdout_selected"
    )
    if allowed_models:
        notes.append(
            "Selected ZIP holdout and forward forecasts are restricted to candidate families that "
            f"pass aggregate temporal holdout ({', '.join(allowed_models)})."
        )
    else:
        notes.append(
            "No candidate family cleared aggregate temporal holdout, so ZIP-level selection falls back "
            "to the full model family set."
        )

    for evaluation in zip_evaluations:
        zip_code = str(evaluation["zip"])
        training_history = np.asarray(evaluation["training_history"], dtype=float)
        holdout_actuals = np.asarray(evaluation["holdout_actuals"], dtype=float)
        training_metrics = pd.DataFrame(evaluation["training_metrics"]).copy()
        training_scales = dict(evaluation["training_scales"])
        selected_model = _select_forecast_model(
            training_metrics,
            history=training_history,
            allowed_models=allowed_models or None,
        )
        if selected_model is None:
            continue
        for horizon, actual in enumerate(holdout_actuals, start=1):
            prediction = _predict_with_model(training_history, selected_model, horizon=horizon)
            if pd.isna(prediction) or pd.isna(actual):
                continue
            error = float(actual - prediction)
            _update_holdout(
                "selected_zip_model",
                "selected_zip_model",
                zip_code,
                error,
                float(actual),
            )
            for interval_level in (80, 95):
                lower_bound, upper_bound = _forecast_interval_bounds(
                    prediction=float(prediction),
                    interval_level=interval_level,
                    error_scale=training_scales[selected_model],
                    history=training_history,
                    horizon=horizon,
                )
                _update_calibration(
                    "selected_zip_model",
                    "selected_zip_model",
                    zip_code,
                    interval_level,
                    prediction=float(prediction),
                    actual=float(actual),
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )

    holdout_rows = candidate_holdout_rows
    holdout_rows.extend(
        _holdout_rows_for_scope(
            scope="selected_zip_model",
            selected_family=selected_family,
        )
    )

    calibration_rows: list[dict[str, object]] = []
    for (scope, model_name, interval_level), stats in calibration_stats.items():
        target_coverage = interval_level / 100.0
        empirical_coverage = _safe_ratio(stats["hits"], stats["count"])
        coverage_gap = (
            abs(empirical_coverage - target_coverage) if pd.notna(empirical_coverage) else np.nan
        )
        equal_share = _safe_ratio(stats["equal_bound_count"], stats["count"])
        extreme_share = _safe_ratio(stats["extreme_upper_ratio_count"], stats["count"])
        coverage_pass = pd.notna(coverage_gap) and coverage_gap <= 0.10
        shape_pass = stats["equal_bound_count"] == 0 and stats["extreme_upper_ratio_count"] == 0
        calibration_rows.append(
            {
                "evaluation_scope": scope,
                "model_name": model_name,
                "selected_family": (model_name if scope == "candidate_model" else selected_family),
                "interval_level": interval_level,
                "eligible_zip_count": int(len(eligible)),
                "zip_count": int(len(stats["zips"])),
                "holdout_quarters": TEMPORAL_HOLDOUT_QUARTERS,
                "evaluation_points": int(stats["count"]),
                "empirical_coverage": empirical_coverage,
                "target_coverage": target_coverage,
                "coverage_gap": coverage_gap,
                "mean_interval_width": float(np.mean(stats["widths"]))
                if stats["widths"]
                else np.nan,
                "median_interval_width": float(np.median(stats["widths"]))
                if stats["widths"]
                else np.nan,
                "equal_bound_count": int(stats["equal_bound_count"]),
                "equal_bound_share": equal_share,
                "extreme_upper_ratio_count": int(stats["extreme_upper_ratio_count"]),
                "extreme_upper_ratio_share": extreme_share,
                "max_upper_to_forecast_ratio": stats["max_upper_to_forecast_ratio"],
                "coverage_pass": int(coverage_pass),
                "shape_pass": int(shape_pass),
                "calibration_pass": int(coverage_pass and shape_pass),
            }
        )

    if not holdout_rows and not notes:
        notes.append(
            "Temporal holdout evaluation did not produce any eligible ZIP-model comparisons."
        )
    else:
        notes.append(
            f"Temporal holdout evaluated the last {TEMPORAL_HOLDOUT_QUARTERS} quarters for "
            f"{len(eligible)} modeled ZIPs that passed the forecast history gate."
        )

    holdout_df = pd.DataFrame.from_records(holdout_rows, columns=holdout_columns)
    if not holdout_df.empty:
        holdout_df = holdout_df.sort_values(
            ["evaluation_scope", "model_name"],
            kind="mergesort",
            ignore_index=True,
        )
    calibration_df = pd.DataFrame.from_records(calibration_rows, columns=calibration_columns)
    if not calibration_df.empty:
        calibration_df = calibration_df.sort_values(
            ["evaluation_scope", "model_name", "interval_level"],
            kind="mergesort",
            ignore_index=True,
        )
    return holdout_df, calibration_df, notes


def _build_interval_shape_artifacts(forecast_intervals: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "evaluation_scope",
        "model_name",
        "selected_family",
        "interval_level",
        "eligible_zip_count",
        "zip_count",
        "holdout_quarters",
        "evaluation_points",
        "empirical_coverage",
        "target_coverage",
        "coverage_gap",
        "mean_interval_width",
        "median_interval_width",
        "equal_bound_count",
        "equal_bound_share",
        "extreme_upper_ratio_count",
        "extreme_upper_ratio_share",
        "max_upper_to_forecast_ratio",
        "coverage_pass",
        "shape_pass",
        "calibration_pass",
    ]
    if forecast_intervals.empty:
        return pd.DataFrame(columns=columns)

    frame = forecast_intervals.copy()
    frame["interval_level"] = pd.to_numeric(frame["interval_level"], errors="coerce")
    frame["forecast_value"] = pd.to_numeric(frame["forecast_value"], errors="coerce")
    frame["lower_bound"] = pd.to_numeric(frame["lower_bound"], errors="coerce")
    frame["upper_bound"] = pd.to_numeric(frame["upper_bound"], errors="coerce")
    grouped_rows: list[dict[str, object]] = []
    for interval_level, level_frame in frame.groupby("interval_level", dropna=True):
        selected_families = sorted(
            level_frame["selected_model"].dropna().astype(str).unique().tolist()
        )
        ratio = np.where(
            pd.to_numeric(level_frame["forecast_value"], errors="coerce") > 0,
            pd.to_numeric(level_frame["upper_bound"], errors="coerce")
            / pd.to_numeric(level_frame["forecast_value"], errors="coerce"),
            np.nan,
        )
        equal_count = int(
            np.isclose(
                pd.to_numeric(level_frame["lower_bound"], errors="coerce"),
                pd.to_numeric(level_frame["upper_bound"], errors="coerce"),
            ).sum()
        )
        extreme_count = int(np.sum(pd.Series(ratio).fillna(-np.inf) > 100.0))
        shape_pass = equal_count == 0 and extreme_count == 0
        grouped_rows.append(
            {
                "evaluation_scope": "forecast_output_shape",
                "model_name": "selected_zip_model",
                "selected_family": (
                    selected_families[0] if len(selected_families) == 1 else "mixed_selected_models"
                ),
                "interval_level": int(interval_level),
                "eligible_zip_count": int(level_frame["zip"].astype(str).nunique()),
                "zip_count": int(level_frame["zip"].astype(str).nunique()),
                "holdout_quarters": np.nan,
                "evaluation_points": int(len(level_frame)),
                "empirical_coverage": np.nan,
                "target_coverage": np.nan,
                "coverage_gap": np.nan,
                "mean_interval_width": float(
                    (
                        pd.to_numeric(level_frame["upper_bound"], errors="coerce")
                        - pd.to_numeric(level_frame["lower_bound"], errors="coerce")
                    ).mean()
                ),
                "median_interval_width": float(
                    np.median(
                        pd.to_numeric(level_frame["upper_bound"], errors="coerce")
                        - pd.to_numeric(level_frame["lower_bound"], errors="coerce")
                    )
                ),
                "equal_bound_count": equal_count,
                "equal_bound_share": _safe_ratio(equal_count, len(level_frame)),
                "extreme_upper_ratio_count": extreme_count,
                "extreme_upper_ratio_share": _safe_ratio(extreme_count, len(level_frame)),
                "max_upper_to_forecast_ratio": float(pd.Series(ratio).max())
                if pd.Series(ratio).notna().any()
                else np.nan,
                "coverage_pass": 0,
                "shape_pass": int(shape_pass),
                "calibration_pass": int(shape_pass),
            }
        )

    diagnostics = pd.DataFrame.from_records(grouped_rows, columns=columns)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(
            ["evaluation_scope", "interval_level"],
            kind="mergesort",
            ignore_index=True,
        )
    return diagnostics


def _build_scenario_artifacts(crime_forecasts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "zip",
        "forecast_period_start",
        "horizon_quarters",
        "selected_model",
        "scenario_name",
        "scenario_multiplier",
        "baseline_total_rate_per_1000",
        "scenario_total_rate_per_1000",
        "scenario_delta_rate_per_1000",
        "evidence_posture",
        "causal_interpretation_allowed",
    ]
    if crime_forecasts.empty:
        return pd.DataFrame(columns=columns)

    forecasts = crime_forecasts.copy()
    if "forecast_tier" in forecasts.columns:
        forecasts = forecasts.loc[
            forecasts["forecast_tier"].astype(str) == "high_confidence"
        ].copy()
    if forecasts.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for row in forecasts.itertuples(index=False):
        forecast_period = pd.Timestamp(row.forecast_period_start)
        for scenario_name, base_multiplier in SCENARIO_MULTIPLIERS.items():
            multiplier = base_multiplier
            if scenario_name == "seasonal_peak" and forecast_period.quarter in (2, 3):
                multiplier = 1.08
            scenario_value = float(row.predicted_total_rate_per_1000) * multiplier
            rows.append(
                {
                    "zip": row.zip,
                    "forecast_period_start": forecast_period,
                    "horizon_quarters": int(row.horizon_quarters),
                    "selected_model": row.selected_model,
                    "scenario_name": scenario_name,
                    "scenario_multiplier": float(multiplier),
                    "baseline_total_rate_per_1000": float(row.predicted_total_rate_per_1000),
                    "scenario_total_rate_per_1000": float(scenario_value),
                    "scenario_delta_rate_per_1000": float(
                        scenario_value - float(row.predicted_total_rate_per_1000)
                    ),
                    "evidence_posture": "exploratory_non_causal",
                    "causal_interpretation_allowed": 0,
                }
            )

    return pd.DataFrame.from_records(rows, columns=columns).sort_values(
        ["zip", "scenario_name", "forecast_period_start"],
        kind="mergesort",
        ignore_index=True,
    )


def _build_benchmark_artifacts(
    model_df: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "zip",
        "crime_cluster",
        "socioeconomic_cluster",
        "market_cluster",
        "home_value",
        "total_rate_per_1000",
        "violent_rate_per_1000",
        "property_rate_per_1000",
        "median_household_income",
        "poverty_rate",
        "home_value_percentile",
        "total_rate_per_1000_percentile",
        "median_household_income_percentile",
        "home_value_zscore",
        "total_rate_per_1000_zscore",
        "median_household_income_zscore",
        "total_rate_vs_metro_pct",
        "home_value_vs_metro_pct",
        "violent_rate_vs_cluster_pct",
        "property_rate_vs_cluster_pct",
        "is_top_quartile_home_value",
        "is_top_quartile_crime_rate",
        "is_bottom_quartile_income",
    ]
    if model_df.empty:
        return pd.DataFrame(columns=columns)

    frame = model_df.copy()
    numeric_columns = [
        "home_value",
        "total_rate_per_1000",
        "violent_rate_per_1000",
        "property_rate_per_1000",
        "median_household_income",
        "poverty_rate",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not cluster_assignments.empty and "zip" in cluster_assignments.columns:
        frame = frame.merge(cluster_assignments, on="zip", how="left")
    else:
        for column in ("crime_cluster", "socioeconomic_cluster", "market_cluster"):
            frame[column] = pd.NA

    metro_means = frame[numeric_columns].mean()
    for column in ("home_value", "total_rate_per_1000", "median_household_income"):
        frame[f"{column}_percentile"] = frame[column].rank(method="average", pct=True)
        column_std = frame[column].std(ddof=0)
        if pd.notna(column_std) and column_std > 0:
            frame[f"{column}_zscore"] = (frame[column] - frame[column].mean()) / column_std
        else:
            frame[f"{column}_zscore"] = np.nan

    frame["total_rate_vs_metro_pct"] = np.where(
        metro_means["total_rate_per_1000"] > 0,
        ((frame["total_rate_per_1000"] / metro_means["total_rate_per_1000"]) - 1.0) * 100.0,
        np.nan,
    )
    frame["home_value_vs_metro_pct"] = np.where(
        metro_means["home_value"] > 0,
        ((frame["home_value"] / metro_means["home_value"]) - 1.0) * 100.0,
        np.nan,
    )

    if "crime_cluster" in frame.columns:
        cluster_means = frame.groupby("crime_cluster", dropna=False)[
            ["violent_rate_per_1000", "property_rate_per_1000"]
        ].mean()
        frame["violent_rate_vs_cluster_pct"] = frame.apply(
            lambda row: (
                (
                    (
                        row["violent_rate_per_1000"]
                        / cluster_means.loc[row["crime_cluster"], "violent_rate_per_1000"]
                    )
                    - 1.0
                )
                * 100.0
                if pd.notna(row["crime_cluster"])
                and row["crime_cluster"] in cluster_means.index
                and pd.notna(cluster_means.loc[row["crime_cluster"], "violent_rate_per_1000"])
                and cluster_means.loc[row["crime_cluster"], "violent_rate_per_1000"] != 0
                else np.nan
            ),
            axis=1,
        )
        frame["property_rate_vs_cluster_pct"] = frame.apply(
            lambda row: (
                (
                    (
                        row["property_rate_per_1000"]
                        / cluster_means.loc[row["crime_cluster"], "property_rate_per_1000"]
                    )
                    - 1.0
                )
                * 100.0
                if pd.notna(row["crime_cluster"])
                and row["crime_cluster"] in cluster_means.index
                and pd.notna(cluster_means.loc[row["crime_cluster"], "property_rate_per_1000"])
                and cluster_means.loc[row["crime_cluster"], "property_rate_per_1000"] != 0
                else np.nan
            ),
            axis=1,
        )
    else:
        frame["violent_rate_vs_cluster_pct"] = np.nan
        frame["property_rate_vs_cluster_pct"] = np.nan

    frame["is_top_quartile_home_value"] = (frame["home_value_percentile"] >= 0.75).astype(int)
    frame["is_top_quartile_crime_rate"] = (frame["total_rate_per_1000_percentile"] >= 0.75).astype(
        int
    )
    frame["is_bottom_quartile_income"] = (
        frame["median_household_income_percentile"] <= 0.25
    ).astype(int)

    return frame[columns].sort_values("zip", kind="mergesort", ignore_index=True)


def _build_drift_artifacts(
    temporal_summary: pd.DataFrame,
    temporal_series: dict[str, pd.DataFrame],
    *,
    data_cutoff: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "domain",
        "entity_type",
        "entity_id",
        "metric",
        "latest_period_start",
        "baseline_periods",
        "latest_value",
        "baseline_mean",
        "baseline_std",
        "z_score",
        "relative_change_pct",
        "drift_flag",
        "trailing_contiguous_quarters",
        "quarter_completeness_ratio",
    ]
    notes: list[str] = []
    if temporal_summary.empty or not temporal_series:
        notes.append("Temporal summary inputs were unavailable, so drift diagnostics are empty.")
        return pd.DataFrame(columns=columns), notes

    eligible = temporal_summary.loc[
        pd.to_numeric(temporal_summary["drift_gate_pass"], errors="coerce").fillna(0).astype(int)
        == 1
    ].copy()
    if eligible.empty:
        notes.append(
            "No modeled ZIP met the trailing-contiguous quarterly history gate required for drift checks."
        )
        return pd.DataFrame(columns=columns), notes

    notes.append(
        f"Drift checks use only modeled ZIPs with >= {DRIFT_HISTORY_MIN_QUARTERS} trailing contiguous quarters "
        f"({len(eligible)} of {len(temporal_summary)} modeled ZIPs passed)."
    )
    rows: list[dict[str, object]] = []
    entities: list[tuple[str, str, pd.Series]] = []
    metro_panel = []
    completeness_lookup = eligible.set_index("zip")
    for zip_code in eligible["zip"].astype(str):
        ordered = temporal_series.get(zip_code)
        if ordered is None or ordered.empty:
            continue
        metro_panel.append(
            ordered[["period_start", "total_rate_per_1000"]].assign(zip=zip_code).copy()
        )
        zip_series = ordered.sort_values("period_start", kind="mergesort").set_index(
            "period_start"
        )["total_rate_per_1000"]
        entities.append(("zip", zip_code, zip_series))
    if not metro_panel:
        notes.append("No eligible ZIP panels were available after drift gating.")
        return pd.DataFrame(columns=columns), notes

    metro_series = (
        pd.concat(metro_panel, ignore_index=True)
        .groupby("period_start")["total_rate_per_1000"]
        .mean()
        .sort_index(kind="mergesort")
    )
    entities.append(("portfolio", "metro", metro_series))

    _cutoff = (data_cutoff if data_cutoff is not None else pd.Timestamp.today()).normalize()
    for entity_type, entity_id, series in entities:
        if len(series) < DRIFT_HISTORY_MIN_QUARTERS:
            notes.append(
                f"{entity_type} {entity_id}: fewer than {DRIFT_HISTORY_MIN_QUARTERS} periods were available for drift checks."
            )
            continue
        baseline = series.iloc[-DRIFT_HISTORY_MIN_QUARTERS:-1]
        latest_value = float(series.iloc[-1])
        # DQA-D2: pro-rate latest value if the latest period is a partial quarter
        latest_period_ts = pd.Timestamp(series.index[-1])
        try:
            quarter_period = pd.Period(latest_period_ts, freq="Q")
            quarter_end = quarter_period.end_time.normalize()
            quarter_length = max(1, (quarter_end - latest_period_ts).days + 1)
            days_elapsed = max(1, min((_cutoff - latest_period_ts).days + 1, quarter_length))
            quarter_completeness = days_elapsed / quarter_length
        except Exception:
            quarter_completeness = 1.0
        if quarter_completeness < DRIFT_MIN_COMPLETENESS:
            # Insufficient data in this quarter — skip normal drift computation
            z_score = float("nan")
            relative_change = float("nan")
            drift_flag = -1  # insufficient data sentinel
            adjusted_latest = latest_value
        else:
            if 0.0 < quarter_completeness < 1.0:
                adjusted_latest = latest_value / quarter_completeness
            else:
                adjusted_latest = latest_value
            baseline_mean = float(baseline.mean())
            baseline_std = float(baseline.std(ddof=0))
            z_score = (
                (adjusted_latest - baseline_mean) / baseline_std if baseline_std > 0 else np.nan
            )
            relative_change = (
                ((adjusted_latest / baseline_mean) - 1.0) * 100.0 if baseline_mean > 0 else np.nan
            )
            drift_flag = int(
                (pd.notna(z_score) and abs(z_score) >= 1.5)
                or (pd.notna(relative_change) and abs(relative_change) >= 25.0)
            )
        baseline_mean = float(baseline.mean())
        baseline_std = float(baseline.std(ddof=0))
        rows.append(
            {
                "domain": "crime",
                "entity_type": entity_type,
                "entity_id": entity_id,
                "metric": "total_rate_per_1000",
                "latest_period_start": series.index[-1],
                "baseline_periods": int(len(baseline)),
                "latest_value": latest_value,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "z_score": z_score,
                "relative_change_pct": relative_change,
                "drift_flag": drift_flag,
                "trailing_contiguous_quarters": (
                    int(completeness_lookup.loc[entity_id, "trailing_contiguous_quarters"])
                    if entity_type == "zip" and entity_id in completeness_lookup.index
                    else int(len(series))
                ),
                "quarter_completeness_ratio": round(quarter_completeness, 4),
            }
        )

    if not rows and not notes:
        notes.append("No series met the minimum history requirement for drift diagnostics.")
    notes.append("Housing drift is not yet modeled in this offline artifact set.")
    return pd.DataFrame.from_records(rows, columns=columns), notes
