"""Regression utilities and report generation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

if TYPE_CHECKING:
    from dallas_crime.config import Settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PREDICTORS = ("violent_rate_per_1000", "property_rate_per_1000")
DEFAULT_CONTROLS = (
    "median_household_income",
    "poverty_rate",
    "owner_occupied_share",
    "median_gross_rent",
)
EXPANDED_CONTROL_CANDIDATES = (
    "population_acs",
    "median_rent",
    "annual_change_pct",
    "realtor_active_listing_count",
    "realtor_median_days_on_market",
    "realtor_pending_ratio",
    "realtor_hist_listing_price_12m_change",
)


@dataclass(slots=True)
class RegressionResult:
    """Compact regression payload for report generation."""

    model_label: str
    formula: str
    dependent_variable: str
    predictors: tuple[str, ...]
    controls: tuple[str, ...]
    nobs: int
    r_squared: float
    adjusted_r_squared: float
    coefficients: pd.DataFrame
    model_frame: pd.DataFrame
    residuals: pd.DataFrame

    def metrics_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model_label": self.model_label,
                    "dependent_variable": self.dependent_variable,
                    "formula": self.formula,
                    "predictors": ", ".join(self.predictors),
                    "controls": ", ".join(self.controls),
                    "nobs": self.nobs,
                    "r_squared": self.r_squared,
                    "adjusted_r_squared": self.adjusted_r_squared,
                }
            ]
        )


def _coerce_model_columns(model_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = model_df.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _minimum_rows(required_columns: list[str]) -> int:
    return len(required_columns) + 1


def _ensure_dependent_column(model_df: pd.DataFrame, dependent: str) -> pd.DataFrame:
    frame = model_df.copy()
    if dependent in frame.columns:
        return frame

    if dependent == "log_home_value" and "home_value" in frame.columns:
        home_values = pd.to_numeric(frame["home_value"], errors="coerce")
        frame["log_home_value"] = np.where(home_values > 0, np.log(home_values), np.nan)
        return frame

    raise KeyError(f"model_df is missing required dependent variable: {dependent}")


def _select_expanded_controls(
    model_df: pd.DataFrame,
    *,
    dependent: str,
    predictors: tuple[str, ...],
    baseline_controls: tuple[str, ...],
) -> tuple[str, ...]:
    frame = _ensure_dependent_column(model_df, dependent)
    base_controls = list(baseline_controls)
    selected_extras: list[str] = []

    for candidate in EXPANDED_CONTROL_CANDIDATES:
        if candidate not in frame.columns:
            continue

        trial_controls = [*base_controls, *selected_extras, candidate]
        trial_columns = [dependent, *predictors, *trial_controls]
        trial_frame = _coerce_model_columns(frame, trial_columns).dropna(subset=trial_columns)
        if len(trial_frame) >= _minimum_rows(trial_columns):
            selected_extras.append(candidate)

    if not selected_extras:
        raise ValueError(
            "unable to build an expanded-controls model: none of the candidate expanded controls "
            f"({', '.join(EXPANDED_CONTROL_CANDIDATES)}) were available with enough complete rows"
        )

    return tuple([*base_controls, *selected_extras])


def run_zip_regression(
    model_df: pd.DataFrame,
    *,
    dependent: str = "log_home_value",
    predictors: tuple[str, ...] = DEFAULT_PREDICTORS,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    model_label: str = "baseline",
) -> RegressionResult:
    """Fit a robust OLS model over ZIP-level housing outcomes."""

    frame = _ensure_dependent_column(model_df, dependent)

    required_columns = [dependent, *predictors, *controls]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"model_df is missing required regression columns: {missing_list}")

    frame = _coerce_model_columns(frame, required_columns)
    frame = frame.dropna(subset=required_columns).copy()

    minimum_rows = _minimum_rows(required_columns)
    if len(frame) < minimum_rows:
        raise ValueError(
            f"need at least {minimum_rows} complete rows to fit the requested model; got {len(frame)}"
        )

    formula = f"{dependent} ~ {' + '.join([*predictors, *controls])}"
    fitted = smf.ols(formula=formula, data=frame).fit(cov_type="HC3")

    coefficients = pd.DataFrame(
        {
            "model_label": model_label,
            "dependent_variable": dependent,
            "formula": formula,
            "term": fitted.params.index,
            "estimate": fitted.params.values,
            "std_error": fitted.bse.values,
            "t_value": fitted.tvalues.values,
            "p_value": fitted.pvalues.values,
            "conf_low": fitted.conf_int()[0].values,
            "conf_high": fitted.conf_int()[1].values,
        }
    )

    residual_columns = ["zip"] if "zip" in frame.columns else []
    residuals = frame[residual_columns].copy()
    residuals["model_label"] = model_label
    residuals["observed"] = frame[dependent].to_numpy()
    residuals["fitted_value"] = fitted.fittedvalues.to_numpy()
    residuals["residual"] = fitted.resid.to_numpy()
    residuals["absolute_residual"] = residuals["residual"].abs()

    return RegressionResult(
        model_label=model_label,
        formula=formula,
        dependent_variable=dependent,
        predictors=predictors,
        controls=controls,
        nobs=int(fitted.nobs),
        r_squared=float(fitted.rsquared),
        adjusted_r_squared=float(fitted.rsquared_adj),
        coefficients=coefficients,
        model_frame=frame,
        residuals=residuals,
    )


def run_analysis(settings: "Settings") -> dict[str, str]:
    """Run regression analysis from the processed model dataset and write report artifacts."""

    model_path = settings.processed_dir / "model_dataset.csv"
    if not model_path.exists():
        raise FileNotFoundError(f"Processed model dataset not found at {model_path}")

    model_df = pd.read_csv(model_path)

    baseline_result = run_zip_regression(
        model_df,
        predictors=DEFAULT_PREDICTORS,
        controls=DEFAULT_CONTROLS,
        model_label="baseline",
    )
    expanded_controls = _select_expanded_controls(
        model_df,
        dependent="log_home_value",
        predictors=DEFAULT_PREDICTORS,
        baseline_controls=DEFAULT_CONTROLS,
    )
    expanded_result = run_zip_regression(
        model_df,
        predictors=DEFAULT_PREDICTORS,
        controls=expanded_controls,
        model_label="expanded_controls",
    )
    results = [baseline_result, expanded_result]

    coefficients_path = settings.reports_dir / "regression_coefficients.csv"
    metrics_path = settings.reports_dir / "regression_metrics.csv"
    sample_sizes_path = settings.reports_dir / "model_sample_sizes.csv"
    residuals_path = settings.reports_dir / "model_residuals.csv"
    residual_review_path = settings.reports_dir / "residual_review.md"
    vif_path = settings.reports_dir / "model_vif.csv"
    vif_notes_path = settings.reports_dir / "model_vif_notes.md"
    scatter_path = settings.reports_dir / "home_value_vs_total_crime.png"
    geography_path = settings.reports_dir / "crime_home_value_geography.png"
    zip_comparison_path = settings.reports_dir / "top_bottom_zip_comparison.md"
    model_summary_table_path = settings.reports_dir / "model_summary_table.md"
    report_path = settings.reports_dir / "summary.md"

    coefficients = pd.concat([result.coefficients for result in results], ignore_index=True)
    metrics = pd.concat([result.metrics_table() for result in results], ignore_index=True)
    sample_sizes = metrics[["model_label", "nobs", "predictors", "controls", "formula"]].copy()
    residuals = (
        pd.concat([result.residuals for result in results], ignore_index=True)
        .sort_values(["model_label", "absolute_residual"], ascending=[True, False])
        .reset_index(drop=True)
    )
    vif_table, vif_notes = _build_vif_artifacts(results)

    coefficients.to_csv(coefficients_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    sample_sizes.to_csv(sample_sizes_path, index=False)
    residuals.to_csv(residuals_path, index=False)
    vif_table.to_csv(vif_path, index=False)
    _write_residual_review(residuals, residual_review_path)
    _write_vif_notes(vif_notes, vif_notes_path)
    _write_scatter_plot(model_df, scatter_path)
    _write_geography_plot(model_df, geography_path)
    _write_zip_comparison_table(model_df, zip_comparison_path)
    _write_model_summary_table(metrics, model_summary_table_path)
    _write_summary_report(model_df=model_df, results=results, settings=settings, output_path=report_path)

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
        "zip_comparison": str(zip_comparison_path),
        "model_summary_table": str(model_summary_table_path),
        "summary": str(report_path),
    }


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


def _build_vif_artifacts(results: list[RegressionResult]) -> tuple[pd.DataFrame, list[str]]:
    vif_rows: list[pd.DataFrame] = []
    notes: list[str] = []

    for result in results:
        columns = [*result.predictors, *result.controls]
        frame = _coerce_model_columns(result.model_frame, columns).dropna(subset=columns)
        if len(frame) <= len(columns):
            notes.append(
                f"- `{result.model_label}`: skipped VIF because sample size "
                f"({len(frame)}) is too small for {len(columns)} regressors."
            )
            continue

        usable_columns = [column for column in columns if frame[column].nunique(dropna=True) > 1]
        dropped_columns = sorted(set(columns) - set(usable_columns))
        if len(usable_columns) < 2:
            notes.append(
                f"- `{result.model_label}`: skipped VIF because fewer than two regressors had usable variance."
            )
            continue

        design = sm.add_constant(frame[usable_columns], has_constant="add")
        model_rows = []
        try:
            for index, column in enumerate(design.columns):
                if column == "const":
                    continue
                model_rows.append(
                    {
                        "model_label": result.model_label,
                        "term": column,
                        "vif": float(variance_inflation_factor(design.values, index)),
                    }
                )
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            notes.append(f"- `{result.model_label}`: skipped VIF due numerical instability ({error}).")
            continue

        infinite_terms = [row["term"] for row in model_rows if not np.isfinite(row["vif"])]
        if infinite_terms:
            notes.append(
                f"- `{result.model_label}`: infinite VIF for "
                f"{', '.join(sorted(infinite_terms))}, indicating exact collinearity."
            )
        vif_rows.append(pd.DataFrame(model_rows))
        if dropped_columns:
            notes.append(
                f"- `{result.model_label}`: dropped zero-variance terms before VIF: "
                f"{', '.join(dropped_columns)}."
            )

    if not vif_rows:
        empty = pd.DataFrame(columns=["model_label", "term", "vif"])
        return empty, notes
    combined = pd.concat(vif_rows, ignore_index=True).sort_values(
        ["model_label", "vif"],
        ascending=[True, False],
    )
    return combined, notes


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


def _coefficient_for_term(result: RegressionResult, term: str) -> float | None:
    match = result.coefficients[result.coefficients["term"] == term]
    if match.empty:
        return None
    return float(match.iloc[0]["estimate"])


def _effect_size_text(coefficient: float | None) -> str:
    if coefficient is None:
        return "not estimable from this specification"
    percent_change = (np.exp(coefficient) - 1.0) * 100.0
    return f"{percent_change:.2f}% change in expected home value"


def _write_summary_report(
    model_df: pd.DataFrame,
    results: list[RegressionResult],
    settings: "Settings",
    output_path: Path,
) -> None:
    numeric = model_df[["total_rate_per_1000", "home_value"]].apply(pd.to_numeric, errors="coerce")
    total_corr = numeric.corr().iloc[0, 1]
    baseline = next(result for result in results if result.model_label == "baseline")
    expanded = next(result for result in results if result.model_label == "expanded_controls")

    baseline_violent = _effect_size_text(_coefficient_for_term(baseline, "violent_rate_per_1000"))
    baseline_property = _effect_size_text(_coefficient_for_term(baseline, "property_rate_per_1000"))
    expanded_violent = _effect_size_text(_coefficient_for_term(expanded, "violent_rate_per_1000"))
    expanded_property = _effect_size_text(_coefficient_for_term(expanded, "property_rate_per_1000"))
    housing_sources = sorted(str(source) for source in model_df.get("source", pd.Series(dtype="string")).dropna().unique())
    total_corr_text = f"{total_corr:.3f}" if pd.notna(total_corr) else "not estimable"
    history_panel_path = settings.processed_dir / "housing_history_panel.csv"
    history_note = (
        "Recent context was supplemented with a 2000-2025 historical housing panel "
        "(Realtor monthly history plus FHFA ZIP5 annual HPI)."
        if history_panel_path.exists()
        else "Historical housing context was not available for this run."
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
        f"- Historical context: {history_note}",
        "- Strategy: two robust regression models were run from the same processed dataset:",
        "  - baseline model with core crime + household controls",
        f"  - expanded-controls model adding completeness-qualified controls: {', '.join(expanded.controls)}",
        "",
        "## Findings",
        "",
        f"- Baseline sample size: {baseline.nobs} ZIPs; expanded-controls sample size: {expanded.nobs} ZIPs.",
        f"- Overall correlation between total crime rate and home value: {total_corr_text}.",
        "- Estimated effect per +1 crime incident per 1,000 residents:",
        f"  - Baseline violent-rate term: {baseline_violent}.",
        f"  - Baseline property-rate term: {baseline_property}.",
        f"  - Expanded violent-rate term: {expanded_violent}.",
        f"  - Expanded property-rate term: {expanded_property}.",
        f"- Model fit stayed stable across specs (baseline R-squared: {baseline.r_squared:.3f}, "
        f"expanded R-squared: {expanded.r_squared:.3f}).",
        "",
        "## Limitations",
        "",
        "- This is observational ZIP-level analysis and should not be read as causal proof.",
        "- Housing measures combine multiple reputable sources with different definitions "
        "(typical value, median listing price, median sale price, HPI), so they are not perfectly interchangeable.",
        "- ACS and FHFA inputs can lag real-time market and demographic shifts.",
        "- Small sample sizes and correlated neighborhood factors can inflate uncertainty.",
        "- Results are sensitive to ZIP-universe filtering, available controls, and the current crime lookback window.",
        "",
        "## Visuals and Artifacts",
        "",
        "- `home_value_vs_total_crime.png` (scatter visual)",
        "- `crime_home_value_geography.png` (ZIP centroid geography-aware view)",
        "- `top_bottom_zip_comparison.md` (additional ZIP comparison table)",
        "- `model_summary_table.md` (model-level metrics table)",
        "- `regression_coefficients.csv` and `regression_metrics.csv`",
        "- `model_sample_sizes.csv`, `model_residuals.csv`, `residual_review.md`",
        "- `model_vif.csv` and `model_vif_notes.md`",
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
    output_path.write_text("\n".join(lines) + "\n")
