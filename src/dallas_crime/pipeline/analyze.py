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
import statsmodels.formula.api as smf

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


@dataclass(slots=True)
class RegressionResult:
    """Compact regression payload for report generation."""

    formula: str
    dependent_variable: str
    nobs: int
    r_squared: float
    adjusted_r_squared: float
    coefficients: pd.DataFrame
    model_frame: pd.DataFrame

    def metrics_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "dependent_variable": self.dependent_variable,
                    "formula": self.formula,
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


def run_zip_regression(
    model_df: pd.DataFrame,
    *,
    dependent: str = "log_home_value",
    predictors: tuple[str, ...] = DEFAULT_PREDICTORS,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
) -> RegressionResult:
    """Fit a robust OLS model over ZIP-level housing outcomes."""

    frame = model_df.copy()
    if dependent not in frame.columns:
        if dependent == "log_home_value" and "home_value" in frame.columns:
            home_values = pd.to_numeric(frame["home_value"], errors="coerce")
            frame["log_home_value"] = np.where(home_values > 0, np.log(home_values), np.nan)
        else:
            raise KeyError(f"model_df is missing required dependent variable: {dependent}")

    required_columns = [dependent, *predictors, *controls]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"model_df is missing required regression columns: {missing_list}")

    frame = _coerce_model_columns(frame, required_columns)
    frame = frame.dropna(subset=required_columns).copy()

    minimum_rows = len(required_columns) + 1
    if len(frame) < minimum_rows:
        raise ValueError(
            f"need at least {minimum_rows} complete rows to fit the requested model; got {len(frame)}"
        )

    formula = f"{dependent} ~ {' + '.join([*predictors, *controls])}"
    fitted = smf.ols(formula=formula, data=frame).fit(cov_type="HC3")

    coefficients = pd.DataFrame(
        {
            "term": fitted.params.index,
            "estimate": fitted.params.values,
            "std_error": fitted.bse.values,
            "t_value": fitted.tvalues.values,
            "p_value": fitted.pvalues.values,
            "conf_low": fitted.conf_int()[0].values,
            "conf_high": fitted.conf_int()[1].values,
        }
    )

    return RegressionResult(
        formula=formula,
        dependent_variable=dependent,
        nobs=int(fitted.nobs),
        r_squared=float(fitted.rsquared),
        adjusted_r_squared=float(fitted.rsquared_adj),
        coefficients=coefficients,
        model_frame=frame,
    )


def run_analysis(settings: "Settings") -> dict[str, str]:
    """Run regression analysis from the processed model dataset and write report artifacts."""

    model_path = settings.processed_dir / "model_dataset.csv"
    if not model_path.exists():
        raise FileNotFoundError(f"Processed model dataset not found at {model_path}")

    model_df = pd.read_csv(model_path)
    result = run_zip_regression(model_df)

    coefficients_path = settings.reports_dir / "regression_coefficients.csv"
    metrics_path = settings.reports_dir / "regression_metrics.csv"
    scatter_path = settings.reports_dir / "home_value_vs_total_crime.png"
    report_path = settings.reports_dir / "summary.md"

    result.coefficients.to_csv(coefficients_path, index=False)
    result.metrics_table().to_csv(metrics_path, index=False)
    _write_scatter_plot(model_df, scatter_path)
    _write_summary_report(model_df, result, report_path)

    return {
        "coefficients": str(coefficients_path),
        "metrics": str(metrics_path),
        "scatter_plot": str(scatter_path),
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


def _write_summary_report(
    model_df: pd.DataFrame,
    result: RegressionResult,
    output_path: Path,
) -> None:
    numeric = model_df[["total_rate_per_1000", "home_value"]].apply(pd.to_numeric, errors="coerce")
    total_corr = numeric.corr().iloc[0, 1]
    top_home_values = (
        model_df[["zip", "home_value", "total_rate_per_1000"]]
        .copy()
        .assign(
            home_value=lambda frame: pd.to_numeric(frame["home_value"], errors="coerce"),
            total_rate_per_1000=lambda frame: pd.to_numeric(
                frame["total_rate_per_1000"], errors="coerce"
            ),
        )
        .sort_values("home_value", ascending=False)
        .head(10)
    )

    table_lines = [
        "| zip | home_value | total_rate_per_1000 |",
        "| --- | ---: | ---: |",
    ]
    for row in top_home_values.itertuples(index=False):
        table_lines.append(f"| {row.zip} | {row.home_value:.0f} | {row.total_rate_per_1000:.2f} |")

    lines = [
        "# Dallas Crime and Housing Summary",
        "",
        f"- ZIPs modeled: {result.nobs}",
        f"- Regression formula: `{result.formula}`",
        f"- R-squared: {result.r_squared:.3f}",
        f"- Adjusted R-squared: {result.adjusted_r_squared:.3f}",
        f"- Correlation between total crime rate and home value: {total_corr:.3f}",
        "",
        "## Highest Home Value ZIPs",
        "",
        *table_lines,
        "",
        "## Outputs",
        "",
        "- `regression_coefficients.csv`",
        "- `regression_metrics.csv`",
        "- `home_value_vs_total_crime.png`",
    ]
    output_path.write_text("\n".join(lines) + "\n")
