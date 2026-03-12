from pathlib import Path

import pandas as pd

from dallas_crime.config import Settings
from dallas_crime.pipeline.analyze import run_analysis


def test_run_analysis_writes_report_artifacts(tmp_path: Path):
    settings = Settings.from_env(project_root=tmp_path)
    settings.ensure_directories()

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
            "centroid_latitude": [32.77, 32.775, 32.78, 32.785, 32.79, 32.795, 32.8, 32.805, 32.81, 32.815],
            "centroid_longitude": [-96.82, -96.815, -96.81, -96.805, -96.8, -96.795, -96.79, -96.785, -96.78, -96.775],
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
        }
    )
    model_df.to_csv(settings.processed_dir / "model_dataset.csv", index=False)

    outputs = run_analysis(settings)

    expected_keys = {
        "coefficients",
        "metrics",
        "sample_sizes",
        "residuals",
        "residual_review",
        "vif",
        "vif_notes",
        "scatter_plot",
        "geography_plot",
        "zip_comparison",
        "model_summary_table",
        "summary",
    }
    assert set(outputs) == expected_keys
    for output in outputs.values():
        assert Path(output).exists()

    metrics = pd.read_csv(outputs["metrics"])
    assert set(metrics["model_label"]) == {"baseline", "expanded_controls"}
    assert {"nobs", "r_squared", "adjusted_r_squared"} <= set(metrics.columns)

    coefficients = pd.read_csv(outputs["coefficients"])
    assert set(coefficients["model_label"]) == {"baseline", "expanded_controls"}
    assert {"term", "estimate", "std_error", "p_value"} <= set(coefficients.columns)

    sample_sizes = pd.read_csv(outputs["sample_sizes"])
    assert set(sample_sizes["model_label"]) == {"baseline", "expanded_controls"}
    assert sample_sizes["nobs"].min() >= 8

    residuals = pd.read_csv(outputs["residuals"])
    assert set(residuals["model_label"]) == {"baseline", "expanded_controls"}
    assert {"observed", "fitted_value", "residual", "absolute_residual"} <= set(residuals.columns)

    vif = pd.read_csv(outputs["vif"])
    assert set(vif["model_label"]) == {"baseline", "expanded_controls"}
    assert {"term", "vif"} <= set(vif.columns)

    summary = Path(outputs["summary"]).read_text()
    assert "Dallas Crime and Housing Report" in summary
    assert "## Methods" in summary
    assert "## Findings" in summary
    assert "## Limitations" in summary
    assert "top_bottom_zip_comparison.md" in summary
    assert "crime_home_value_geography.png" in summary
    assert "dallas-crime acquire && dallas-crime build && dallas-crime analyze" in summary
