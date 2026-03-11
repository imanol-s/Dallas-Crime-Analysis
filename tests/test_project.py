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
    model_df.to_csv(settings.processed_dir / "model_dataset.csv", index=False)

    outputs = run_analysis(settings)

    for output in outputs.values():
        assert Path(output).exists()
    summary = Path(outputs["summary"]).read_text()
    assert "Dallas Crime and Housing Summary" in summary
    assert "Regression formula" in summary
