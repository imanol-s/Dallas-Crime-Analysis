# Refresh Workflow

## Prerequisites

1. Python environment:
   - `uv venv`
   - `source .venv/bin/activate`
   - `uv pip install -e ".[dev]"`
2. Optional API keys:
   - `FIRECRAWL_API_KEY`
   - `CENSUS_API_KEY`
3. Optional runtime overrides in `.env` (see `.env.example`).

## Standard refresh commands

1. Acquire raw data:
   - `dallas-crime acquire`
2. Build processed datasets:
   - `dallas-crime build`
3. Run analysis and generate report artifacts:
   - `dallas-crime analyze`

Documented one-command regeneration path from committed code and environment variables:

```bash
dallas-crime acquire && dallas-crime build && dallas-crime analyze
```

## Outputs to review after refresh

- Processed dataset:
  - `data/processed/model_dataset.csv`
  - `data/processed/housing_history_panel.csv`
  - `data/processed/qa_summary.json`
- Analysis artifacts in `reports/`:
  - `regression_coefficients.csv`
  - `regression_metrics.csv`
  - `model_sample_sizes.csv`
  - `model_residuals.csv`
  - `residual_review.md`
  - `model_vif.csv`
  - `model_vif_notes.md`
  - `home_value_vs_total_crime.png`
  - `crime_home_value_geography.png`
  - `top_bottom_zip_comparison.md`
  - `model_summary_table.md`
  - `summary.md`

## Key runtime controls

- `DCA_CRIME_LOOKBACK_DAYS`: active crime window for acquisition.
- `DCA_MIN_TOTAL_INCIDENTS_PER_ZIP`: study-universe threshold to exclude singleton ZIP matches.
- `DCA_MAX_HOUSING_ZIPS`: optional cap for bounded smoke runs.
- `DCA_HOUSING_SCRAPE_BATCH_SIZE`: Firecrawl search batch size for current housing.

## Smoke-run option for quick validation

When testing the pipeline with limited acquisition scope, set:

- `DCA_CRIME_LOOKBACK_DAYS=14`
- `DCA_MAX_HOUSING_ZIPS=10`

Then run the same `acquire`, `build`, `analyze` sequence.
