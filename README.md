# Dallas Crime and Housing Refresh

This repository is the Python-first refresh of the original Dallas crime/housing coursework.
It preserves the Dallas ZIP-level focus while replacing the legacy one-off workflow with a
reproducible CLI pipeline and documented analysis artifacts.

## Legacy artifact policy

The legacy R coursework files are retained for historical context only and are not active
pipeline entry points:

- `DallasArrests.Rmd`
- `DallasArrests.pdf`
- `DallasArrests_cleaned.csv`
- `Dallas_Arrests_Property_Summary.csv`
- `Police_Arrests.csv`
- `Zillow-dallas.csv`

All production refreshes should use the Python CLI commands below.

## What the Python workflow does

- Acquires Dallas crime, ACS ZIP controls, and multi-source ZIP housing signals.
- Requests housing only for crime ZIPs that clear the configured minimum-incident threshold, reducing raw out-of-area singleton pollution in downstream sources.
- Writes a separate historical housing panel covering 2025 back to 2000 from Realtor monthly history and official FHFA ZIP5 annual HPI.
- Uses Firecrawl to search Zillow first, then directly scrape Realtor.com and Redfin ZIP market pages as fallbacks, and merges structured Realtor ZIP inventory/history feeds for broader coverage and recent trend features.
- Builds ZIP-level processed datasets with rates and controls.
- Excludes ZIPs with fewer than `DCA_MIN_TOTAL_INCIDENTS_PER_ZIP` incidents in the active crime window to avoid singleton garbage matches in the study universe.
- Runs two regression model specifications from the same processed dataset:
  - baseline model
  - expanded-controls model
- Exports model coefficients, metrics, diagnostics, and narrative report artifacts.

## Project layout

- `src/dallas_crime/cli.py`: CLI entrypoint.
- `src/dallas_crime/acquire/`: source acquisition helpers.
- `src/dallas_crime/pipeline/`: build and analysis logic.
- `tests/`: fixture-backed acquisition and pipeline tests.
- `docs/`: methodology, source dictionary, and refresh workflow docs.

Runtime outputs are intentionally gitignored:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `reports/`
- `.firecrawl/`

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional configuration lives in `.env.example`. Defaults target:

- Dallas OpenData Police Incidents
- Texas ZCTA ACS 5-year data
- Zillow ZIP home-value pages discovered through Firecrawl
- Realtor.com ZIP market pages for housing/rent fallback coverage
- Realtor ZIP inventory current and historical bulk feeds for structured price and market trend features
- FHFA ZIP5 annual HPI for official year-2000 backfill
- Redfin ZIP housing-market pages for sale-price fallback coverage

## Commands

Show resolved config:

```bash
dallas-crime show-config
```

Dry-run acquisition:

```bash
dallas-crime acquire --dry-run
```

Fetch raw sources:

```bash
dallas-crime acquire
```

Build processed datasets:

```bash
dallas-crime build
```

Run analysis and reporting artifacts:

```bash
dallas-crime analyze
```

If `Makefile` is available, the same flow is:

```bash
make setup
make test
make smoke
```

## Analysis artifacts

`dallas-crime analyze` writes to `reports/`:

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

## Documentation

- `docs/methodology.md`
- `docs/source_dictionary.md`
- `docs/refresh_workflow.md`
- `docs/legacy_artifacts_policy.md`

## CI

GitHub Actions CI runs on pushes and pull requests:

- dependency install
- test suite (`pytest`)
- CLI smoke checks (`show-config`, `build`, `analyze`) against generated sample raw inputs

## Notes

- The study universe defaults to ZIPs with at least 2 incidents in the current crime refresh (`DCA_MIN_TOTAL_INCIDENTS_PER_ZIP`).

- Pipeline acquisition uses Firecrawl CLI, and `.mcp.json` provides Firecrawl MCP for agent tooling.
- Store Firecrawl secrets in macOS Keychain, not repo files: `security add-generic-password -U -a "$USER" -s firecrawl_api_key -w "<FIRECRAWL_API_KEY>"`.
- Default crime window is the most recent 365 days (`DCA_CRIME_LOOKBACK_DAYS`).
- Minimum study-universe threshold defaults to 2 incidents per ZIP (`DCA_MIN_TOTAL_INCIDENTS_PER_ZIP`).
- Live smoke runs can be bounded with `DCA_MAX_HOUSING_ZIPS`.
- Firecrawl batching can be tuned with `DCA_HOUSING_SCRAPE_BATCH_SIZE`.
