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

- Acquires current-window Dallas crime, longer-horizon crime history, current ACS ZIP controls,
  multi-year ACS snapshots, and multi-source ZIP housing signals.
- Requests housing only for crime ZIPs that clear the configured minimum-incident threshold, reducing raw out-of-area singleton pollution in downstream sources.
- Derives a quarterly ZIP-level crime history panel and additive temporal crime features from the
  longer-horizon crime pull when available.
- Writes a separate historical housing panel covering 2025 back to 2000 from Realtor monthly history and official FHFA ZIP5 annual HPI.
- Derives additive ZIP-level history coverage and trend features from that housing panel for downstream modeling.
- Uses Firecrawl to search Zillow first, then directly scrape Realtor.com and Redfin ZIP market pages as fallbacks, and merges structured Realtor ZIP inventory/history feeds for broader coverage and recent trend features.
- Builds ZIP-level processed datasets with rates and controls.
- Adds additive interaction and aggregate risk-pressure features to the model dataset.
- Excludes ZIPs with fewer than `DCA_MIN_TOTAL_INCIDENTS_PER_ZIP` incidents in the active crime window to avoid singleton garbage matches in the study universe.
- Runs two robust regression model specifications from the same processed dataset:
  - baseline model
  - expanded-controls model
- Exports feature-selection diagnostics, predictive model-family selection artifacts,
  12-month crime forecasts with confidence intervals, stress scenarios, benchmark and drift diagnostics,
  policy recommendations by segment, and narrative summary artifacts.

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

This writes the current-window `crime_records.csv`, longer-horizon `crime_history_records.csv`,
latest-year `acs_zcta.csv`, multi-year `acs_zcta_snapshots.csv`, the housing raw artifacts,
and the category sidecars:
`dfw_zip_economic_sidecar.csv`, `dfw_zip_real_estate_sidecar.csv`,
`dfw_zip_law_enforcement_sidecar.csv`, `dfw_zip_social_services_sidecar.csv`,
`dfw_zip_infrastructure_sidecar.csv`.

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
- `cluster_assignments.csv`
- `cluster_profiles.csv`
- `spatial_diagnostics.csv`
- `spatial_hotspots.csv`
- `model_validation_metrics.csv`
- `model_validation_notes.md`
- `crime_trend_decomposition.csv`
- `crime_trend_decomposition.md`
- `feature_selection_metrics.csv`
- `feature_selection_notes.md`
- `feature_power_retention_metrics.csv`
- `feature_power_retention_notes.md`
- `predictive_model_metrics.csv`
- `predictive_model_predictions.csv`
- `model_selection_notes.md`
- `forecast_model_metrics.csv`
- `crime_forecasts.csv`
- `forecast_confidence_intervals.csv`
- `forecast_notes.md`
- `scenario_impacts.csv`
- `scenario_notes.md`
- `zip_benchmarks.csv`
- `benchmark_summary.md`
- `model_drift_diagnostics.csv`
- `model_drift_notes.md`
- `comprehensive_validation_metrics.csv`
- `comprehensive_validation_notes.md`
- `policy_recommendations_by_segment.csv`
- `policy_recommendations_by_segment.md`

## Documentation

- `docs/methodology.md`
- `docs/source_dictionary.md`
- `docs/refresh_workflow.md`
- `docs/legacy_artifacts_policy.md`
- `docs/v2_technical_compendium.md`
- `docs/architecture_decisions.md`
- `docs/technical_documentation_index.md`
- `docs/dashboard_specifications.md`
- `V2_EXECUTION_TRACKER.md`

## Agent note

- OpenCode project subagent: `.opencode/agents/acquire.md` provides `@acquire` for acquisition-focused work, including live source refreshes, validation, quality reporting, and `src/dallas_crime/acquire/`-scoped changes.
- OpenCode project subagents: `.opencode/agents/project-manager.md`, `.opencode/agents/data-engineer.md`, and `.opencode/agents/data-scientist.md` provide roadmap coordination, pipeline engineering, and analysis/modeling specialists for V2 work.
- Codex/project skill: `.agent/skills/acquire/SKILL.md` and `.agents/skills/acquire/SKILL.md` provide matching acquisition guidance for skill-based workflows.
- Codex/project skills: `.agent/skills/` and `.agents/skills/` also include `project-manager`, `data-engineer`, and `data-scientist` skills for matching role-based workflows.

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
- Default crime-history window is the most recent 1825 days
  (`DCA_CRIME_HISTORY_LOOKBACK_DAYS`).
- Default ACS snapshot years are the most recent five ACS releases ending at
  `DCA_CENSUS_YEAR` (`DCA_CENSUS_SNAPSHOT_YEARS`).
- Minimum study-universe threshold defaults to 2 incidents per ZIP (`DCA_MIN_TOTAL_INCIDENTS_PER_ZIP`).
- Live smoke runs can be bounded with `DCA_MAX_HOUSING_ZIPS`.
- Firecrawl batching can be tuned with `DCA_HOUSING_SCRAPE_BATCH_SIZE`.
