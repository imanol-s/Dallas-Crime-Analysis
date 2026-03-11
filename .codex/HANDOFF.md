# Handoff

## Status

- Branch: `Develpoment/1.1`
- Refresh completed as a Python-first rebuild of the old Dallas crime/housing project.
- Legacy R artifacts remain in the repo for reference; the new implementation lives under `src/dallas_crime/`.

## What changed

- Added a `uv`-managed Python package and CLI in `src/dallas_crime/cli.py`.
- Added acquisition modules for:
  - Dallas OpenData crime pulls
  - Census ACS ZCTA pulls
  - Firecrawl/Zillow housing discovery and parsing
- Added ZIP-level build and analysis modules:
  - crime aggregation and rates per 1,000
  - ACS/housing merge into a model dataset
  - robust OLS regression and report artifact generation
- Added tests and fixtures under `tests/`.
- Rewrote `README.md` and added `.env.example`.
- Added `.gitignore` entries for generated runtime outputs, `.firecrawl/`, `.matplotlib/`, and local Python caches.

## Verified

- Local environment created with `uv venv`.
- Dependencies installed with `uv pip install -e ".[dev]"`.
- Test suite passes:
  - `UV_CACHE_DIR=.uv-cache .venv/bin/pytest -q`
  - Result: `13 passed`
- Live bounded smoke run completed:
  - `DCA_CRIME_LOOKBACK_DAYS=14`
  - `DCA_MAX_HOUSING_ZIPS=10`
  - Commands run:
    - `dallas-crime acquire`
    - `dallas-crime build`
    - `dallas-crime analyze`

## Smoke run outputs

- Raw:
  - `data/raw/crime_records.csv`
  - `data/raw/acs_zcta.csv`
  - `data/raw/housing_market.csv`
- Processed:
  - `data/processed/crime_zip.csv`
  - `data/processed/housing_zip.csv`
  - `data/processed/acs_controls.csv`
  - `data/processed/model_dataset.csv`
- Reports:
  - `reports/regression_coefficients.csv`
  - `reports/regression_metrics.csv`
  - `reports/home_value_vs_total_crime.png`
  - `reports/summary.md`

## Notes

- The default crime source is `https://www.dallasopendata.com/resource/qv6i-rri7.json`.
- The default ACS year is `2024`.
- Firecrawl is used through the CLI, not MCP.
- Full live acquisition can be slow because housing data is fetched per ZIP. Use `DCA_MAX_HOUSING_ZIPS` for bounded smoke runs.
- Runtime outputs under `data/`, `reports/`, `.firecrawl/`, `.matplotlib/`, `.venv/`, and `.uv-cache/` are intended to remain untracked.

## Suggested next steps

- Decide whether to keep the legacy R artifacts in the repo or archive them elsewhere.
- Tighten housing acquisition if you want broader ZIP coverage or parallel Firecrawl execution.
- Expand the report beyond the current regression + scatter output if you want a richer narrative notebook or dashboard.
