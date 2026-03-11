# Dallas Crime and Housing Refresh

This repository now contains a Python-first refresh of the original Dallas crime/housing project. The new workflow keeps the Dallas focus, but replaces the older R Markdown analysis with a reproducible data pipeline, explicit raw/processed dataset boundaries, and a small CLI.

The legacy coursework artifacts are still present in the repo for reference:

- `DallasArrests.Rmd`
- `DallasArrests.pdf`
- `DallasArrests_cleaned.csv`
- `Dallas_Arrests_Property_Summary.csv`
- `Police_Arrests.csv`
- `Zillow-dallas.csv`

## What the Python project does

- Pulls official Dallas crime data from Dallas OpenData.
- Pulls ZIP-level ACS controls from the Census API.
- Uses Firecrawl search + scrape against Zillow ZIP market pages to collect housing values.
- Aggregates crime to ZIP-level rates per 1,000 residents.
- Builds a merged model dataset and runs a robust OLS regression.
- Writes report artifacts to `reports/`.

## Project layout

- `src/dallas_crime/cli.py`: CLI entrypoint.
- `src/dallas_crime/acquire/`: source acquisition helpers.
- `src/dallas_crime/pipeline/`: build and analysis logic.
- `tests/`: fixture-backed acquisition and pipeline tests.

Runtime outputs are ignored by git:

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

Optional configuration lives in `.env.example`. The defaults already point at:

- Dallas OpenData `Police Incidents`
- Texas ZCTA ACS 5-year data
- Zillow ZIP home-value pages discovered through Firecrawl

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

Run the regression and write report outputs:

```bash
dallas-crime analyze
```

## Test

```bash
pytest
```

## Notes

- Firecrawl CLI is used directly rather than an MCP integration.
- The default crime window is the most recent 365 days, configurable via `DCA_CRIME_LOOKBACK_DAYS`.
- Live smoke runs can be bounded with `DCA_MAX_HOUSING_ZIPS`.
- `reports/summary.md` and the CSV/PNG outputs are generated artifacts and are intentionally ignored.
