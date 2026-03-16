# CLAUDE.md

Read `AGENTS.md` before making any changes — it is the authoritative guide for this repository.

## Parallel Code Review Pipeline

This project has three specialist agents mapped to its domain layers. When running a parallel review, use the Task tool to spawn all three concurrently.

### Agent: Data Engineer

**Scope:** `src/dallas_crime/acquire/` — acquisition layer (crime.py, census.py, housing.py, utils.py)
**Gate:** `ruff check src/dallas_crime/acquire/ tests/test_acquire.py` → `pytest -q tests/test_acquire.py`
**Fix loop:** up to 5 cycles. Report: status, files changed, any breaking changes to source contracts or `AcquisitionError` handling.

### Agent: Data Analyst

**Scope:** `src/dallas_crime/pipeline/build.py`, `src/dallas_crime/config.py` — transformations and settings
**Gate:** `ruff check src/dallas_crime/pipeline/build.py src/dallas_crime/config.py tests/test_pipeline.py` → `pytest -q tests/test_pipeline.py`
**Fix loop:** up to 5 cycles. Never remove existing `model_dataset.csv` columns — they are additive only.

### Agent: Data Scientist

**Scope:** `src/dallas_crime/pipeline/analyze/` — regressions, VIF, plots, reports (core.py, forecast.py, reporting.py, segmentation.py, spatial.py)
**Gate:** `ruff check src/dallas_crime/pipeline/analyze/ tests/test_project.py` → `pytest -q tests/test_project.py`
**Fix loop:** up to 5 cycles. HC3 robust standard errors (`cov_type="HC3"`) must remain on all regression fits.

### Orchestrator steps (after all 3 agents report green)

1. Run `make smoke` — must pass with no network access.
2. Run full `pytest -q` — must show `84 passed, 1 warning`.
3. Run `ruff check src/ tests/` — zero errors.
4. Commit with a summary of all agent changes.

## Project Type

This is a **data analytics / data science project**. Procedural and functional Python is the correct style. Do not over-engineer with OOP patterns, premature abstractions, or framework migrations. Verbosity in feature engineering is acceptable when it aids readability and auditability.

## Environment

- Always activate the project venv before running commands: use `uv run` or `.venv/bin/python`
- Lint: `uv tool run ruff check src/ tests/`
- Test: `uv run pytest -q`
- Smoke: `make smoke`

## Data Quality Principles

All pipeline outputs must satisfy these three principles. Verify on every data refresh.

### Accuracy
Values must be factually correct, with no duplicates or wrong types. Enforce:
- No duplicate ZIPs in model_dataset.csv
- Numeric columns contain only numeric types (no sentinel strings)
- Crime rates ≥ 0; population > 0 for all modeled ZIPs
- VIF < 10 for all active regression predictors

### Completeness
All required fields must be populated. Enforce:
- Cell-level missingness ≤ 5% in model_dataset.csv
- No column used in regression may exceed 30% null
- All modeled ZIPs must have crime, housing, and census records joined
- Source completeness scores computed and stored per ZIP

### Timeliness
Data must be available when downstream processes need it. Enforce:
- Partial-quarter data pro-rated before drift scoring (floor: 0.33 completeness)
- Quarters below 0.33 completeness excluded from drift (flag = -1)
- Crime history lag features require ≥ 8 quarters; forecast requires ≥ 12 quarters
- ACS snapshot features require ≥ 2 vintage years
