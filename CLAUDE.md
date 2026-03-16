# CLAUDE.md

Authoritative guide for this repository. Read this before making any changes.

---

## Project Overview

A reproducible ZIP-level crime and housing analysis pipeline for the Dallas metro area.
Python package at `src/dallas_crime/`, exposed via the `dallas-crime` CLI.
Stages: `acquire` → `build` → `analyze`.

This is a **data analytics / data science project**. Procedural and functional Python is the correct style. Do not over-engineer with OOP patterns, premature abstractions, or framework migrations. Verbosity in feature engineering is acceptable when it aids readability and auditability.

### Where Current Data Lives

| What | Location | Notes |
|------|----------|-------|
| Raw data | `data/raw/` (59 MB) | Crime, housing, ACS, sidecars |
| Processed datasets | `data/processed/` (2.8 MB) | `model_dataset.csv` is the primary input |
| Source code | `src/dallas_crime/` | acquire → pipeline → analyze |
| Analysis reports | `reports/` | Regression, forecasts, clusters, figures |
| Quick EDA visuals | `analysis_output/` | Exploratory plots |
| Tests | `tests/` | Unit + fixtures |
| Scripts | `scripts/` | dq_metrics, smoke fixture gen |
| Legacy R coursework | `archive/` | Gitignored |

Runtime outputs (`data/`, `reports/`, `.firecrawl/`, `.matplotlib/`) are **untracked** — never commit them.

Roadmap execution log: `task.md` is the line-by-line tracker; read it at session start with `RUNLOG.md`.

---

## Environment

- **Python ≥ 3.11**; target `py311` for ruff.
- Always use the project venv: `uv run` or `.venv/bin/python`.
- **No secrets in source.** API keys via environment variables only (see `.env.example`).

---

## Build / Test / Lint Commands

```bash
# Environment setup (run once)
uv venv && pip install -e ".[dev]"

# Full pipeline
make acquire       # fetch crime, ACS, housing raw data
make build         # transform raw → model_dataset.csv
make analyze       # run regressions → reports/

# Individual CLI commands (same as make targets)
.venv/bin/dallas-crime acquire
.venv/bin/dallas-crime build
.venv/bin/dallas-crime analyze
.venv/bin/dallas-crime show-config

# Tests
uv run pytest -q                                     # all tests (84 passed, 1 warning)
uv run pytest -q tests/test_pipeline.py              # single file
uv run pytest -q tests/test_pipeline.py::test_normalize_zip_variants  # single test
uv run pytest -q -k "keyword"                        # filter by name

# Smoke test (no network, no API keys required)
make smoke

# Lint (ruff — configured in pyproject.toml)
uv tool run ruff check src/ tests/
uv tool run ruff format src/ tests/
```

CI runs on every push/PR: `pytest -q` then `make smoke` (see `.github/workflows/ci.yml`).

---

## Project Structure

```
src/dallas_crime/
  cli.py              # Typer CLI entry point; thin dispatch only
  config.py           # Settings dataclass; no side effects
  acquire/
    crime.py          # Dallas OpenData crime fetch
    census.py         # ACS ZCTA fetch with bulk-table fallback
    housing.py        # Multi-source housing (Zillow → Realtor → Redfin)
    utils.py          # Shared acquisition helpers, AcquisitionError
  pipeline/
    build.py          # All transformations; build_all(settings) orchestrator
    analyze/
      core.py         # Regression, VIF, expanded controls; run_analysis(settings)
      forecast.py     # Crime trend forecasting, drift scoring
      reporting.py    # Report generation, policy recommendations
      segmentation.py # ZIP clustering, segment analysis
      spatial.py      # Spatial analysis, neighbor effects
tests/
  test_pipeline.py    # Build + analyze unit/integration tests
  test_project.py     # End-to-end run_analysis() artifact tests
  test_acquire.py     # Acquisition layer unit tests
scripts/
  dq_metrics.py       # Data quality metrics report
  create_smoke_inputs.py  # Generates minimal raw CSVs for offline smoke tests
data/raw/
  dfw_zip_enrichment.csv  # Stable enrichment sidecar; joined in build_all()
```

---

## Skills

| Skill | Purpose |
|-------|---------|
| `/dq-audit` | Full data quality audit (accuracy, completeness, timeliness, bounds, VIF) |
| `/data-observability-gate` | Classify and validate data-layer changes before commit |

---

## Code Style

### Imports

- Every module starts with `from __future__ import annotations`.
- Standard library first, then third-party (`numpy`, `pandas`, `statsmodels`, `matplotlib`), then intra-package.
- Use `TYPE_CHECKING` guard for imports that only serve type annotations:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from dallas_crime.config import Settings
  ```

### Formatting

- Line length: **100 characters** (ruff enforced).
- Ruff is the sole formatter and linter — no Black, no isort separately.
- Trailing commas in multi-line collections.

### Type annotations

- All public functions must have full parameter and return type annotations.
- Use `tuple[str, ...]` not `Tuple[str, ...]`; use `list[str]` not `List[str]` (PEP 585).
- `pd.DataFrame`, `pd.Series`, `Path` are preferred over `Any`.
- `@dataclass(slots=True)` for result/payload objects (see `RegressionResult`).

### Naming conventions

- Modules, variables, functions: `snake_case`.
- Private helpers: prefixed with `_` (e.g., `_coerce_model_columns`, `_write_scatter_plot`).
- Public-facing module-level constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CONTROLS`, `MISSING_STRINGS`).
- Immutable constant sets: use `frozenset` (e.g., `DEFAULT_VIOLENT_LABELS`).
- Tuple constants preferred over list for fixed sequences of control/predictor names.

### Keyword-only arguments

Multi-parameter public functions must use `*` to enforce keyword-only call sites:
```python
def run_zip_regression(
    model_df: pd.DataFrame,
    *,
    dependent: str = "log_home_value",
    predictors: tuple[str, ...] = DEFAULT_PREDICTORS,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    model_label: str = "baseline",
) -> RegressionResult:
```

### Error handling

- Raise `KeyError` for missing required DataFrame columns; include the column name(s) in the message.
- Raise `ValueError` for invalid inputs or insufficient rows.
- Raise `FileNotFoundError` for missing required input files.
- Use `AcquisitionError` (from `dallas_crime.acquire.utils`) for acquisition-layer failures.
- Numeric coercion: always use `pd.to_numeric(..., errors="coerce")` and `pd.to_datetime(..., errors="coerce")` — never raise on bad data; NaN-out instead.
- In VIF/stats blocks, catch `(TypeError, ValueError, np.linalg.LinAlgError)` together and record a note rather than crashing.

### Pandas conventions

- After any dedup or sort, call `.reset_index(drop=True)`.
- Use `np.where(condition, a, b)` for vectorized conditional assignment instead of pandas `.where()`.
- Use `_safe_divide(numerator, denominator)` helper (defined in `build.py`) for ratio columns — avoids ZeroDivisionError.
- Always `.copy()` a frame before mutating it (avoids `SettingWithCopyWarning`).
- Keep I/O out of transformation functions: `build.py` helpers are pure transforms; `build_all()` does all file reads/writes.

### Matplotlib

- Always call `matplotlib.use("Agg")` before importing `pyplot` for headless safety.
- Always call `plt.close(fig)` after `fig.savefig(...)`.
- Use `os.environ.setdefault("MPLCONFIGDIR", ...)` to redirect the config dir.

---

## Architecture Rules

- **`cli.py` is thin.** No business logic; dispatches to `build_all(settings)` and `run_analysis(settings)` only.
- **`config.py` is pure.** `Settings` is a dataclass with no side effects; loaded once and passed down.
- **Transformations are I/O-free.** `build.py` helpers take DataFrames and return DataFrames; `build_all()` owns all file I/O.
- **Enrichment sidecar.** `data/raw/dfw_zip_enrichment.csv` is stable raw enrichment; `build_all()` left-joins it after `build_model_dataset()`. Idempotent — no-op if file absent. Never discard it.
- **Expanded controls are selected dynamically.** `_select_expanded_controls()` in `core.py` iterates `EXPANDED_CONTROL_CANDIDATES` and accepts a candidate only if it leaves ≥ (n_columns + 1) complete rows. Do not hard-code the expanded formula.
- **Regression uses HC3 robust standard errors.** Always `smf.ols(...).fit(cov_type="HC3")`.
- **Columns are additive.** Never remove an existing `model_dataset.csv` column.

---

## Testing Guidelines

- Tests live in `tests/`; run with `uv run pytest -q`.
- Use `tmp_path` (pytest fixture) for integration tests that write files.
- Use `unittest.mock.patch` to stub network calls in acquisition tests.
- Always assert DataFrame shape and specific column values — do not just assert "not empty".
- The smoke script (`scripts/create_smoke_inputs.py`) generates offline-safe minimal inputs; CI depends on it. Do not break its column contracts.

---

## Key Invariants (do not break)

- `uv run pytest -q` must pass (`84 passed, 1 warning`) before any commit.
- `make smoke` must pass with no network access.
- `data/raw/dfw_zip_enrichment.csv` must remain intact (70 rows, 11 enrichment columns).
- Regression model sample sizes: baseline n=70, expanded n=61 (current; will vary after re-acquire).

---

## Parallel Code Review Pipeline

Three specialist agents mapped to domain layers. Spawn all three concurrently via the Task tool.

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

1. `make smoke` — must pass with no network access.
2. `uv run pytest -q` — must show `84 passed, 1 warning`.
3. `uv tool run ruff check src/ tests/` — zero errors.
4. Commit with a summary of all agent changes.

---

## Data Quality Principles

All pipeline outputs must satisfy accuracy, completeness, and timeliness. Run `/dq-audit` to verify, or `/data-observability-gate` to classify a specific change.

### Key thresholds (`build_model_dataset` validation contract)

```yaml
null_thresholds:
  zip: 0.00
  home_value: 0.05
  total_rate_per_1000: 0.05
  median_household_income: 0.10
  poverty_rate: 0.10
row_count:
  min: 50
  max: 100
statistical_bounds:
  home_value: {min: 50000, max: 3000000}       # Highland Park ZIPs reach ~$2M
  total_rate_per_1000: {min: 0, max: 2500}     # small-pop downtown ZIPs inflate rate
  population: {min: 1}
```

### Guards summary

- **Accuracy:** No duplicate ZIPs (`_dedupe_zip_rows`), numeric coercion, rates >= 0, VIF < 10.
- **Completeness:** Cell-level <= 5% null, regression columns <= 30% null, ZIP join integrity via inner join, per-ZIP completeness scores.
- **Timeliness:** Pro-rate partial quarters (floor 0.33), history depth >= 8 for lag-4, forecast >= 12 quarters, ACS >= 2 vintages.
