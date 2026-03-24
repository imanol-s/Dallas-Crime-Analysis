"""Error-scenario and negative-path tests covering acquisition, pipeline, and analysis."""

from __future__ import annotations

import io
import json
import subprocess
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request

import pandas as pd
import pytest

from dallas_crime.acquire.census import (
    CensusRequest,
    fetch_census_payload,
    normalize_census_payload,
)
from dallas_crime.acquire.crime import (
    DallasCrimeSourceConfig,
    fetch_crime_payload,
)
from dallas_crime.acquire.housing import (
    run_firecrawl_command,
)
from dallas_crime.acquire.utils import (
    AcquisitionError as UtilsAcquisitionError,
    run_with_retry,
)
from dallas_crime.pipeline.build import (
    aggregate_crime_data,
    prepare_housing_features,
)
from dallas_crime.pipeline.analyze import run_zip_regression
from dallas_crime.pipeline.analyze.core import (
    _build_vif_artifacts,
    _select_expanded_controls,
)
from dallas_crime.pipeline.analyze.forecast import (
    _build_forecast_artifacts,
    _prepare_temporal_analysis_inputs,
)


# ── Acquire: HTTP error scenarios ──────────────────────────────────


def test_fetch_crime_payload_raises_on_404():
    """HTTPError 404 from Socrata should propagate as AcquisitionError."""
    config = DallasCrimeSourceConfig(
        dataset_url="https://example.com/resource/test",
        where_clause="1=1",
    )

    def mock_opener(request: Request):
        raise HTTPError(
            url=request.full_url,
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b""),
        )

    with pytest.raises(UtilsAcquisitionError, match="Dallas OpenData crime request"):
        fetch_crime_payload(
            config,
            opener=mock_opener,
            max_attempts=1,
            backoff_seconds=0.0,
        )


def test_fetch_crime_payload_raises_on_500():
    """HTTPError 500 from Socrata should propagate as AcquisitionError."""
    config = DallasCrimeSourceConfig(
        dataset_url="https://example.com/resource/test",
        where_clause="1=1",
    )

    def mock_opener(request: Request):
        raise HTTPError(
            url=request.full_url,
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b""),
        )

    with pytest.raises(UtilsAcquisitionError):
        fetch_crime_payload(
            config,
            opener=mock_opener,
            max_attempts=1,
            backoff_seconds=0.0,
        )


def test_fetch_census_payload_raises_on_404():
    """HTTPError 404 from Census API should propagate as AcquisitionError."""
    request = CensusRequest(
        year=2022,
        variables=["B01001_001E"],
        geography="zip code tabulation area:*",
    )

    def mock_opener(http_request: Request):
        raise HTTPError(
            url=http_request.full_url,
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b""),
        )

    with pytest.raises(UtilsAcquisitionError, match="Census ACS request"):
        fetch_census_payload(
            request,
            opener=mock_opener,
            max_attempts=1,
            backoff_seconds=0.0,
        )


def test_fetch_census_payload_raises_on_500():
    """HTTPError 500 from Census API should propagate as AcquisitionError."""
    request = CensusRequest(
        year=2022,
        variables=["B01001_001E"],
        geography="zip code tabulation area:*",
    )

    def mock_opener(http_request: Request):
        raise HTTPError(
            url=http_request.full_url,
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b""),
        )

    with pytest.raises(UtilsAcquisitionError):
        fetch_census_payload(
            request,
            opener=mock_opener,
            max_attempts=1,
            backoff_seconds=0.0,
        )


# ── Acquire: Malformed JSON scenarios ─────────────────────────────


def test_fetch_crime_payload_raises_on_malformed_json():
    """Non-JSON responses from Socrata should propagate as AcquisitionError."""
    config = DallasCrimeSourceConfig(
        dataset_url="https://example.com/resource/test",
        where_clause="1=1",
    )

    def mock_opener(request: Request):
        response = MagicMock()
        response.read.return_value = b"<html>not json</html>"
        return response

    with pytest.raises(UtilsAcquisitionError):
        fetch_crime_payload(
            config,
            opener=mock_opener,
            max_attempts=1,
            backoff_seconds=0.0,
        )


def test_fetch_crime_payload_raises_on_non_array_json():
    """JSON object (not array) from Socrata should raise AcquisitionError."""
    config = DallasCrimeSourceConfig(
        dataset_url="https://example.com/resource/test",
        where_clause="1=1",
    )

    def mock_opener(request: Request):
        response = MagicMock()
        response.read.return_value = json.dumps({"error": "bad request"}).encode()
        return response

    with pytest.raises(UtilsAcquisitionError):
        fetch_crime_payload(
            config,
            opener=mock_opener,
            max_attempts=1,
            backoff_seconds=0.0,
        )


def test_normalize_census_payload_raises_on_single_row():
    """Census payload with only the header row (no data) should return empty DataFrame."""
    payload = [["NAME", "B01001_001E", "zip code tabulation area"]]
    result = normalize_census_payload(payload)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ── Acquire: Timeout scenarios ─────────────────────────────────────


def test_run_firecrawl_command_raises_on_timeout():
    """subprocess.TimeoutExpired during Firecrawl should propagate as AcquisitionError."""
    with patch("dallas_crime.acquire.housing.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["firecrawl", "scrape"],
            timeout=5,
        )
        with pytest.raises(UtilsAcquisitionError, match="Firecrawl command"):
            run_firecrawl_command(
                ["firecrawl", "scrape", "--url", "https://example.com"],
                timeout_seconds=5,
                max_attempts=1,
                backoff_seconds=0.0,
            )


def test_run_with_retry_raises_after_exhausting_attempts():
    """run_with_retry should wrap the final exception in AcquisitionError."""
    call_count = 0

    def failing_fn():
        nonlocal call_count
        call_count += 1
        raise TimeoutError("simulated timeout")

    with pytest.raises(UtilsAcquisitionError, match="failed after"):
        run_with_retry(
            "test operation",
            failing_fn,
            max_attempts=2,
            backoff_seconds=0.0,
            retryable_exceptions=(TimeoutError,),
        )
    assert call_count == 2


# ── Pipeline: Missing columns in intermediate DataFrames ──────────


def test_run_zip_regression_raises_on_missing_columns():
    """run_zip_regression should raise KeyError when required columns are missing."""
    model_df = pd.DataFrame(
        {
            "zip": ["75201", "75202", "75203"],
            "home_value": [200000, 250000, 300000],
        }
    )
    with pytest.raises(KeyError, match="missing required regression columns"):
        run_zip_regression(model_df)


def test_run_zip_regression_raises_on_insufficient_rows():
    """run_zip_regression should raise ValueError when too few complete rows."""
    model_df = pd.DataFrame(
        {
            "zip": ["75201"],
            "home_value": [200000],
            "total_rate_per_1000": [30.0],
            "violent_rate_per_1000": [10.0],
            "property_rate_per_1000": [20.0],
            "median_household_income": [50000],
            "poverty_rate": [0.15],
            "owner_occupied_share": [0.4],
            "median_gross_rent": [1200],
            "educational_attainment": [0.3],
        }
    )
    with pytest.raises(ValueError, match="need at least"):
        run_zip_regression(model_df)


def test_select_expanded_controls_raises_when_no_candidates_available():
    """_select_expanded_controls should raise ValueError when no candidates have enough rows."""
    model_df = pd.DataFrame(
        {
            "zip": [f"75{i:03d}" for i in range(200, 210)],
            "home_value": [200000 + i * 10000 for i in range(10)],
            "violent_rate_per_1000": [10.0 - i * 0.5 for i in range(10)],
            "property_rate_per_1000": [20.0 - i * 0.5 for i in range(10)],
            "median_household_income": [50000 + i * 2000 for i in range(10)],
            "poverty_rate": [0.20 - i * 0.01 for i in range(10)],
            "owner_occupied_share": [0.30 + i * 0.02 for i in range(10)],
            "median_gross_rent": [1100 + i * 50 for i in range(10)],
            "educational_attainment": [0.25 + i * 0.02 for i in range(10)],
        }
    )
    # No expanded candidate columns present at all
    with pytest.raises(ValueError, match="unable to build an expanded-controls model"):
        _select_expanded_controls(
            model_df,
            dependent="log_home_value",
            predictors=("violent_rate_per_1000", "property_rate_per_1000"),
            baseline_controls=(
                "median_household_income",
                "poverty_rate",
                "owner_occupied_share",
                "median_gross_rent",
                "educational_attainment",
            ),
        )


# ── Pipeline: Empty DataFrames flowing through pipeline ────────────


def test_prepare_temporal_analysis_inputs_with_empty_panel():
    """Empty crime history panel should return empty summary and notes."""
    summary, series, notes = _prepare_temporal_analysis_inputs(pd.DataFrame())
    assert summary.empty
    assert series == {}
    assert any("not available" in note for note in notes)


def test_prepare_temporal_analysis_inputs_with_missing_columns():
    """Panel missing required columns should return empty summary and notes."""
    panel = pd.DataFrame({"zip": ["75201"], "some_other_column": [10]})
    summary, series, notes = _prepare_temporal_analysis_inputs(panel)
    assert summary.empty
    assert any("missing" in note for note in notes)


def test_build_forecast_artifacts_with_empty_summary():
    """Empty temporal summary should return empty forecast artifacts."""
    metrics, forecasts, intervals, notes = _build_forecast_artifacts(pd.DataFrame(), {})
    assert metrics.empty
    assert forecasts.empty
    assert intervals.empty
    assert any("unavailable" in note for note in notes)


def test_build_vif_artifacts_handles_single_row_model():
    """VIF should handle models with sample size too small for regressors."""
    from dallas_crime.pipeline.analyze.core import RegressionResult

    result = RegressionResult(
        model_label="tiny",
        formula="log_home_value ~ violent_rate_per_1000 + property_rate_per_1000",
        dependent_variable="log_home_value",
        predictors=("violent_rate_per_1000", "property_rate_per_1000"),
        controls=(),
        nobs=1,
        r_squared=0.0,
        adjusted_r_squared=0.0,
        coefficients=pd.DataFrame(),
        model_frame=pd.DataFrame(
            {
                "log_home_value": [12.2],
                "violent_rate_per_1000": [5.0],
                "property_rate_per_1000": [15.0],
            }
        ),
        residuals=pd.DataFrame(),
    )
    vif_table, notes = _build_vif_artifacts([result])
    assert vif_table.empty
    assert any("too small" in note for note in notes)


def test_aggregate_crime_data_preserves_structure_with_nan_dates():
    """Crime records with all-NaN dates should still produce a valid DataFrame."""
    crime = pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "reported_at": [pd.NaT, pd.NaT],
            "offense_family": ["violent", "property"],
            "incident_id": ["A1", "A2"],
        }
    )
    pop = pd.DataFrame({"zip": ["75201", "75202"], "population": [1000, 2000]})
    result = aggregate_crime_data(
        crime,
        pop,
        zip_col="zip",
        date_col="reported_at",
        category_col="offense_family",
    )
    assert isinstance(result, pd.DataFrame)


def test_prepare_housing_features_handles_all_null_home_values():
    """Housing features with all-null home_value should produce zero-row result."""
    housing = pd.DataFrame(
        {
            "zip": ["75201", "75202"],
            "home_value": [None, None],
            "source": ["zillow", "realtor"],
        }
    )
    result = prepare_housing_features(housing)
    assert isinstance(result, pd.DataFrame)
