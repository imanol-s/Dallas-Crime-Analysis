"""Shared helpers for acquisition retries and JSON artifacts."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import io
import json
from pathlib import Path
import time
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


T = TypeVar("T")

REQUEST_TIMEOUT_SECONDS = 30


class AcquisitionError(RuntimeError):
    """Raised when a source acquisition step fails after retries."""


DFW_COUNTY_NAMES = frozenset(
    {
        "COLLIN COUNTY",
        "DALLAS COUNTY",
        "DENTON COUNTY",
        "ELLIS COUNTY",
        "ERATH COUNTY",
        "HOOD COUNTY",
        "HUNT COUNTY",
        "JOHNSON COUNTY",
        "KAUFMAN COUNTY",
        "NAVARRO COUNTY",
        "PALO PINTO COUNTY",
        "PARKER COUNTY",
        "ROCKWALL COUNTY",
        "SOMERVELL COUNTY",
        "TARRANT COUNTY",
        "WISE COUNTY",
    }
)
ZCTA_COUNTY_CROSSWALK_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
    "tab20_zcta520_county20_natl.txt"
)


def utc_timestamp() -> str:
    """Return a stable UTC ISO-8601 timestamp."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def run_with_retry(
    operation: str,
    action: Callable[[], T],
    *,
    max_attempts: int,
    backoff_seconds: float,
    retryable_exceptions: tuple[type[BaseException], ...],
    hint: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run an operation with deterministic exponential backoff."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be >= 0")

    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except retryable_exceptions as exc:
            if attempt == max_attempts:
                message = f"{operation} failed after {max_attempts} attempt(s). Last error: {exc}"
                if hint:
                    message = f"{message} {hint}"
                raise AcquisitionError(message) from exc

            wait_seconds = backoff_seconds * (2 ** (attempt - 1))
            if wait_seconds > 0:
                sleep(wait_seconds)

    raise AssertionError("run_with_retry exhausted without returning or raising")


def write_json_artifact(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact with stable formatting."""

    serialized = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(f"{serialized}\n", encoding="utf-8")
    return path


def _normalize_zip(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 5:
        return None
    zip_code = digits[:5]
    return zip_code if len(zip_code) == 5 else None


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_crosswalk_rows_from_cache(cache_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with cache_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            zip_code = _normalize_zip(
                row.get("zip") or row.get("GEOID_ZCTA5_20") or row.get("zcta") or row.get("zcta5")
            )
            if zip_code is None:
                continue
            county_name = str(
                row.get("county_name") or row.get("NAMELSAD_COUNTY_20") or row.get("county") or ""
            ).strip()
            if not county_name:
                continue
            if "area_score" in row:
                area_score = _to_float(row.get("area_score"))
            else:
                area_score = _to_float(row.get("AREALAND_PART")) + _to_float(
                    row.get("AREAWATER_PART")
                )
            rows.append({"zip": zip_code, "county_name": county_name, "area_score": area_score})
    return rows


def _download_crosswalk_rows(*, timeout_seconds: int) -> list[dict[str, object]]:
    request = Request(ZCTA_COUNTY_CROSSWALK_URL)
    with urlopen(request, timeout=timeout_seconds) as response:
        text_stream = io.TextIOWrapper(response, encoding="utf-8")
        reader = csv.DictReader(text_stream, delimiter="|")
        rows: list[dict[str, object]] = []
        for row in reader:
            zip_code = _normalize_zip(row.get("GEOID_ZCTA5_20"))
            if zip_code is None:
                continue
            county_name = str(row.get("NAMELSAD_COUNTY_20") or "").strip()
            if not county_name:
                continue
            area_score = _to_float(row.get("AREALAND_PART")) + _to_float(row.get("AREAWATER_PART"))
            rows.append({"zip": zip_code, "county_name": county_name, "area_score": area_score})
    return rows


def load_dfw_zip_set(
    zip_codes: list[str] | set[str] | tuple[str, ...],
    *,
    cache_path: Path | None = None,
    timeout_seconds: int = 15,
    max_attempts: int = 1,
    backoff_seconds: float = 0.0,
) -> set[str]:
    """Resolve candidate ZIP codes to the official NCTCOG 16-county DFW set."""

    normalized_targets = {
        zip_code for zip_code in (_normalize_zip(value) for value in zip_codes) if zip_code
    }
    if not normalized_targets:
        return set()

    rows: list[dict[str, object]]
    if cache_path is not None and cache_path.exists():
        rows = _iter_crosswalk_rows_from_cache(cache_path)
    else:
        rows = run_with_retry(
            "Census ZCTA-to-county crosswalk request",
            lambda: _download_crosswalk_rows(timeout_seconds=timeout_seconds),
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            retryable_exceptions=(
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                UnicodeDecodeError,
                csv.Error,
            ),
            hint="Verify Census bulk URL access and network availability.",
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["zip", "county_name", "area_score"])
                writer.writeheader()
                writer.writerows(rows)

    primary_county_by_zip: dict[str, tuple[str, float]] = {}
    for row in rows:
        zip_code = _normalize_zip(row.get("zip"))
        if zip_code is None or zip_code not in normalized_targets:
            continue
        county_name = str(row.get("county_name") or "").strip().upper()
        if not county_name:
            continue
        area_score = _to_float(row.get("area_score"))
        current = primary_county_by_zip.get(zip_code)
        if current is None or area_score >= current[1]:
            primary_county_by_zip[zip_code] = (county_name, area_score)

    return {
        zip_code
        for zip_code, (county_name, _area_score) in primary_county_by_zip.items()
        if county_name in DFW_COUNTY_NAMES
    }
