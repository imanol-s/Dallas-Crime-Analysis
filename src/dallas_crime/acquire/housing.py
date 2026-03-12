"""Firecrawl-backed housing acquisition helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from dallas_crime.acquire.utils import AcquisitionError, run_with_retry, utc_timestamp, write_json_artifact

if TYPE_CHECKING:
    from dallas_crime.config import Settings

_ZIP_PATTERN = re.compile(r"\b(\d{5})\b")
_PRICE_PATTERN = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([KMB])?", re.IGNORECASE)
_DATE_PATTERN = re.compile(
    r"(?:Updated on |Key indicators as of |In |\()"
    r"(\d{1,2}/\d{1,2}/\d{4}|[A-Z][a-z]+ \d{1,2}, \d{4}|[A-Z][a-z]+ \d{4})"
)
_CHANGE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)%\s+(?:1-yr|over the past year)", re.IGNORECASE)
_UP_DOWN_CHANGE_PATTERN = re.compile(r"\b(up|down)\s+(\d+(?:\.\d+)?)%\s+(?:since|compared to)\s+last year", re.IGNORECASE)
_YEAR_OVER_YEAR_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?)%\s+year-over-year", re.IGNORECASE)
_REDFIN_SALE_SENTENCE_PATTERN = re.compile(
    r"median sale price of a home in \d{5} was\s+(\$\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*[KMB]?)",
    re.IGNORECASE,
)
_REDFIN_PRICE_SECTION_PATTERN = re.compile(
    r"Median Sale Price\s+(\$\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*[KMB]?)",
    re.IGNORECASE,
)
_REALTOR_HOME_ROW_PATTERN = re.compile(
    r"\|\s*Median home \$\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
    re.IGNORECASE,
)
_REALTOR_RENT_ROW_PATTERN = re.compile(
    r"\|\s*Median rent\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
    re.IGNORECASE,
)
_SOURCE_PRIORITY = {"zillow": 0, "realtor": 1, "redfin": 2, "unknown": 99}
_SOURCE_URLS = {
    "zillow": "https://www.zillow.com/home-values/",
    "realtor": "https://www.realtor.com/local/market/texas/zipcode-{zip}",
    "realtor_csv": "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip.csv",
    "realtor_history": "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip_History.csv",
    "fhfa_zip5": "https://www.fhfa.gov/hpi/download/annual/hpi_at_zip5.xlsx",
    "redfin": "https://www.redfin.com/zipcode/{zip}/housing-market",
}
_METRIC_LABELS = {
    "zillow": "typical_home_value",
    "realtor": "median_home_price",
    "realtor_csv": "median_listing_price",
    "realtor_history": "median_listing_price",
    "fhfa_zip5": "fhfa_hpi",
    "redfin": "median_sale_price",
    "unknown": "housing_price_signal",
}
_SOURCE_PRIORITY["realtor_csv"] = 2
_SOURCE_PRIORITY["redfin"] = 3
_REALTOR_INVENTORY_COLUMNS = {
    "month_date_yyyymm": "realtor_month_date_yyyymm",
    "postal_code": "zip",
    "median_listing_price": "realtor_listing_price",
    "median_listing_price_yy": "realtor_listing_price_yy",
    "active_listing_count": "realtor_active_listing_count",
    "median_days_on_market": "realtor_median_days_on_market",
    "median_listing_price_per_square_foot": "realtor_listing_price_per_square_foot",
    "pending_ratio": "realtor_pending_ratio",
    "quality_flag": "realtor_quality_flag",
}
_REALTOR_HISTORY_COLUMNS = {
    "month_date_yyyymm": "realtor_hist_month_date_yyyymm",
    "postal_code": "zip",
    "median_listing_price": "realtor_hist_listing_price",
    "active_listing_count": "realtor_hist_active_listing_count",
    "median_days_on_market": "realtor_hist_median_days_on_market",
    "pending_ratio": "realtor_hist_pending_ratio",
    "quality_flag": "realtor_hist_quality_flag",
}

_METRIC_PATTERNS = {
    "home_value": (
        re.compile(r"(typical|average)\s+home\s+value", re.IGNORECASE),
        re.compile(r"zillow\s+home\s+value", re.IGNORECASE),
    ),
    "median_rent": (
        re.compile(r"median\s+rent", re.IGNORECASE),
        re.compile(r"average\s+rent", re.IGNORECASE),
    ),
}


@dataclass(frozen=True)
class FirecrawlSearchRequest:
    """Search request for Zillow market pages."""

    query: str
    limit: int = 5
    country: str = "US"
    scrape: bool = True
    scrape_formats: tuple[str, ...] = ("markdown",)
    only_main_content: bool = True
    output_path: str | None = None
    json_output: bool = True


@dataclass(frozen=True)
class FirecrawlScrapeRequest:
    """Direct scrape request shape retained for future expansion."""

    urls: tuple[str, ...]
    formats: tuple[str, ...] = ("markdown",)
    only_main_content: bool = True
    country: str = "US"
    wait_for_ms: int | None = 3000
    output_path: str | None = None
    json_output: bool = True


def build_firecrawl_search_command(request: FirecrawlSearchRequest) -> list[str]:
    """Build a Firecrawl search command."""

    command = [
        "firecrawl",
        "search",
        request.query,
        "--limit",
        str(request.limit),
        "--country",
        request.country,
    ]
    if request.scrape:
        command.append("--scrape")
        command.extend(["--scrape-formats", ",".join(request.scrape_formats)])
    if request.only_main_content:
        command.append("--only-main-content")
    if request.output_path:
        command.extend(["-o", str(request.output_path)])
    if request.json_output:
        command.append("--json")
    return command


def build_firecrawl_scrape_command(request: FirecrawlScrapeRequest) -> list[str]:
    """Build a Firecrawl scrape command."""

    command = [
        "firecrawl",
        "scrape",
        *request.urls,
        "--format",
        ",".join(request.formats),
        "--country",
        request.country,
    ]
    if request.only_main_content:
        command.append("--only-main-content")
    if request.wait_for_ms is not None:
        command.extend(["--wait-for", str(request.wait_for_ms)])
    if request.output_path:
        command.extend(["-o", str(request.output_path)])
    if request.json_output:
        command.append("--json")
    return command


def run_firecrawl_command(
    command: Sequence[str],
    *,
    timeout_seconds: int = 90,
    max_attempts: int = 1,
    backoff_seconds: float = 0.0,
) -> Any:
    """Run a Firecrawl CLI command and decode JSON when possible."""

    def _run_command() -> Any:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            error_text = (completed.stderr or completed.stdout).strip()
            raise ValueError(
                f"Firecrawl command exited with code {completed.returncode}: {error_text}"
            )

        stdout = completed.stdout.strip()
        if not stdout:
            return {}
        if stdout.lower().startswith("no results found"):
            return {}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Firecrawl command did not return valid JSON while --json was requested."
            ) from exc

    return run_with_retry(
        "Firecrawl command",
        _run_command,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        retryable_exceptions=(subprocess.TimeoutExpired, OSError, ValueError),
        hint=(
            "Verify FIRECRAWL_API_KEY and local `firecrawl` CLI availability, and consider raising "
            "DCA_ACQUIRE_MAX_ATTEMPTS or DCA_FIRECRAWL_TIMEOUT_SECONDS."
        ),
    )


def parse_firecrawl_search_results(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize Firecrawl search output across common response shapes."""

    if isinstance(payload, Mapping):
        data = payload.get("data") or payload.get("results") or []
        if isinstance(data, Mapping):
            items = data.get("web") or data.get("documents") or data.get("results") or []
        else:
            items = data
    else:
        items = payload

    normalized: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        normalized.append(
            {
                "title": item.get("title"),
                "url": item.get("url") or item.get("sourceURL"),
                "description": item.get("description"),
            }
        )
    return normalized


def normalize_firecrawl_documents(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    """Extract housing metrics from scraped Firecrawl documents."""

    if isinstance(payload, Mapping):
        data = payload.get("data") or payload.get("documents") or payload
        if isinstance(data, Mapping):
            documents = data.get("web") or data.get("documents") or [payload]
        else:
            documents = list(data)
    else:
        documents = list(payload)

    rows = [
        extract_housing_metrics(
            _extract_markdown(doc),
            url=_extract_url(doc),
            description=_extract_description(doc),
        )
        for doc in documents
    ]
    metric_keys = {"home_value", "median_rent", "annual_change_pct", "as_of_date"}
    normalized = [
        row
        for row in rows
        if any(row.get(metric) is not None for metric in metric_keys)
    ]
    return pd.DataFrame.from_records(normalized)


def extract_housing_metrics(
    markdown: str,
    *,
    url: str | None = None,
    zip_code: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Extract a compact housing metric record from Firecrawl markdown."""

    text = "\n".join(part for part in (description or "", markdown or "") if part)
    resolved_zip = zip_code or _extract_zip(text) or _extract_zip(url or "")
    source_name = _infer_source_name(url)
    record: dict[str, Any] = {
        "zip": resolved_zip,
        "source_url": url,
        "home_value": None,
        "median_rent": None,
        "annual_change_pct": None,
        "as_of_date": None,
        "source": source_name,
        "metric_label": _METRIC_LABELS[source_name],
        "supplemental_sources": None,
    }

    if source_name == "realtor":
        record.update(_extract_realtor_metrics(text))
    elif source_name == "redfin":
        record.update(_extract_redfin_metrics(text))
    else:
        record.update(_extract_zillow_metrics(text))

    if record["annual_change_pct"] is None:
        record["annual_change_pct"] = _extract_percent(text)
    if record["as_of_date"] is None:
        record["as_of_date"] = _extract_date(text)
    return record


def _load_requested_zip_codes(settings: "Settings") -> list[str]:
    crime_path = settings.raw_dir / "crime_records.csv"
    if not crime_path.exists():
        raise AcquisitionError(
            f"Crime raw dataset not found at {crime_path}. Run crime acquisition before housing."
        )

    crime_frame = pd.read_csv(crime_path, usecols=["zip"])
    zip_counts = (
        crime_frame["zip"]
        .dropna()
        .astype(str)
        .map(lambda value: str(value).zfill(5))
        .where(lambda series: series.str.fullmatch(r"\d{5}"))
        .dropna()
        .value_counts()
    )
    zip_codes = sorted(
        zip_code
        for zip_code, count in zip_counts.items()
        if count >= settings.min_total_incidents_per_zip and settings.allows_study_zip(zip_code)
    )
    if not zip_codes:
        raise AcquisitionError(
            "No ZIP codes met the crime incident threshold for housing lookup. "
            "Lower DCA_MIN_TOTAL_INCIDENTS_PER_ZIP or refresh the crime dataset."
        )
    if settings.max_housing_zips:
        zip_codes = zip_codes[: settings.max_housing_zips]
    return zip_codes


def fetch_housing_dataset(settings: "Settings") -> str:
    """Scrape housing market signals with Zillow first, then direct-source fallbacks."""

    zip_codes = _load_requested_zip_codes(settings)
    if settings.housing_scrape_batch_size < 1:
        raise AcquisitionError("DCA_HOUSING_SCRAPE_BATCH_SIZE must be >= 1.")

    records: list[dict[str, Any]] = []
    status_by_zip: dict[str, dict[str, Any]] = {
        zip_code: {"zip": zip_code, "status": "pending", "records": 0}
        for zip_code in zip_codes
    }
    records.extend(_collect_zillow_records(settings, zip_codes, status_by_zip))

    merged = _merge_housing_records(records)
    remaining_zips = _remaining_zip_codes(zip_codes, merged)

    for source_name, builder in (("realtor", _build_realtor_market_url), ("redfin", _build_redfin_market_url)):
        if not remaining_zips:
            break
        records.extend(
            _collect_direct_source_records(
                settings,
                remaining_zips,
                status_by_zip,
                source_name=source_name,
                url_builder=builder,
            )
        )
        merged = _merge_housing_records(records)
        remaining_zips = _remaining_zip_codes(zip_codes, merged)

    print("[housing] loading Realtor current ZIP inventory feed.", flush=True)
    realtor_inventory = fetch_realtor_zip_inventory(
        zip_codes,
        timeout_seconds=settings.acquire_timeout_seconds,
        max_attempts=settings.acquire_max_attempts,
        backoff_seconds=settings.acquire_backoff_seconds,
    )
    local_history_path = settings.raw_dir / "housing_market_history.csv"
    realtor_history = _load_local_realtor_history_summary(local_history_path, zip_codes)
    if realtor_history.empty:
        print("[housing] loading Realtor historical ZIP feed.", flush=True)
        realtor_history = fetch_realtor_zip_history(
            zip_codes,
            timeout_seconds=settings.acquire_timeout_seconds,
            max_attempts=settings.acquire_max_attempts,
            backoff_seconds=settings.acquire_backoff_seconds,
        )
    else:
        print(
            f"[housing] reusing local Realtor historical ZIP panel from {local_history_path}.",
            flush=True,
        )
    if not realtor_inventory.empty:
        csv_records = _realtor_inventory_records(realtor_inventory, remaining_zips)
        if csv_records:
            records.extend(csv_records)
            merged = _merge_housing_records(records)
            remaining_zips = _remaining_zip_codes(zip_codes, merged)
        housing = _merge_realtor_inventory_features(merged, realtor_inventory)
    else:
        housing = merged.copy()
    if not realtor_history.empty:
        housing = _merge_realtor_history_features(housing, realtor_history)

    received_zips = (
        sorted(
            {
                str(value).zfill(5)
                for value in housing["zip"].dropna().astype(str)
                if re.fullmatch(r"\d{5}", str(value).zfill(5))
            }
        )
        if not housing.empty
        else []
    )
    missing_zips = sorted(set(zip_codes) - set(received_zips))
    coverage_rate = (len(received_zips) / len(zip_codes)) if zip_codes else 0.0
    coverage_path = settings.raw_dir / "housing_zip_coverage.json"
    record_counts = _record_counts_by_zip(records)
    for zip_code in zip_codes:
        status = status_by_zip[zip_code]
        if zip_code in received_zips:
            status_by_zip[zip_code] = {"zip": zip_code, "status": "ok", "records": int(record_counts.get(zip_code, 1))}
        elif status.get("status") == "pending":
            status_by_zip[zip_code] = {"zip": zip_code, "status": "no_metrics", "records": int(record_counts.get(zip_code, 0))}
    zip_status = [status_by_zip[zip_code] for zip_code in zip_codes]
    write_json_artifact(
        coverage_path,
        {
            "dataset": "housing_market",
            "retrieved_at_utc": utc_timestamp(),
            "requested_zips": zip_codes,
            "received_zips": received_zips,
            "missing_zips": missing_zips,
            "requested_zip_count": len(zip_codes),
            "received_zip_count": len(received_zips),
            "coverage_rate": round(coverage_rate, 4),
            "zip_status": zip_status,
        },
    )

    output_path = settings.raw_dir / "housing_market.csv"
    if not housing.empty:
        housing.to_csv(output_path, index=False)

    metadata_path = settings.raw_dir / "housing_market.metadata.json"
    write_json_artifact(
        metadata_path,
        {
            "dataset": "housing_market",
            "retrieved_at_utc": utc_timestamp(),
            "source_url": _SOURCE_URLS["zillow"],
            "query": {
                "strategy": "search_then_direct_fallbacks",
                "query_template": settings.housing_query_template,
                "max_housing_zips": settings.max_housing_zips,
                "source_priority": ["zillow", "realtor", "redfin"],
                "discovery_tool": "search_and_direct_scrape",
                "search_scrape_enabled": True,
                "search_batch_size": settings.housing_scrape_batch_size,
                "search_limit_per_batch": max(settings.housing_scrape_batch_size * 4, 20),
                "scrape_formats": ["markdown"],
            },
            "row_counts": {
                "records_extracted": len(records),
                "records_written": int(len(housing)),
                "requested_zip_count": len(zip_codes),
                "received_zip_count": len(received_zips),
            },
            "coverage_summary": {
                "requested_zip_codes": len(zip_codes),
                "received_zip_codes": len(received_zips),
                "missing_zip_codes": len(missing_zips),
                "coverage_rate": round(coverage_rate, 4),
            },
            "source_summary": {
                "zillow": int((housing["source"] == "zillow").sum()) if not housing.empty else 0,
                "realtor": int((housing["source"] == "realtor").sum()) if not housing.empty else 0,
                "realtor_csv": int((housing["source"] == "realtor_csv").sum()) if not housing.empty else 0,
                "redfin": int((housing["source"] == "redfin").sum()) if not housing.empty else 0,
            },
            "source_url_patterns": {name: url for name, url in _SOURCE_URLS.items()},
            "realtor_inventory": {
                "source_url": _SOURCE_URLS["realtor_csv"],
                "matched_zip_count": int(realtor_inventory["zip"].nunique()) if not realtor_inventory.empty else 0,
                "latest_as_of_date": (
                    realtor_inventory["realtor_as_of_date"].max().date().isoformat()
                    if not realtor_inventory.empty and realtor_inventory["realtor_as_of_date"].notna().any()
                    else None
                ),
                "non_null_feature_counts": {
                    "realtor_listing_price": int(realtor_inventory["realtor_listing_price"].notna().sum())
                    if "realtor_listing_price" in realtor_inventory
                    else 0,
                    "realtor_active_listing_count": int(realtor_inventory["realtor_active_listing_count"].notna().sum())
                    if "realtor_active_listing_count" in realtor_inventory
                    else 0,
                    "realtor_median_days_on_market": int(
                        realtor_inventory["realtor_median_days_on_market"].notna().sum()
                    )
                    if "realtor_median_days_on_market" in realtor_inventory
                    else 0,
                    "realtor_pending_ratio": int(realtor_inventory["realtor_pending_ratio"].notna().sum())
                    if "realtor_pending_ratio" in realtor_inventory
                    else 0,
                },
            },
            "realtor_history": {
                "source_url": _SOURCE_URLS["realtor_history"],
                "matched_zip_count": int(realtor_history["zip"].nunique()) if not realtor_history.empty else 0,
                "feature_non_null_counts": {
                    column: int(realtor_history[column].notna().sum())
                    for column in realtor_history.columns
                    if column.startswith("realtor_hist_") and column != "realtor_hist_as_of_date"
                },
            },
            "sources": [
                {
                    "name": "zillow",
                    "url_pattern": _SOURCE_URLS["zillow"],
                    "metric_label": _METRIC_LABELS["zillow"],
                    "records_written": int((housing["source"] == "zillow").sum()) if not housing.empty else 0,
                },
                {
                    "name": "realtor",
                    "url_pattern": _SOURCE_URLS["realtor"],
                    "metric_label": _METRIC_LABELS["realtor"],
                    "records_written": int((housing["source"] == "realtor").sum()) if not housing.empty else 0,
                },
                {
                    "name": "realtor_csv",
                    "url_pattern": _SOURCE_URLS["realtor_csv"],
                    "metric_label": _METRIC_LABELS["realtor_csv"],
                    "records_written": int((housing["source"] == "realtor_csv").sum()) if not housing.empty else 0,
                },
                {
                    "name": "redfin",
                    "url_pattern": _SOURCE_URLS["redfin"],
                    "metric_label": _METRIC_LABELS["redfin"],
                    "records_written": int((housing["source"] == "redfin").sum()) if not housing.empty else 0,
                },
            ],
            "coverage_artifact": str(coverage_path),
            "output_path": str(output_path),
        },
    )
    fetch_housing_history_dataset(settings, zip_codes=zip_codes)

    if housing.empty:
        raise AcquisitionError(
            "Firecrawl returned no usable housing metrics for the configured ZIP set. "
            f"Review per-ZIP statuses in {coverage_path}."
        )

    return str(output_path)


def _chunked(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _build_batch_housing_query(settings: "Settings", zip_codes: Sequence[str]) -> str:
    if not zip_codes:
        return settings.housing_query_template
    if "{zip}" in settings.housing_query_template:
        base = settings.housing_query_template.replace("{zip}", "").strip()
    else:
        base = settings.housing_query_template.strip()
    if len(zip_codes) == 1:
        zip_clause = zip_codes[0]
    else:
        zip_clause = f"({' OR '.join(zip_codes)})"
    return " ".join([base, zip_clause]).strip()


def fetch_realtor_zip_inventory(
    zip_codes: Sequence[str],
    *,
    timeout_seconds: int = 60,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch structured Realtor ZIP inventory metrics for the requested ZIPs."""

    if not zip_codes:
        return pd.DataFrame(columns=["zip"])

    source_path = _download_remote_tabular_source(
        _SOURCE_URLS["realtor_csv"],
        suffix=".csv",
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )
    try:
        frame = pd.read_csv(
            source_path,
            usecols=list(_REALTOR_INVENTORY_COLUMNS.keys()),
            dtype={"postal_code": str, "month_date_yyyymm": str},
        )
    finally:
        source_path.unlink(missing_ok=True)
    frame = frame.rename(columns=_REALTOR_INVENTORY_COLUMNS)
    frame["zip"] = frame["zip"].astype(str).str.zfill(5)
    frame = frame[frame["zip"].isin(set(zip_codes))].copy()
    if frame.empty:
        return frame

    frame["realtor_as_of_date"] = pd.to_datetime(frame["realtor_month_date_yyyymm"], format="%Y%m", errors="coerce")
    numeric_columns = [
        "realtor_listing_price",
        "realtor_listing_price_yy",
        "realtor_active_listing_count",
        "realtor_median_days_on_market",
        "realtor_listing_price_per_square_foot",
        "realtor_pending_ratio",
        "realtor_quality_flag",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["zip", "realtor_as_of_date"]).drop_duplicates(subset=["zip"], keep="last")
    return frame.reset_index(drop=True)


def fetch_realtor_zip_history(
    zip_codes: Sequence[str],
    *,
    chunksize: int = 250_000,
    timeout_seconds: int = 60,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Stream the Realtor ZIP history CSV and compute recent market features."""

    if not zip_codes:
        return pd.DataFrame(columns=["zip"])

    filtered_chunks = _read_realtor_history_chunks(
        zip_codes,
        chunksize=chunksize,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )

    if not filtered_chunks:
        return pd.DataFrame(columns=["zip"])

    frame = pd.concat(filtered_chunks, ignore_index=True)
    frame["realtor_hist_as_of_date"] = pd.to_datetime(
        frame["realtor_hist_month_date_yyyymm"], format="%Y%m", errors="coerce"
    )
    numeric_columns = [
        "realtor_hist_listing_price",
        "realtor_hist_active_listing_count",
        "realtor_hist_median_days_on_market",
        "realtor_hist_pending_ratio",
        "realtor_hist_quality_flag",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["realtor_hist_as_of_date"]).sort_values(["zip", "realtor_hist_as_of_date"])
    return _summarize_realtor_history(frame)


def fetch_realtor_zip_history_panel(
    zip_codes: Sequence[str],
    *,
    year_floor: int = 2000,
    year_ceiling: int = 2025,
    chunksize: int = 250_000,
    timeout_seconds: int = 60,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Stream monthly Realtor ZIP history as a long panel."""

    if not zip_codes:
        return pd.DataFrame(columns=["zip"])

    filtered_chunks = _read_realtor_history_chunks(
        zip_codes,
        chunksize=chunksize,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )

    if not filtered_chunks:
        return pd.DataFrame(columns=["zip"])

    frame = pd.concat(filtered_chunks, ignore_index=True)
    frame["period_start"] = pd.to_datetime(frame["realtor_hist_month_date_yyyymm"], format="%Y%m", errors="coerce")
    frame = frame.dropna(subset=["period_start"]).copy()
    frame = frame[
        frame["period_start"].dt.year.between(year_floor, year_ceiling, inclusive="both")
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=["zip"])

    numeric_columns = [
        "realtor_hist_listing_price",
        "realtor_hist_active_listing_count",
        "realtor_hist_median_days_on_market",
        "realtor_hist_pending_ratio",
        "realtor_hist_quality_flag",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["period_end"] = frame["period_start"] + pd.offsets.MonthEnd(0)
    frame["period_year"] = frame["period_start"].dt.year.astype("Int64")
    frame["period_month"] = frame["period_start"].dt.month.astype("Int64")
    frame["frequency"] = "monthly"
    frame["source"] = "realtor_history"
    frame["source_url"] = _SOURCE_URLS["realtor_history"]
    frame["metric_label"] = _METRIC_LABELS["realtor_history"]
    frame["price_signal_value"] = frame["realtor_hist_listing_price"]
    frame["price_signal_unit"] = "usd_median_listing_price"

    columns = [
        "zip",
        "period_start",
        "period_end",
        "period_year",
        "period_month",
        "frequency",
        "source",
        "source_url",
        "metric_label",
        "price_signal_value",
        "price_signal_unit",
        "realtor_hist_listing_price",
        "realtor_hist_active_listing_count",
        "realtor_hist_median_days_on_market",
        "realtor_hist_pending_ratio",
        "realtor_hist_quality_flag",
    ]
    return frame[columns].sort_values(["zip", "period_start"], ignore_index=True)


def fetch_fhfa_zip5_history(
    zip_codes: Sequence[str],
    *,
    year_floor: int = 2000,
    year_ceiling: int = 2025,
    timeout_seconds: int = 60,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch official FHFA annual ZIP5 HPI history filtered to the requested ZIPs."""

    if not zip_codes:
        return pd.DataFrame(columns=["zip"])

    source_path = _download_remote_tabular_source(
        _SOURCE_URLS["fhfa_zip5"],
        suffix=".xlsx",
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )
    try:
        frame = pd.read_excel(
            source_path,
            sheet_name="ZIP5",
            skiprows=4,
            dtype={0: str},
        )
    finally:
        source_path.unlink(missing_ok=True)
    frame = frame.iloc[:, :6].copy()
    frame.columns = [
        "zip",
        "period_year",
        "fhfa_annual_change_pct",
        "fhfa_hpi",
        "fhfa_hpi_1990_base",
        "fhfa_hpi_2000_base",
    ]
    frame["zip"] = frame["zip"].astype(str).str.zfill(5)
    frame["period_year"] = pd.to_numeric(frame["period_year"], errors="coerce").astype("Int64")
    frame = frame[frame["zip"].isin(set(zip_codes))].copy()
    frame = frame[
        frame["period_year"].notna() & frame["period_year"].between(year_floor, year_ceiling, inclusive="both")
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=["zip"])

    for column in ("fhfa_annual_change_pct", "fhfa_hpi", "fhfa_hpi_1990_base", "fhfa_hpi_2000_base"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["period_start"] = pd.to_datetime(frame["period_year"].astype(str) + "-01-01", errors="coerce")
    frame["period_end"] = pd.to_datetime(frame["period_year"].astype(str) + "-12-31", errors="coerce")
    frame["period_month"] = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Int64")
    frame["frequency"] = "annual"
    frame["source"] = "fhfa_zip5"
    frame["source_url"] = _SOURCE_URLS["fhfa_zip5"]
    frame["metric_label"] = _METRIC_LABELS["fhfa_zip5"]
    frame["price_signal_value"] = frame["fhfa_hpi_2000_base"].where(frame["fhfa_hpi_2000_base"].notna(), frame["fhfa_hpi"])
    frame["price_signal_unit"] = frame["fhfa_hpi_2000_base"].notna().map(
        lambda has_base: "index_2000_base" if has_base else "index_native"
    )

    columns = [
        "zip",
        "period_start",
        "period_end",
        "period_year",
        "period_month",
        "frequency",
        "source",
        "source_url",
        "metric_label",
        "price_signal_value",
        "price_signal_unit",
        "fhfa_annual_change_pct",
        "fhfa_hpi",
        "fhfa_hpi_1990_base",
        "fhfa_hpi_2000_base",
    ]
    return frame[columns].sort_values(["zip", "period_year"], ignore_index=True)


def fetch_housing_history_dataset(
    settings: "Settings",
    *,
    zip_codes: Sequence[str] | None = None,
    year_floor: int = 2000,
    year_ceiling: int = 2025,
) -> str:
    """Build a historical housing panel that reaches back to the requested year floor."""

    requested_zips = list(zip_codes) if zip_codes is not None else _load_requested_zip_codes(settings)
    print("[housing] building historical Realtor ZIP panel.", flush=True)
    realtor_panel = fetch_realtor_zip_history_panel(
        requested_zips,
        year_floor=year_floor,
        year_ceiling=year_ceiling,
        timeout_seconds=settings.acquire_timeout_seconds,
        max_attempts=settings.acquire_max_attempts,
        backoff_seconds=settings.acquire_backoff_seconds,
    )
    print("[housing] building FHFA ZIP5 annual panel.", flush=True)
    fhfa_panel = fetch_fhfa_zip5_history(
        requested_zips,
        year_floor=year_floor,
        year_ceiling=year_ceiling,
        timeout_seconds=settings.acquire_timeout_seconds,
        max_attempts=settings.acquire_max_attempts,
        backoff_seconds=settings.acquire_backoff_seconds,
    )

    panels = [frame for frame in (realtor_panel, fhfa_panel) if not frame.empty]
    history = pd.concat(panels, ignore_index=True, sort=False) if panels else pd.DataFrame(columns=["zip"])
    if history.empty:
        raise AcquisitionError(
            f"No historical housing records were available between {year_floor} and {year_ceiling} for the requested ZIPs."
        )

    history = history.sort_values(["zip", "period_start", "source"], ignore_index=True)
    output_path = settings.raw_dir / "housing_market_history.csv"
    history.to_csv(output_path, index=False)

    coverage_by_source = {}
    for source_name, source_frame in history.groupby("source", dropna=False):
        coverage_by_source[str(source_name)] = {
            "rows": int(len(source_frame)),
            "zip_count": int(source_frame["zip"].nunique()),
            "min_year": int(source_frame["period_year"].dropna().min()) if source_frame["period_year"].notna().any() else None,
            "max_year": int(source_frame["period_year"].dropna().max()) if source_frame["period_year"].notna().any() else None,
        }

    metadata_path = settings.raw_dir / "housing_market_history.metadata.json"
    write_json_artifact(
        metadata_path,
        {
            "dataset": "housing_market_history",
            "retrieved_at_utc": utc_timestamp(),
            "target_year_floor": year_floor,
            "target_year_ceiling": year_ceiling,
            "requested_zip_count": len(requested_zips),
            "row_count": int(len(history)),
            "zip_count": int(history["zip"].nunique()),
            "min_year": int(history["period_year"].dropna().min()) if history["period_year"].notna().any() else None,
            "max_year": int(history["period_year"].dropna().max()) if history["period_year"].notna().any() else None,
            "source_summary": coverage_by_source,
            "source_url_patterns": {
                "realtor_history": _SOURCE_URLS["realtor_history"],
                "fhfa_zip5": _SOURCE_URLS["fhfa_zip5"],
            },
            "output_path": str(output_path),
        },
    )
    return str(output_path)


def _download_remote_tabular_source(
    url: str,
    *,
    suffix: str,
    timeout_seconds: int,
    max_attempts: int,
    backoff_seconds: float,
) -> Path:
    def _download() -> Path:
        request = Request(url, headers={"User-Agent": "dallas-crime-analysis/1.0"})
        with urlopen(request, timeout=timeout_seconds) as response:
            with tempfile.NamedTemporaryFile(prefix="dca-", suffix=suffix, delete=False) as handle:
                shutil.copyfileobj(response, handle)
                return Path(handle.name)

    return run_with_retry(
        f"Bulk source download for {url}",
        _download,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        retryable_exceptions=(HTTPError, URLError, TimeoutError, OSError),
        hint=(
            "Verify network access to the source URL, and consider raising "
            "DCA_ACQUIRE_MAX_ATTEMPTS or DCA_ACQUIRE_TIMEOUT_SECONDS."
        ),
    )


def _read_realtor_history_chunks(
    zip_codes: Sequence[str],
    *,
    chunksize: int,
    timeout_seconds: int,
    max_attempts: int,
    backoff_seconds: float,
) -> list[pd.DataFrame]:
    zip_set = set(zip_codes)
    source_path = _download_remote_tabular_source(
        _SOURCE_URLS["realtor_history"],
        suffix=".csv",
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )
    filtered_chunks: list[pd.DataFrame] = []
    try:
        for chunk in pd.read_csv(
            source_path,
            usecols=list(_REALTOR_HISTORY_COLUMNS.keys()),
            dtype={"postal_code": str, "month_date_yyyymm": str},
            chunksize=chunksize,
        ):
            chunk = chunk.rename(columns=_REALTOR_HISTORY_COLUMNS)
            chunk["zip"] = chunk["zip"].astype(str).str.zfill(5)
            subset = chunk[chunk["zip"].isin(zip_set)].copy()
            if not subset.empty:
                filtered_chunks.append(subset)
    finally:
        source_path.unlink(missing_ok=True)
    return filtered_chunks


def _load_local_realtor_history_summary(history_path: Path, zip_codes: Sequence[str]) -> pd.DataFrame:
    if not history_path.exists() or not zip_codes:
        return pd.DataFrame(columns=["zip"])

    usecols = [
        "zip",
        "source",
        "period_start",
        "realtor_hist_listing_price",
        "realtor_hist_active_listing_count",
        "realtor_hist_median_days_on_market",
        "realtor_hist_pending_ratio",
        "realtor_hist_quality_flag",
    ]
    frame = pd.read_csv(history_path, usecols=usecols, dtype={"zip": str, "source": str})
    frame = frame[(frame["source"] == "realtor_history") & (frame["zip"].isin(set(zip_codes)))].copy()
    if frame.empty:
        return pd.DataFrame(columns=["zip"])

    frame["realtor_hist_as_of_date"] = pd.to_datetime(frame["period_start"], errors="coerce")
    frame = frame.dropna(subset=["realtor_hist_as_of_date"]).copy()
    for column in (
        "realtor_hist_listing_price",
        "realtor_hist_active_listing_count",
        "realtor_hist_median_days_on_market",
        "realtor_hist_pending_ratio",
        "realtor_hist_quality_flag",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["zip", "realtor_hist_as_of_date"], kind="mergesort")
    return _summarize_realtor_history(frame)


def _collect_zillow_records(
    settings: "Settings",
    zip_codes: Sequence[str],
    status_by_zip: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    batches = list(_chunked(zip_codes, settings.housing_scrape_batch_size))
    total_batches = len(batches)

    for batch_index, batch in enumerate(batches, start=1):
        query = _build_batch_housing_query(settings, batch)
        search_limit = max(len(batch) * 4, 20)
        print(
            f"[housing] batch {batch_index}/{total_batches}: searching {len(batch)} ZIP(s).",
            flush=True,
        )
        search_command = build_firecrawl_search_command(
            FirecrawlSearchRequest(
                query=query,
                limit=search_limit,
                scrape=True,
                scrape_formats=("markdown",),
            )
        )
        try:
            payload = run_firecrawl_command(
                search_command,
                timeout_seconds=settings.firecrawl_timeout_seconds,
                max_attempts=settings.acquire_max_attempts,
                backoff_seconds=settings.acquire_backoff_seconds,
            )
            frame = normalize_firecrawl_documents(payload)
            frame = _filter_to_batch_zips(frame, batch)
            if frame.empty:
                print(
                    f"[housing] batch {batch_index}/{total_batches}: no metrics returned.",
                    flush=True,
                )
                continue
            extracted_records = frame.to_dict(orient="records")
            records.extend(extracted_records)
            print(
                f"[housing] batch {batch_index}/{total_batches}: extracted {len(extracted_records)} record(s).",
                flush=True,
            )
        except AcquisitionError as exc:
            for zip_code in batch:
                status_by_zip[zip_code] = {
                    "zip": zip_code,
                    "status": "error",
                    "records": 0,
                    "error": str(exc),
                }
            print(f"[housing] batch {batch_index}/{total_batches}: error {exc}", flush=True)

    return records


def _collect_direct_source_records(
    settings: "Settings",
    zip_codes: Sequence[str],
    status_by_zip: dict[str, dict[str, Any]],
    *,
    source_name: str,
    url_builder,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = len(zip_codes)
    for index, zip_code in enumerate(zip_codes, start=1):
        url = url_builder(zip_code)
        print(f"[housing] {source_name} {index}/{total}: scraping {zip_code}.", flush=True)
        scrape_command = build_firecrawl_scrape_command(
            FirecrawlScrapeRequest(
                urls=(url,),
                formats=("markdown",),
                wait_for_ms=3000,
            )
        )
        try:
            payload = run_firecrawl_command(
                scrape_command,
                timeout_seconds=settings.firecrawl_timeout_seconds,
                max_attempts=settings.acquire_max_attempts,
                backoff_seconds=settings.acquire_backoff_seconds,
            )
            frame = normalize_firecrawl_documents(payload)
            frame = _filter_to_batch_zips(frame, [zip_code])
            if frame.empty:
                print(
                    f"[housing] {source_name} {index}/{total}: no metrics returned for {zip_code}.",
                    flush=True,
                )
                continue
            extracted_records = frame.to_dict(orient="records")
            records.extend(extracted_records)
            print(
                f"[housing] {source_name} {index}/{total}: extracted {len(extracted_records)} record(s).",
                flush=True,
            )
        except AcquisitionError as exc:
            status_by_zip[zip_code] = {
                "zip": zip_code,
                "status": "error",
                "records": 0,
                "error": str(exc),
            }
            print(f"[housing] {source_name} {index}/{total}: error {exc}", flush=True)
    return records


def _filter_to_batch_zips(frame: pd.DataFrame, batch: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    batch_set = set(batch)
    frame = frame.copy()
    source_urls = (
        frame["source_url"]
        if "source_url" in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    source_name = source_urls.map(_infer_source_name)
    source_zip = source_urls.map(_extract_zip_from_source_url)
    extracted_zip = frame["zip"].astype("string").str.extract(r"(\d{5})", expand=False)

    frame["zip"] = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    zillow_mask = source_name == "zillow"
    non_zillow_mask = ~zillow_mask

    frame.loc[zillow_mask, "zip"] = source_zip[zillow_mask].where(source_zip[zillow_mask].isin(batch_set))
    frame.loc[non_zillow_mask, "zip"] = source_zip[non_zillow_mask].where(
        source_zip[non_zillow_mask].isin(batch_set),
        extracted_zip[non_zillow_mask],
    )
    frame["zip"] = frame["zip"].astype("string").str.extract(r"(\d{5})", expand=False)
    return frame[frame["zip"].isin(batch_set)].copy()


def _merge_housing_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "zip",
                "source_url",
                "home_value",
                "median_rent",
                "annual_change_pct",
                "as_of_date",
                "source",
                "metric_label",
                "supplemental_sources",
            ]
        )

    frame = pd.DataFrame.from_records(records).copy()
    frame["zip"] = frame["zip"].astype("string").str.extract(r"(\d{5})", expand=False)
    frame = frame.dropna(subset=["zip"]).copy()
    source_urls = (
        frame["source_url"]
        if "source_url" in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    source_url_zip = source_urls.map(_extract_zip_from_source_url)
    valid_url_zip = source_url_zip.isna() | (source_url_zip == frame["zip"])
    frame = frame.loc[valid_url_zip].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "zip",
                "source_url",
                "home_value",
                "median_rent",
                "annual_change_pct",
                "as_of_date",
                "source",
                "metric_label",
                "supplemental_sources",
            ]
        )
    frame["_priority"] = frame["source"].map(_SOURCE_PRIORITY).fillna(_SOURCE_PRIORITY["unknown"])
    frame["_as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")

    merged_rows: list[dict[str, Any]] = []
    for zip_code, group in frame.groupby("zip", sort=True):
        ordered = group.sort_values(["_priority", "_as_of_date"], ascending=[True, False], na_position="last")
        primary_candidates = ordered[ordered["home_value"].notna()]
        primary = primary_candidates.iloc[0] if not primary_candidates.empty else ordered.iloc[0]

        merged = primary.drop(labels=["_priority", "_as_of_date"]).to_dict()
        supplemental_sources: list[str] = []
        for _, row in ordered.iterrows():
            source_name = str(row.get("source") or "unknown")
            if source_name != merged.get("source"):
                supplemental_sources.append(source_name)
            for column in ("median_rent", "annual_change_pct", "as_of_date"):
                if pd.isna(merged.get(column)) and pd.notna(row.get(column)):
                    merged[column] = row.get(column)

        merged["supplemental_sources"] = ";".join(dict.fromkeys(supplemental_sources)) or None
        merged_rows.append(merged)

    merged_frame = pd.DataFrame.from_records(merged_rows)
    if merged_frame.empty:
        return merged_frame
    merged_frame = merged_frame.dropna(subset=["home_value"]).sort_values("zip").reset_index(drop=True)
    return merged_frame


def _record_counts_by_zip(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        zip_code = str(record.get("zip") or "")
        if re.fullmatch(r"\d{5}", zip_code):
            counts[zip_code] = counts.get(zip_code, 0) + 1
    return counts


def _remaining_zip_codes(zip_codes: Sequence[str], housing: pd.DataFrame) -> list[str]:
    if housing.empty or "zip" not in housing.columns:
        return list(zip_codes)
    received = {
        str(value).zfill(5)
        for value in housing["zip"].dropna().astype(str)
        if re.fullmatch(r"\d{5}", str(value).zfill(5))
    }
    return [zip_code for zip_code in zip_codes if zip_code not in received]


def _realtor_inventory_records(
    inventory: pd.DataFrame,
    remaining_zips: Sequence[str],
) -> list[dict[str, Any]]:
    """Create housing records from structured Realtor ZIP inventory rows."""

    if inventory.empty or not remaining_zips:
        return []
    remaining = set(remaining_zips)
    frame = inventory[inventory["zip"].isin(remaining)].copy()
    if frame.empty:
        return []

    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        listing_price = row.get("realtor_listing_price")
        if pd.isna(listing_price):
            continue
        as_of_date = row.get("realtor_as_of_date")
        as_of_iso = as_of_date.date().isoformat() if pd.notna(as_of_date) else None
        annual_change = row.get("realtor_listing_price_yy")
        annual_change_pct = float(annual_change) * 100 if pd.notna(annual_change) else None
        records.append(
            {
                "zip": row["zip"],
                "source_url": _SOURCE_URLS["realtor_csv"],
                "home_value": float(listing_price),
                "median_rent": None,
                "annual_change_pct": annual_change_pct,
                "as_of_date": as_of_iso,
                "source": "realtor_csv",
                "metric_label": _METRIC_LABELS["realtor_csv"],
                "supplemental_sources": None,
            }
        )
    return records


def _merge_realtor_inventory_features(
    housing: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Merge structured Realtor ZIP inventory fields onto the selected housing rows."""

    if housing.empty or inventory.empty:
        return housing
    feature_columns = [
        "zip",
        "realtor_listing_price",
        "realtor_listing_price_yy",
        "realtor_active_listing_count",
        "realtor_median_days_on_market",
        "realtor_listing_price_per_square_foot",
        "realtor_pending_ratio",
        "realtor_quality_flag",
    ]
    return housing.merge(inventory[feature_columns], on="zip", how="left")


def _merge_realtor_history_features(
    housing: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    if housing.empty or history.empty:
        return housing
    return housing.merge(history, on="zip", how="left")


def _build_realtor_market_url(zip_code: str) -> str:
    return _SOURCE_URLS["realtor"].format(zip=zip_code)


def _build_redfin_market_url(zip_code: str) -> str:
    return _SOURCE_URLS["redfin"].format(zip=zip_code)


def _summarize_realtor_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize recent Realtor ZIP history into stable 12-month features."""

    if frame.empty:
        return pd.DataFrame(columns=["zip"])

    latest_month = frame["realtor_hist_as_of_date"].max()
    if pd.isna(latest_month):
        return pd.DataFrame(columns=["zip"])
    cutoff = latest_month - pd.DateOffset(months=11)
    recent = frame[frame["realtor_hist_as_of_date"] >= cutoff].copy()
    if recent.empty:
        return pd.DataFrame(columns=["zip"])

    rows: list[dict[str, Any]] = []
    for zip_code, group in recent.groupby("zip", sort=True):
        group = group.sort_values("realtor_hist_as_of_date")
        first_price = group["realtor_hist_listing_price"].dropna().iloc[0] if group["realtor_hist_listing_price"].notna().any() else None
        last_price = group["realtor_hist_listing_price"].dropna().iloc[-1] if group["realtor_hist_listing_price"].notna().any() else None
        price_change = None
        if first_price not in (None, 0) and last_price is not None:
            price_change = (float(last_price) / float(first_price)) - 1

        rows.append(
            {
                "zip": zip_code,
                "realtor_hist_months_observed": int(group["realtor_hist_as_of_date"].nunique()),
                "realtor_hist_listing_price_12m_avg": group["realtor_hist_listing_price"].mean(),
                "realtor_hist_listing_price_12m_change": price_change,
                "realtor_hist_active_listing_count_12m_avg": group["realtor_hist_active_listing_count"].mean(),
                "realtor_hist_median_days_on_market_12m_avg": group["realtor_hist_median_days_on_market"].mean(),
                "realtor_hist_pending_ratio_12m_avg": group["realtor_hist_pending_ratio"].mean(),
                "realtor_hist_quality_flag_12m_max": group["realtor_hist_quality_flag"].max(),
            }
        )
    return pd.DataFrame.from_records(rows)


def _extract_markdown(document: Mapping[str, Any]) -> str:
    if "markdown" in document:
        return str(document["markdown"])
    data = document.get("data")
    if isinstance(data, Mapping) and "markdown" in data:
        return str(data["markdown"])
    return ""


def _extract_url(document: Mapping[str, Any]) -> str | None:
    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("sourceURL", "url"):
            if key in metadata and metadata[key]:
                return str(metadata[key])
    for key in ("url", "sourceURL"):
        if key in document and document[key]:
            return str(document[key])
    return None


def _extract_description(document: Mapping[str, Any]) -> str | None:
    if "description" in document and document["description"]:
        return str(document["description"])
    return None


def _infer_source_name(url: str | None) -> str:
    if not url:
        return "unknown"
    netloc = urlparse(url).netloc.lower()
    if "zillow.com" in netloc:
        return "zillow"
    if "realtor.com" in netloc:
        return "realtor"
    if "redfin.com" in netloc:
        return "redfin"
    return "unknown"


def _extract_zip_from_source_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    source_name = _infer_source_name(url)
    digits = re.findall(r"\d{5}", parsed.path or "")
    if not digits:
        return None
    if source_name == "zillow":
        return digits[-1]
    return digits[0]


def _extract_zillow_metrics(text: str) -> dict[str, Any]:
    home_value = None
    median_rent = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for metric, patterns in _METRIC_PATTERNS.items():
            if metric == "home_value" and home_value is not None:
                continue
            if metric == "median_rent" and median_rent is not None:
                continue
            if any(pattern.search(stripped) for pattern in patterns):
                value = _extract_price(stripped)
                if metric == "home_value":
                    home_value = value
                else:
                    median_rent = value
        if home_value is None and "home value is" in stripped.lower():
            home_value = _extract_price(stripped)
    return {
        "home_value": home_value,
        "median_rent": median_rent,
        "annual_change_pct": _extract_percent(text),
        "as_of_date": _extract_date(text),
    }


def _extract_redfin_metrics(text: str) -> dict[str, Any]:
    sale_match = _REDFIN_SALE_SENTENCE_PATTERN.search(text) or _REDFIN_PRICE_SECTION_PATTERN.search(text)
    home_value = _extract_price(sale_match.group(1)) if sale_match else None
    change_match = _UP_DOWN_CHANGE_PATTERN.search(text) or _YEAR_OVER_YEAR_PATTERN.search(text)
    annual_change_pct = _coerce_percent_change(change_match)
    return {
        "home_value": home_value,
        "median_rent": None,
        "annual_change_pct": annual_change_pct,
        "as_of_date": _extract_date(text),
    }


def _extract_realtor_metrics(text: str) -> dict[str, Any]:
    home_row = _REALTOR_HOME_ROW_PATTERN.search(text)
    rent_row = _REALTOR_RENT_ROW_PATTERN.search(text)
    home_value = _extract_price(home_row.group(1)) if home_row else None
    median_rent = _extract_price(rent_row.group(1)) if rent_row else None
    annual_change_pct = None
    if home_row:
        annual_change_pct = _coerce_percent(home_row.group(2))
    return {
        "home_value": home_value,
        "median_rent": median_rent,
        "annual_change_pct": annual_change_pct,
        "as_of_date": _extract_date(text),
    }


def _extract_zip(text: str) -> str | None:
    match = _ZIP_PATTERN.search(text or "")
    return match.group(1) if match else None


def _extract_price(text: str) -> float | None:
    matches = list(_PRICE_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    value = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").upper()
    multiplier = {"": 1.0, "K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[suffix]
    return value * multiplier


def _extract_date(text: str) -> str | None:
    match = _DATE_PATTERN.search(text)
    if not match:
        return None
    parsed = pd.to_datetime(match.group(1), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _extract_percent(text: str) -> float | None:
    match = _CHANGE_PATTERN.search(text)
    return float(match.group(1)) if match else None


def _coerce_percent(text: str | None) -> float | None:
    if not text:
        return None
    cleaned = str(text).replace("%", "").replace("+", "").strip()
    if cleaned.lower().startswith("down "):
        cleaned = f"-{cleaned[5:]}"
    elif cleaned.lower().startswith("up "):
        cleaned = cleaned[3:]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _coerce_percent_change(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    if len(match.groups()) == 2:
        direction = match.group(1).lower()
        value = float(match.group(2))
        return value if direction == "up" else -value
    return float(match.group(1))
