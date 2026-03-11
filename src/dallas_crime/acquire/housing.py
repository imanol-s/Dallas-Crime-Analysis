"""Firecrawl-backed housing acquisition helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import subprocess
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import pandas as pd

if TYPE_CHECKING:
    from dallas_crime.config import Settings

_ZIP_PATTERN = re.compile(r"\b(\d{5})\b")
_PRICE_PATTERN = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([KMB])?", re.IGNORECASE)
_DATE_PATTERN = re.compile(r"(?:Updated on |\()(\d{1,2}/\d{1,2}/\d{4}|[A-Z][a-z]+ \d{1,2}, \d{4})")
_CHANGE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)%\s+(?:1-yr|over the past year)", re.IGNORECASE)

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


def run_firecrawl_command(command: Sequence[str]) -> Any:
    """Run a Firecrawl CLI command and decode JSON when possible."""

    completed = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return stdout


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
    record: dict[str, Any] = {
        "zip": resolved_zip,
        "source_url": url,
        "home_value": None,
        "median_rent": None,
        "annual_change_pct": _extract_percent(text),
        "as_of_date": _extract_date(text),
        "source": "firecrawl",
    }

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for metric, patterns in _METRIC_PATTERNS.items():
            if record[metric] is not None:
                continue
            if any(pattern.search(stripped) for pattern in patterns):
                record[metric] = _extract_price(stripped)
        if record["home_value"] is None and "home value is" in stripped.lower():
            record["home_value"] = _extract_price(stripped)
    return record


def fetch_housing_dataset(settings: "Settings") -> str:
    """Discover ZIP-specific Zillow pages with Firecrawl and persist metrics."""

    crime_path = settings.raw_dir / "crime_records.csv"
    if not crime_path.exists():
        raise FileNotFoundError(
            f"Crime raw dataset not found at {crime_path}. Run acquisition in order."
        )

    crime_frame = pd.read_csv(crime_path, usecols=["zip"])
    zip_codes = sorted(
        {
            str(value).zfill(5)
            for value in crime_frame["zip"].dropna().astype(str)
            if re.fullmatch(r"\d{5}", str(value).zfill(5))
        }
    )
    if not zip_codes:
        raise ValueError("No ZIP codes were available from crime_records.csv for housing lookup.")
    if settings.max_housing_zips:
        zip_codes = zip_codes[: settings.max_housing_zips]

    records: list[dict[str, Any]] = []
    for zip_code in zip_codes:
        command = build_firecrawl_search_command(
            FirecrawlSearchRequest(
                query=settings.housing_query_for_zip(zip_code),
                limit=1,
                scrape=True,
                scrape_formats=("markdown",),
            )
        )
        payload = run_firecrawl_command(command)
        frame = normalize_firecrawl_documents(payload)
        if frame.empty:
            continue
        frame["zip"] = frame["zip"].fillna(zip_code)
        records.extend(frame.to_dict(orient="records"))

    housing = pd.DataFrame.from_records(records)
    if housing.empty:
        raise ValueError("Firecrawl returned no housing metrics for the configured ZIP search set.")

    housing = (
        housing.dropna(subset=["zip", "home_value"])
        .sort_values("zip")
        .drop_duplicates(subset=["zip"], keep="last")
    )
    output_path = settings.raw_dir / "housing_market.csv"
    housing.to_csv(output_path, index=False)
    return str(output_path)


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
