"""Configuration helpers for the Dallas crime analysis project."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path


DEFAULT_CRIME_SOURCE_URL = "https://www.dallasopendata.com/resource/qv6i-rri7.json"
DEFAULT_HOUSING_QUERY_TEMPLATE = "site:zillow.com/home-values/ Dallas TX {zip} typical home value"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    project_root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    firecrawl_cache_dir: Path
    firecrawl_api_key: str | None
    census_api_key: str | None
    census_year: int
    census_state_fips: str
    crime_source_url: str
    crime_limit: int
    crime_lookback_days: int
    crime_where_clause: str | None
    housing_query_template: str
    max_housing_zips: int | None

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = project_root.resolve() if project_root else Path.cwd().resolve()
        data_dir = _env_path("DCA_DATA_DIR", root / "data")
        reports_dir = _env_path("DCA_REPORTS_DIR", root / "reports")
        return cls(
            project_root=root,
            data_dir=data_dir,
            raw_dir=_env_path("DCA_RAW_DIR", data_dir / "raw"),
            interim_dir=_env_path("DCA_INTERIM_DIR", data_dir / "interim"),
            processed_dir=_env_path("DCA_PROCESSED_DIR", data_dir / "processed"),
            reports_dir=reports_dir,
            firecrawl_cache_dir=_env_path("DCA_FIRECRAWL_CACHE_DIR", root / ".firecrawl"),
            firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
            census_api_key=os.getenv("CENSUS_API_KEY"),
            census_year=int(os.getenv("DCA_CENSUS_YEAR", "2024")),
            census_state_fips=os.getenv("DCA_CENSUS_STATE_FIPS", "48"),
            crime_source_url=os.getenv("DCA_CRIME_SOURCE_URL", DEFAULT_CRIME_SOURCE_URL),
            crime_limit=int(os.getenv("DCA_CRIME_LIMIT", "50000")),
            crime_lookback_days=int(os.getenv("DCA_CRIME_LOOKBACK_DAYS", "365")),
            crime_where_clause=os.getenv("DCA_CRIME_WHERE"),
            housing_query_template=os.getenv(
                "DCA_HOUSING_SOURCE_QUERY",
                DEFAULT_HOUSING_QUERY_TEMPLATE,
            ),
            max_housing_zips=int(os.getenv("DCA_MAX_HOUSING_ZIPS")) if os.getenv("DCA_MAX_HOUSING_ZIPS") else None,
        )

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.reports_dir,
            self.firecrawl_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolved_crime_where_clause(self) -> str:
        if self.crime_where_clause:
            return self.crime_where_clause
        cutoff = datetime.now(UTC).date() - timedelta(days=self.crime_lookback_days)
        return f"date1 >= '{cutoff.isoformat()}T00:00:00'"

    def housing_query_for_zip(self, zip_code: str) -> str:
        if "{zip}" in self.housing_query_template:
            return self.housing_query_template.format(zip=zip_code)
        return f"{self.housing_query_template} {zip_code}".strip()

    def describe(self) -> dict[str, str | None]:
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "raw_dir": str(self.raw_dir),
            "interim_dir": str(self.interim_dir),
            "processed_dir": str(self.processed_dir),
            "reports_dir": str(self.reports_dir),
            "firecrawl_cache_dir": str(self.firecrawl_cache_dir),
            "census_year": str(self.census_year),
            "census_state_fips": self.census_state_fips,
            "crime_source_url": self.crime_source_url,
            "crime_limit": str(self.crime_limit),
            "crime_lookback_days": str(self.crime_lookback_days),
            "crime_where_clause": self.resolved_crime_where_clause(),
            "housing_query_template": self.housing_query_template,
            "max_housing_zips": str(self.max_housing_zips) if self.max_housing_zips else None,
            "firecrawl_api_key_present": "yes" if self.firecrawl_api_key else "no",
            "census_api_key_present": "yes" if self.census_api_key else "no",
        }
