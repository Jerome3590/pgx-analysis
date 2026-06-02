"""Canonical API and S3 paths for scenario/interaction analysis."""

from __future__ import annotations

from typing import Optional

# ── HTTP API (canonical) ─────────────────────────────────────────────────────
API_VISUALIZATIONS_SCENARIO = "/visualizations/scenario"
API_SCENARIO_IMPORTANCE = "/scenario/importance"
API_SCENARIO_INTERACTIONS = "/scenario/interactions"

# ── S3 dashboard bucket layout ───────────────────────────────────────────────
S3_VIZ_SCENARIO_PREFIX = "visualizations/scenario"
SCENARIO_DATA_JSON = "scenario_data.json"


def _bin_segment(bin_name: Optional[str]) -> str:
    return f"/{bin_name}" if bin_name else ""


def s3_scenario_data_key(
    dashboard_prefix: str,
    cohort: str,
    age_band: str,
    bin_name: Optional[str] = None,
) -> str:
    """S3 object key for prebuilt scenario JSON."""
    prefix = dashboard_prefix.strip("/")
    return f"{prefix}/{S3_VIZ_SCENARIO_PREFIX}/{cohort}/{age_band}{_bin_segment(bin_name)}/{SCENARIO_DATA_JSON}"
