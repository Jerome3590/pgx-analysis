"""
Shared fixtures and helpers for dashboard tests (by tab/page).
Paths, Lambda handler, and API event builder.
"""
import json
import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.constants import REQUIRED_COHORTS  # noqa: E402

DASHBOARD_ROOT = REPO_ROOT / "10_risk_dashboard"
OUTPUTS = DASHBOARD_ROOT / "outputs"
METADATA_DIR = OUTPUTS / "metadata"
MODELS_DIR = OUTPUTS / "models"
CPIC_DIR = OUTPUTS / "cpic"
FRONTEND_DIR = DASHBOARD_ROOT / "frontend"

try:
    sys.path.insert(0, str(DASHBOARD_ROOT / "backend"))
    from lambda_function import lambda_handler  # noqa: E402
    LAMBDA_AVAILABLE = True
except Exception:
    LAMBDA_AVAILABLE = False
    lambda_handler = None

BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")

def get_live_session():
    if not BASE_URL:
        return None
    try:
        import requests
        return requests.Session()
    except ImportError:
        return None

def age_band_fname(age_band: str) -> str:
    return age_band.replace("-", "_")

def query_event(method: str, path: str, query: dict = None, body: dict = None) -> dict:
    p = path if path.startswith("/") else "/" + path
    return {
        "httpMethod": method,
        "path": f"/prod{p}",
        "queryStringParameters": query or None,
        "body": json.dumps(body) if body is not None else None,
    }

@pytest.fixture(scope="session")
def lambda_handler_if_available():
    return lambda_handler

@pytest.fixture(scope="session")
def live_session():
    return get_live_session()
