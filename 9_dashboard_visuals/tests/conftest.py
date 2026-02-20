"""Pytest config for dashboard visuals tests."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: run full pipeline (BupaR/DTW/FP-Growth); set RUN_VISUALS_INTEGRATION=1 to run",
    )
