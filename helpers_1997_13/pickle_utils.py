"""Compatibility shim: expose pickle helpers but delegate to
helpers_1997_13.data_utils so callers that import this module keep working.
"""
from typing import Optional
from helpers_1997_13.data_utils import find_preferred_pickle, safe_load_pickle


def find_target_pickle(base_dir: str, name: str = 'target_code_analysis_data.pkl') -> Optional[str]:
    """Backward-compatible name for find_preferred_pickle."""
    return find_preferred_pickle(base_dir, name)


def load_pickle(path: Optional[str]):
    """Backward-compatible loader delegating to safe_load_pickle."""
    return safe_load_pickle(path)
