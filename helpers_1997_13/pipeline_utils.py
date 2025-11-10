#!/usr/bin/env python3
"""
Pipeline utilities for worker processing and multiprocessing coordination.
"""

import os
import sys
import json
import tempfile
import multiprocessing as mp
from typing import Dict, Optional, Tuple

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


def get_multiprocessing_context():
    """
    Get the appropriate multiprocessing context.
    - Default: 'spawn' (more stable with many workers, lower memory usage)
    - On Linux: can use 'fork' for faster startup (but higher memory usage)
    - Can be overridden via PGX_MP_START_METHOD env var ('fork' or 'spawn')
    
    Returns:
        Tuple of (context, method_name)
    """
    mp_start_method = os.getenv('PGX_MP_START_METHOD', '').lower()
    if mp_start_method in ('fork', 'spawn'):
        try:
            return mp.get_context(mp_start_method), mp_start_method
        except (ValueError, RuntimeError) as e:
            print(f"⚠️ Warning: Requested start method '{mp_start_method}' not available: {e}, falling back to 'spawn'")
            return mp.get_context('spawn'), 'spawn'
    
    # Default to spawn for better stability and lower memory usage
    # Fork can be faster but causes high memory usage with many workers
    return mp.get_context('spawn'), 'spawn'


def persist_mappings_to_temp(icd_map: Dict[str, str], cpt_map: Dict[str, str], 
                               icd_target_map: Dict[str, str], drug_map: Dict[str, str] = None) -> Optional[str]:
    """
    Persist mappings to temporary JSON files to reduce memory duplication in spawn mode.
    Returns temp directory path if successful, None otherwise.
    """
    try:
        mapping_temp_dir = tempfile.mkdtemp(prefix='pgx_mappings_')
        
        if icd_map:
            with open(os.path.join(mapping_temp_dir, 'icd_map.json'), 'w') as f:
                json.dump(icd_map, f)
        if cpt_map:
            with open(os.path.join(mapping_temp_dir, 'cpt_map.json'), 'w') as f:
                json.dump(cpt_map, f)
        if icd_target_map:
            with open(os.path.join(mapping_temp_dir, 'icd_target_map.json'), 'w') as f:
                json.dump(icd_target_map, f)
        if drug_map:
            with open(os.path.join(mapping_temp_dir, 'drug_map.json'), 'w') as f:
                json.dump(drug_map, f)
        
        return mapping_temp_dir
    except Exception as e:
        print(f"⚠️ Warning: Could not persist mappings to temp files: {e}")
        return None


def load_mappings_from_temp(mapping_temp_dir: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Load mappings from temporary JSON files.
    Returns (icd_map, cpt_map, icd_target_map, drug_map) tuple.
    """
    if not mapping_temp_dir or not os.path.exists(mapping_temp_dir):
        return {}, {}, {}, {}
    
    try:
        icd_map = {}
        cpt_map = {}
        icd_target_map = {}
        drug_map = {}
        
        icd_path = os.path.join(mapping_temp_dir, 'icd_map.json')
        if os.path.exists(icd_path):
            with open(icd_path, 'r') as f:
                icd_map = json.load(f)
        
        cpt_path = os.path.join(mapping_temp_dir, 'cpt_map.json')
        if os.path.exists(cpt_path):
            with open(cpt_path, 'r') as f:
                cpt_map = json.load(f)
        
        target_path = os.path.join(mapping_temp_dir, 'icd_target_map.json')
        if os.path.exists(target_path):
            with open(target_path, 'r') as f:
                icd_target_map = json.load(f)
        
        drug_path = os.path.join(mapping_temp_dir, 'drug_map.json')
        if os.path.exists(drug_path):
            with open(drug_path, 'r') as f:
                drug_map = json.load(f)
        
        return icd_map, cpt_map, icd_target_map, drug_map
    except Exception as e:
        print(f"⚠️ Warning: Could not load mappings from temp files: {e}")
        return {}, {}, {}, {}


def get_retry_attempts() -> int:
    """Get retry attempts from environment, default to 3."""
    return int(os.getenv('PGX_RETRY_ATTEMPTS', '3'))


def get_timeout_seconds() -> int:
    """Get timeout seconds from environment, default to 3600 (1 hour)."""
    return int(os.getenv('PGX_TIMEOUT_SECONDS', '3600'))

