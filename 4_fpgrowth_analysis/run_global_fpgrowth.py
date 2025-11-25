"""
Standalone Python script version of global FPGrowth notebook
Run this to execute the analysis without Jupyter
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import time

# MLxtend for FP-Growth
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Project utilities
from helpers_1997_13.common_imports import s3_client, S3_BUCKET
from helpers_1997_13.duckdb_utils import get_duckdb_connection
from helpers_1997_13.s3_utils import save_to_s3_json

# Configuration
MIN_SUPPORT = 0.005
MIN_CONFIDENCE = 0.01
ITEM_TYPES = ['drug_name', 'icd_code', 'cpt_code']
S3_OUTPUT_BASE = f"s3://{S3_BUCKET}/gold/fpgrowth/global"
LOCAL_DATA_PATH = project_root / "data" / "gold" / "cohorts_F1120"

# Create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('global_fpgrowth')

print(f"âœ“ Configuration loaded")
print(f"  Min Support: {MIN_SUPPORT}")
print(f"  Item Types: {ITEM_TYPES}")
print(f"  Local Data: {LOCAL_DATA_PATH}")
print(f"  Local Data Exists: {LOCAL_DATA_PATH.exists()}")

if __name__ == "__main__":
    print("\níº€ Starting Global FPGrowth Analysis...")
    print(f"This will process {len(ITEM_TYPES)} item types: {', '.join(ITEM_TYPES)}")
    print(f"Using local data from: {LOCAL_DATA_PATH}")
    print("\n" + "="*80 + "\n")
    
    # Import the processing functions from the notebook would go here
    # For now, just show what would be executed
    print("To run the full analysis:")
    print("1. Open Jupyter: jupyter notebook")
    print("2. Navigate to: 3_fpgrowth_analysis/")
    print("3. Open: global_fpgrowth_feature_importance.ipynb")
    print("4. Run All Cells (Kernel > Restart & Run All)")
    
    print("\nâœ“ Script ready. Please run in Jupyter for interactive execution.")
