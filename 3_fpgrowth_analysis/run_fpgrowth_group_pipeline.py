import os
import sys
import logging
import time
import psutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project path config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.constants import AGE_BANDS, EVENT_YEARS, TIMEOUT_SECONDS
from helpers_1997_13.duckdb_utils import get_duckdb_connection
from helpers_1997_13.fpgrowth_utils import (
    fpgrowth_with_timeout,
    generate_rules_from_itemsets,
    convert_frozensets,
    save_drug_encoding_map
)
from helpers_1997_13.s3_utils import (
    get_global_base_path,
    save_to_s3_json
)
from helpers_1997_13.drug_utils import encode_drug_name


def create_fpgrowth_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_pharmacy_subset(group, logger):
    """Load pharmacy rows from SILVER zone for the given group.
    Early/middle may span multiple years; we pass explicit partitioned paths.
    """
    con = get_duckdb_connection(logger=logger)

    age_band = group["age_band"]
    period = group["period"]
    event_year = group["event_year"]

    # Build list of partitioned paths
    if period in ["early", "middle"] and isinstance(event_year, list):
        years = event_year
    else:
        years = [event_year]

    paths = [
        f"s3://pgxdatalake/silver/pharmacy/age_band={age_band}/event_year={y}/*.parquet"
        for y in years
    ]

    paths_sql = ",".join([f"'{p}'" for p in paths])

    query = f"""
        SELECT mi_person_key, drug_name
        FROM read_parquet([{paths_sql}])
    """

    return con.execute(query).df()


def load_medical_subset(group, logger):
    """Load medical rows from SILVER zone for the given group (for context/logging)."""
    con = get_duckdb_connection(logger=logger)

    age_band = group["age_band"]
    period = group["period"]
    event_year = group["event_year"]

    if period in ["early", "middle"] and isinstance(event_year, list):
        years = event_year
    else:
        years = [event_year]

    paths = [
        f"s3://pgxdatalake/silver/medical/age_band={age_band}/event_year={y}/*.parquet"
        for y in years
    ]
    paths_sql = ",".join([f"'{p}'" for p in paths])

    query = f"""
        SELECT mi_person_key
        FROM read_parquet([{paths_sql}])
    """

    return con.execute(query).df()


def run_group_fpgrowth(group):
    age_band = group['age_band']
    event_year = group['event_year']
    period = group['period']
    
    # Create logger name based on period and event_year
    if period in ["early", "middle"]:
        logger_name = f"group_fpgrowth_{age_band}_{period}_{'-'.join(event_year)}"
    else:
        logger_name = f"group_fpgrowth_{age_band}_{event_year}"
    
    logger = create_fpgrowth_logger(logger_name)

    try:
        logger.info(f"⏳ Starting FP-Growth for {group}")

        def log_usage(stage):
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            logger.info(f"{stage} | Memory: {memory.percent:.1f}% | CPU: {cpu:.1f}%")

        # Step 1: Load data
        t0 = time.time()
        subset = load_pharmacy_subset(group, logger)
        medical_subset = load_medical_subset(group, logger)
        log_usage("Step 1 (Load)")
        logger.info(f"Step 1 completed in {time.time() - t0:.2f}s")

        if subset.empty:
            if period in ["early", "middle"]:
                logger.warning(f"No data for {age_band}, {period} period ({'-'.join(event_year)})")
            else:
                logger.warning(f"No data for {age_band}, {event_year}")
            return (group, False)

        # Step 2: Prepare transactions
        t0 = time.time()
        transactions = subset.groupby("mi_person_key")["drug_name"].apply(list).tolist()
        log_usage("Step 2 (Group)")
        logger.info(f"Step 2 completed in {time.time() - t0:.2f}s")

        # Step 3: Run FP-Growth (30 min timeout)
        t0 = time.time()
        from mlxtend.preprocessing import TransactionEncoder

        te = TransactionEncoder()
        df_tf = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

        logger.info(f"[DIAG] Transactions passed to FP-Growth: {df_tf.shape[0]}")
        logger.info(f"[DIAG] Unique tokens (columns) passed to FP-Growth: {df_tf.shape[1]}")
        itemsets = fpgrowth_with_timeout(df_tf, min_support=0.01, timeout=TIMEOUT_SECONDS)
        if itemsets is None:
            itemsets = pd.DataFrame(columns=["support", "itemsets"])
        rules = generate_rules_from_itemsets(itemsets, metric="confidence", min_threshold=0.5)
        if not rules.empty:
            rules["certainty"] = rules["support"] * rules["confidence"]

        # Get unique drug names from the original pharmacy data
        unique_drugs = subset['drug_name'].unique().tolist()
        logger.info(f"Original pharmacy data contains {len(unique_drugs)} unique drug names")
        logger.info(f"Loaded medical rows: {len(medical_subset)} (for context)")
        logger.info(f"Sample drug names: {unique_drugs[:5]}")
        
        # Create comprehensive drug metrics using original drug names
        drug_metrics = []
        
        # Process itemsets for drug metrics
        logger.info(f"Processing {len(itemsets)} itemsets for drug metrics")
        for _, item in itemsets.iterrows():
            # Get the original itemset (frozenset)
            original_itemset = item['itemsets']
            # Convert frozenset to list to get individual drug names
            drugs = list(original_itemset)
            
            # Log sample drug names from itemsets
            if len(drug_metrics) == 0:  # Only log for first few items
                logger.info(f"Sample drugs from itemsets: {drugs[:3]}")
            
            for drug in drugs:
                # Ensure we're using the original drug name from pharmacy data
                # The drug name should be exactly as it appears in the pharmacy column
                drug_record = {
                    'drug_name': drug,  # This should be the original drug name
                    'support': item['support'],
                    'itemsets': original_itemset,
                    'frequency': len(drugs)
                }
                drug_metrics.append(drug_record)
        
        # Process rules for additional metrics
        for _, rule in rules.iterrows():
            # Get the original antecedents (frozenset)
            original_antecedents = rule['antecedents']
            # Convert frozenset to list to get individual drug names
            drugs = list(original_antecedents)
            
            for drug in drugs:
                # Find existing record for this drug or create new one
                existing_record = next((r for r in drug_metrics if r['drug_name'] == drug), None)
                if existing_record:
                    existing_record.update({
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'antecedents': original_antecedents,
                        'consequents': rule['consequents'],
                        'certainty': rule['certainty']
                    })
                else:
                    drug_record = {
                        'drug_name': drug,  # This should be the original drug name
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'antecedents': original_antecedents,
                        'consequents': rule['consequents'],
                        'certainty': rule['certainty']
                    }
                    drug_metrics.append(drug_record)
        
        # Make JSON-safe structures
        itemsets_records = convert_frozensets(itemsets.to_dict(orient="records"), logger)
        rules_records = convert_frozensets(rules.to_dict(orient="records"), logger)
        drug_metrics = convert_frozensets(drug_metrics, logger)

        # Build global encoding map from drug_metrics
        encoding_map = {}
        for rec in drug_metrics:
            drug = rec.get('drug_name')
            if not drug:
                continue
            metrics = {
                'support': rec.get('support', 0),
                'confidence': rec.get('confidence', 0),
                'certainty': rec.get('certainty', 0),
            }
            encoding_map[drug] = encode_drug_name(drug, metrics, logger)

        log_usage("Step 3 (FPGrowth)")
        logger.info(f"Step 3 completed in {time.time() - t0:.2f}s")

        # Step 4: Save output to GOLD/global
        t0 = time.time()
        base = get_global_base_path()
        if period in ["early", "middle"]:
            suffix = f"{age_band}_{period}_{'-'.join(event_year)}"
        else:
            suffix = f"{age_band}_{event_year}"

        save_to_s3_json(itemsets_records, f"{base}/itemsets_{suffix}.json", logger)
        save_to_s3_json(rules_records, f"{base}/rules_{suffix}.json", logger)
        save_to_s3_json(drug_metrics, f"{base}/drug_metrics_{suffix}.json", logger)
        save_to_s3_json(encoding_map, f"{base}/drug_encoding_map_{suffix}.json", logger)
        log_usage("Step 4 (Save)")
        logger.info(f"✅ Saved outputs for {age_band}, {period} in {time.time() - t0:.2f}s")

        return (group, True)

    except Exception as e:
        logger.error(f"❌ Group failed: {group} — {e}")
        return (group, False)


def execute_global_group_fpgrowth_pipeline():
    logger = create_fpgrowth_logger("global_parallel_group_fpgrowth")

    age_bands = AGE_BANDS
    groups = [
        {"period": "early", "years": ["2016", "2017"]},    # 2016, 2017
        {"period": "middle", "years": EVENT_YEARS[2:4]},   # 2018, 2019
        {"period": "recent", "years": EVENT_YEARS[4:]}     # 2020
    ]
    group_definitions = []
    for ab in age_bands:
        for g in groups:
            if g["period"] in ["early", "middle"]:
                # For early and middle periods, create one group with all years
                group_definitions.append({
                    "age_band": ab, 
                    "period": g["period"], 
                    "event_year": g["years"]  # List of years
                })
            else:
                # For recent period, create individual groups for each year
                for y in g["years"]:
                    group_definitions.append({
                        "age_band": ab, 
                        "period": g["period"], 
                        "event_year": y
                    })

    logger.info(f"🚀 Running {len(group_definitions)} groups using ProcessPoolExecutor (32 cores)")

    results = []
    with ProcessPoolExecutor(max_workers=30) as executor:
        future_to_group = {
            executor.submit(run_group_fpgrowth, group): group for group in group_definitions
        }

        for future in as_completed(future_to_group):
            group = future_to_group[future]
            try:
                group_result, success = future.result(timeout=1800)  # 30 min timeout
                results.append((group_result, success))
            except Exception as e:
                logger.error(f"❌ Group {group} failed with exception: {e}")
                results.append((group, False))

    successful = sum(1 for _, success in results if success)
    total = len(group_definitions)
    logger.info(f"🎉 FP-Growth completed: {successful} / {total} groups succeeded.")


# Run it!
execute_global_group_fpgrowth_pipeline()
