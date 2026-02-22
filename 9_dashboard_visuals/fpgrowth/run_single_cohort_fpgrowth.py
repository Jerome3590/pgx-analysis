#!/usr/bin/env python3
"""
Run FP-Growth analysis for a single cohort/age_band combination.

Item types are cohort-dependent (from cohort_fpgrowth.get_item_types_for_cohort):
- non_opioid_ed (polypharmacy): drug_name only
- opioid_ed: drug_name, icd_code, cpt_code
This script calls process_single_cohort directly for a specific cohort/age_band.
"""

import json
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psutil
except ImportError:
    psutil = None  # noqa: I001

# Add project root to path
# Script lives in 9_dashboard_visuals/fpgrowth; outputs go to 10_risk_dashboard/visualizations/fpgrowth
REPO_ROOT = Path(__file__).resolve().parents[2]
FPGROWTH_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FPGROWTH_CODE_DIR))

# Import cohort_fpgrowth from same directory
import importlib.util
cohort_fpgrowth_path = FPGROWTH_CODE_DIR / "cohort_fpgrowth.py"
spec = importlib.util.spec_from_file_location("cohort_fpgrowth", cohort_fpgrowth_path)
cohort_fpgrowth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort_fpgrowth)

process_single_cohort = cohort_fpgrowth.process_single_cohort
_model_data_paths = cohort_fpgrowth._model_data_paths
get_item_types_for_cohort = cohort_fpgrowth.get_item_types_for_cohort
MIN_SUPPORT = cohort_fpgrowth.MIN_SUPPORT
MIN_CONFIDENCE = cohort_fpgrowth.MIN_CONFIDENCE
S3_OUTPUT_BASE = cohort_fpgrowth.S3_OUTPUT_BASE
MODEL_DATA_ROOT = cohort_fpgrowth.MODEL_DATA_ROOT
LOCAL_DATA_PATH = cohort_fpgrowth.LOCAL_DATA_PATH

def main():
    parser = argparse.ArgumentParser(description="Run FP-Growth for a single cohort/age_band")
    parser.add_argument("--cohort-name", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (e.g., 0-12)")
    parser.add_argument("--event-year", default="train", help="Event year (train, 2019, etc.)")
    parser.add_argument("--project-root", default=None, help="Repo root for model_events resolution (same as BupaR/DTW); default=inferred from script path")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else None
    model_data_paths = None

    # Upfront path check for TRAIN: require model_data; fail fast with paths_checked/path_listings
    if args.event_year == "train":
        model_data_paths = _model_data_paths(args.cohort_name, args.age_band, project_root=project_root)
        if not model_data_paths:
            root = project_root if project_root is not None else REPO_ROOT
            try:
                from py_helpers.model_data_paths import get_model_events_paths_checked, get_path_check_listings
                paths_checked = get_model_events_paths_checked(root, args.cohort_name, args.age_band)
                path_listings = get_path_check_listings(paths_checked) if paths_checked else []
            except Exception:  # noqa: BLE001
                paths_checked = []
                path_listings = []
            print("[ERROR] TRAIN requires model_data; none found for this cohort/age_band.", flush=True)
            print("[ERROR_PARAMS]", {"cohort_name": args.cohort_name, "age_band": args.age_band, "event_year": args.event_year, "error": "TRAIN model_data not found", "paths_checked": paths_checked, "path_listings": path_listings}, flush=True)
            sys.exit(1)
        try:
            from py_helpers.model_data_paths import confirm_paths_exist_with_listings
            all_ok, confirm_listings = confirm_paths_exist_with_listings(model_data_paths)
            for line in confirm_listings:
                print(f"[PATH_CONFIRM] {line}", flush=True)
            if not all_ok:
                print("[ERROR] Model data path(s) missing or empty; aborting.", flush=True)
                print("[ERROR_PARAMS]", {"cohort_name": args.cohort_name, "age_band": args.age_band, "path_listings": confirm_listings}, flush=True)
                sys.exit(1)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Path confirm failed: {e}", flush=True)
            sys.exit(1)

    # Use model_data if available, otherwise use local data path
    # When TRAIN and we have resolved paths, show the actual data root (e.g. /mnt/nvme/4_model_data)
    if args.event_year == "train" and model_data_paths:
        # path is .../cohort_name=X/age_band=Y/file.parquet -> parent.parent.parent = 4_model_data root
        local_data_path = model_data_paths[0].parent.parent.parent
    else:
        local_data_path = MODEL_DATA_ROOT if MODEL_DATA_ROOT.exists() else LOCAL_DATA_PATH

    item_types = get_item_types_for_cohort(args.cohort_name)
    print("="*70, flush=True)
    print(f"FP-GROWTH ITEMSET MINING: {args.cohort_name} / {args.age_band} / {args.event_year}", flush=True)
    print("="*70, flush=True)
    print(f"Data path: {local_data_path}", flush=True)
    print(f"Item types: {', '.join(item_types)}", flush=True)
    print(f"Min support: {MIN_SUPPORT}, Min confidence: {MIN_CONFIDENCE}", flush=True)
    print("", flush=True)
    
    def _log_resources(label: str) -> None:
        if psutil is None:
            return
        try:
            mem = psutil.virtual_memory()
            proc = psutil.Process()
            mem_gb = mem.used / (1024**3)
            cpu = proc.cpu_percent(interval=None)
            print(f"[RESOURCE {label}] mem_used_gb={mem_gb:.1f} cpu_pct={cpu:.1f}", flush=True)
        except Exception:
            pass

    # Process item types in parallel (count = len(item_types) for this cohort)
    any_ok = False
    failures = []  # (item_type, error_msg)
    
    def process_item_type(item_type, idx):
        """Process single item type; returns (item_type, result_or_error)"""
        _log_resources(f"before {item_type}")
        print("", flush=True)
        print("-" * 70, flush=True)
        print(f"[ITEM TYPE {idx}/{len(item_types)}] Processing {item_type}...", flush=True)
        print("-" * 70, flush=True)
        try:
            result = process_single_cohort(
                item_type=item_type,
                cohort_name=args.cohort_name,
                age_band=args.age_band,
                event_year=args.event_year,
                local_data_path=local_data_path,
                s3_output_base=S3_OUTPUT_BASE,
                min_support=MIN_SUPPORT,
                min_confidence=MIN_CONFIDENCE,
                project_root=project_root,
            )
            _log_resources(f"after {item_type}")
            return (item_type, result, None)
        except Exception as e:
            _log_resources(f"after {item_type} (exception)")
            import traceback
            tb = traceback.format_exc()
            return (item_type, None, (e, tb))
    
    # Parallelize item type processing (4 workers for 4 item types)
    with ThreadPoolExecutor(max_workers=len(item_types)) as executor:
        futures = {executor.submit(process_item_type, item_type, idx): item_type
                   for idx, item_type in enumerate(item_types, 1)}
        
        for future in as_completed(futures):
            item_type, result, exception_info = future.result()
            
            if exception_info:
                e, tb = exception_info
                failures.append((item_type, str(e)))
                print(f"[ERROR] {item_type} failed: {e}", flush=True)
                err_params = {"cohort_name": args.cohort_name, "age_band": args.age_band, "item_type": item_type, "error": str(e)}
                print("[ERROR_PARAMS]", err_params, flush=True)
                print(tb, flush=True)
            elif 'error' in result:
                err_msg = result['error']
                failures.append((item_type, err_msg))
                print(f"[ERROR] {item_type}: {err_msg}", flush=True)
                # Log missing/mismatched params so follow-on runs can correct (paths_checked, path, path_listings, etc.)
                params = {k: v for k, v in result.items() if k in ("cohort_name", "age_band", "item_type", "error", "paths_checked", "path", "path_listings")}
                if result.get("paths_checked") and "path_listings" not in params:
                    try:
                        from py_helpers.model_data_paths import get_path_check_listings
                        params["path_listings"] = get_path_check_listings(result["paths_checked"])
                    except Exception:  # noqa: BLE001
                        pass
                if params:
                    print(f"[ERROR_PARAMS] {params}", flush=True)
            else:
                any_ok = True
                itemset_count = result.get('frequent_itemsets', result.get('itemsets_count', 0))
                rules_count = result.get('association_rules', result.get('rules_count', 0))
                print(f"[OK] {item_type}: {itemset_count} itemsets, {rules_count} rules", flush=True)
                if itemset_count > 0:
                    print(f"   Generated {itemset_count} frequent itemsets", flush=True)
                    if rules_count > 0:
                        print(f"   Generated {rules_count} association rules", flush=True)

    print("", flush=True)
    print("="*70, flush=True)
    if any_ok:
        success_count = len(item_types) - len(failures)
        print(f"[OK] FP-GROWTH COMPLETE: {success_count}/{len(item_types)} item types successful", flush=True)
        if failures:
            print(f"   {len(failures)} item types failed (see errors above)", flush=True)
    else:
        print(f"[FAIL] FP-GROWTH FAILED: All {len(item_types)} item types failed", flush=True)
        summary = "; ".join(f"{t}={e}" for t, e in failures) if failures else "no item types produced itemsets"
        # When only "No frequent itemsets" (e.g. small cohort / insufficient transactions), exit 0 so pipeline continues
        only_no_itemsets = failures and all(e == "No frequent itemsets" for _, e in failures)
        if only_no_itemsets:
            print(f"\nNo itemsets for any item type (insufficient transactions for {args.cohort_name}/{args.age_band}); writing empty itemset JSON so workflow continues.")
            age_band_fname = args.age_band.replace("-", "_")
            # Itemsets output dir (same path ensure_itemsets checks for *_itemsets*.json)
            itemsets_dir = (
                REPO_ROOT
                / "10_risk_dashboard"
                / "visualizations"
                / "fpgrowth"
                / "outputs"
                / args.cohort_name
                / age_band_fname
            )
            itemsets_dir.mkdir(parents=True, exist_ok=True)
            empty_message = "No frequent itemsets or rules for this cohort/age band (insufficient transactions)."
            for item_type in item_types:
                for suffix in ("_itemsets.json", "_rules.json", "_itemsets_target_only.json", "_rules_target_only.json"):
                    path = itemsets_dir / f"{item_type}{suffix}"
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump([], f)
                    print(f"Wrote empty {path.name}", flush=True)
            # Optional: write a small metadata JSON so downstream can show a message
            meta_path = itemsets_dir / "_empty_reason.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"empty": True, "message": empty_message, "cohort_name": args.cohort_name, "age_band": args.age_band}, f, indent=2)
            # Dashboard empty-state in plots/ for frontend to show message
            plots_dir = itemsets_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            empty_state = {
                "empty": True,
                "message": empty_message,
                "cohort_name": args.cohort_name,
                "age_band": args.age_band,
            }
            empty_path = plots_dir / "empty_state.json"
            with open(empty_path, "w", encoding="utf-8") as f:
                json.dump(empty_state, f, indent=2)
            print(f"Wrote dashboard empty-state: {empty_path}", flush=True)
        else:
            print(f"\nFP-Growth itemsets creation failed: {summary}. See [ERROR] / [ERROR_PARAMS] above (paths_checked, path_listings).")
            sys.exit(1)

if __name__ == "__main__":
    main()

