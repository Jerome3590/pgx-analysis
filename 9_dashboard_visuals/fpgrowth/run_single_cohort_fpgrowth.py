#!/usr/bin/env python3
"""
Run FP-Growth analysis for a single cohort/age_band combination.

Item types (from cohort_fpgrowth.ITEM_TYPES): drug_name, icd_code, cpt_code, medical_code.
This script calls process_single_cohort directly for a specific cohort/age_band.
"""

import sys
import argparse
from pathlib import Path

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
MIN_SUPPORT = cohort_fpgrowth.MIN_SUPPORT
MIN_CONFIDENCE = cohort_fpgrowth.MIN_CONFIDENCE
ITEM_TYPES = cohort_fpgrowth.ITEM_TYPES
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

    # Use model_data if available, otherwise use local data path
    local_data_path = MODEL_DATA_ROOT if MODEL_DATA_ROOT.exists() else LOCAL_DATA_PATH

    print(f"Running FP-Growth for {args.cohort_name} / {args.age_band} / {args.event_year}")
    print(f"Using data path: {local_data_path}")
    
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

    # Process each item type; track if any succeeded; collect failures for summary
    any_ok = False
    failures = []  # (item_type, error_msg)
    for item_type in ITEM_TYPES:
        _log_resources(f"before {item_type}")
        print(f"\nProcessing {item_type}...", flush=True)
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
            if 'error' in result:
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
                print(f"[OK] {item_type}: {result.get('itemsets_count', 0)} itemsets, {result.get('rules_count', 0)} rules")
            _log_resources(f"after {item_type}")
        except Exception as e:
            _log_resources(f"after {item_type} (exception)")
            failures.append((item_type, str(e)))
            print(f"[ERROR] {item_type} failed: {e}", flush=True)
            err_params = {"cohort_name": args.cohort_name, "age_band": args.age_band, "item_type": item_type, "error": str(e)}
            print("[ERROR_PARAMS]", err_params, flush=True)
            import traceback
            traceback.print_exc()

    if any_ok:
        print("\nFP-Growth itemsets creation complete!")
    else:
        summary = "; ".join(f"{t}={e}" for t, e in failures) if failures else "no item types produced itemsets"
        print(f"\nFP-Growth itemsets creation failed: {summary}. See [ERROR] / [ERROR_PARAMS] above (paths_checked, path_listings).")
        sys.exit(1)

if __name__ == "__main__":
    main()

