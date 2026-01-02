#!/usr/bin/env python3
"""
Analyze storage requirements for Lambda ECR container.

Calculates total size of:
- Model files (joblib + JSON)
- Feature schemas
- Metadata files
- Dependencies

Estimates total container size.
"""

import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent

# Expected cohorts and age bands
OPIOID_ED_AGE_BANDS = ["13-24", "25-44", "45-54", "55-64"]
POLYPHARMACY_AGE_BANDS = ["65-74", "75-84", "85-94"]

def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    if path.exists():
        return path.stat().st_size
    return 0

def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def analyze_cohort_storage(cohort: str, age_bands: list) -> dict:
    """Analyze storage for a cohort."""
    results = {
        'cohort': cohort,
        'age_bands': {},
        'total': 0
    }
    
    for age_band in age_bands:
        age_band_fname = age_band.replace("-", "_")
        cohort_dir = PROJECT_ROOT / '8_final_model' / 'outputs' / cohort / age_band_fname
        
        if not cohort_dir.exists():
            print(f"  {age_band}: Not found (skipping)")
            continue
        
        # Model files
        models_dir = cohort_dir / 'models'
        json_dir = cohort_dir / 'final_model_json'
        
        model_files = {
            'joblib': 0,
            'json_catboost': 0,
            'json_xgboost': 0,
            'json_xgboost_rf': 0,
            'mc_cv_results': 0,
            'feature_schema': 0,
            'total': 0
        }
        
        # Check joblib model
        joblib_file = models_dir / f'{cohort}_{age_band_fname}_final_model.joblib'
        if joblib_file.exists():
            model_files['joblib'] = get_file_size(joblib_file)
        
        # Check JSON models
        json_catboost = json_dir / f'{cohort}_{age_band_fname}_final_model_catboost.json'
        json_xgboost = json_dir / f'{cohort}_{age_band_fname}_final_model_xgboost.json'
        json_xgboost_rf = json_dir / f'{cohort}_{age_band_fname}_final_model_xgboost_rf.json'
        
        if json_catboost.exists():
            model_files['json_catboost'] = get_file_size(json_catboost)
        if json_xgboost.exists():
            model_files['json_xgboost'] = get_file_size(json_xgboost)
        if json_xgboost_rf.exists():
            model_files['json_xgboost_rf'] = get_file_size(json_xgboost_rf)
        
        # MC-CV results
        mc_cv_file = models_dir / f'{cohort}_{age_band_fname}_mc_cv_results.csv'
        if mc_cv_file.exists():
            model_files['mc_cv_results'] = get_file_size(mc_cv_file)
        
        # Feature schema (will be generated)
        # Estimate: ~50KB per age_band
        
        model_files['total'] = sum([
            model_files['joblib'],
            model_files['json_catboost'],
            model_files['json_xgboost'],
            model_files['json_xgboost_rf'],
            model_files['mc_cv_results']
        ])
        
        results['age_bands'][age_band] = model_files
        results['total'] += model_files['total']
        
        print(f"  {age_band}: {format_size(model_files['total'])}")
        print(f"    - CatBoost JSON: {format_size(model_files['json_catboost'])}")
        print(f"    - XGBoost JSON: {format_size(model_files['json_xgboost'])}")
        print(f"    - XGBoost RF JSON: {format_size(model_files['json_xgboost_rf'])}")
        print(f"    - Joblib: {format_size(model_files['joblib'])}")
    
    return results

def main():
    print("=" * 60)
    print("Lambda ECR Container Storage Analysis")
    print("=" * 60)
    print()
    
    # Analyze existing data
    print("Analyzing existing model files...")
    print()
    
    opioid_results = analyze_cohort_storage('opioid_ed', OPIOID_ED_AGE_BANDS)
    print()
    
    polypharmacy_results = analyze_cohort_storage('non_opioid_ed', POLYPHARMACY_AGE_BANDS)
    print()
    
    # Calculate totals
    total_models = opioid_results['total'] + polypharmacy_results['total']
    
    # Estimate other components
    print("Estimating other components...")
    
    # Feature importance CSVs (metadata)
    feature_importance_dir = PROJECT_ROOT / '3_feature_importance' / 'outputs'
    metadata_size = sum(
        get_file_size(f) for f in feature_importance_dir.glob('*.csv')
    )
    print(f"  Feature importance CSVs: {format_size(metadata_size)}")
    
    # Feature schemas (estimated)
    # ~50KB per age_band × 7 age_bands = ~350KB
    schema_size = 50 * 1024 * 7  # 7 age bands
    print(f"  Feature schemas (estimated): {format_size(schema_size)}")
    
    # Metadata JSON files
    # ~100KB per cohort × 2 cohorts = ~200KB
    metadata_json_size = 100 * 1024 * 2
    print(f"  Metadata JSON files (estimated): {format_size(metadata_json_size)}")
    
    # Python dependencies (CatBoost, XGBoost, etc.)
    # CatBoost: ~500MB
    # XGBoost: ~200MB
    # Other (pandas, numpy, etc.): ~300MB
    dependencies_size = (500 + 200 + 300) * 1024 * 1024
    print(f"  Python dependencies (estimated): {format_size(dependencies_size)}")
    
    # Base Lambda Python image
    base_image_size = 500 * 1024 * 1024  # ~500MB
    print(f"  Base Lambda image: {format_size(base_image_size)}")
    
    # Total
    total_size = (
        total_models +
        metadata_size +
        schema_size +
        metadata_json_size +
        dependencies_size +
        base_image_size
    )
    
    print()
    print("=" * 60)
    print("STORAGE SUMMARY")
    print("=" * 60)
    print(f"Models (existing): {format_size(total_models)}")
    print(f"Metadata: {format_size(metadata_size + schema_size + metadata_json_size)}")
    print(f"Dependencies: {format_size(dependencies_size)}")
    print(f"Base image: {format_size(base_image_size)}")
    print(f"{'-' * 60}")
    print(f"TOTAL ESTIMATED: {format_size(total_size)}")
    print("=" * 60)
    print()
    
    # Projection for all age bands
    print("Projection for ALL age bands:")
    print()
    
    # Average size per age_band (from existing data)
    avg_per_age_band = total_models / max(len([k for k in opioid_results['age_bands'].keys()]), 1)
    
    total_age_bands = len(OPIOID_ED_AGE_BANDS) + len(POLYPHARMACY_AGE_BANDS)
    projected_models = avg_per_age_band * total_age_bands
    
    projected_total = (
        projected_models +
        metadata_size +
        schema_size * (total_age_bands / 7) +
        metadata_json_size +
        dependencies_size +
        base_image_size
    )
    
    print(f"  Models (all {total_age_bands} age bands): {format_size(projected_models)}")
    print(f"  Other components: {format_size(projected_total - projected_models)}")
    print(f"  PROJECTED TOTAL: {format_size(projected_total)}")
    print()
    
    # ECR limit check
    ecr_limit = 10 * 1024 * 1024 * 1024  # 10GB
    print("=" * 60)
    print("ECR LIMIT CHECK")
    print("=" * 60)
    print(f"ECR Container Limit: {format_size(ecr_limit)}")
    print(f"Current Usage: {format_size(total_size)}")
    print(f"Projected Usage (all age bands): {format_size(projected_total)}")
    print()
    
    if projected_total < ecr_limit:
        print("✅ SUCCESS: All data fits within ECR 10GB limit!")
        print(f"   Remaining space: {format_size(ecr_limit - projected_total)}")
    else:
        print("⚠️  WARNING: Projected size exceeds ECR limit")
        print(f"   Over by: {format_size(projected_total - ecr_limit)}")
        print("   Consider: Model quantization or selective age band inclusion")
    
    print()
    print("=" * 60)

if __name__ == '__main__':
    main()

