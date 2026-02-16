"""
Example script showing how to use S3UploadTracker with dashboard visualizations.

Usage:
    python examples/track_s3_uploads_example.py
"""

from pathlib import Path
from py_helpers.s3_upload_tracker import S3UploadTracker
from py_helpers.checkpoint_utils import upload_file_to_s3
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example of tracking S3 uploads from dashboard visualizations."""
    
    # Initialize tracker
    tracker = S3UploadTracker("status/s3_upload_tracker.json")
    
    # Example: Upload BupaR visualization with tracking
    cohort = "opioid_ed"
    age_band = "1-0-12"
    
    local_file = Path(f"9_dashboard_visuals/bupar/outputs/{cohort}/{age_band}/plots/"
                      f"{cohort}_{age_band.replace('-', '_')}_activity_frequency_interactive.html")
    
    s3_path = f"s3://pgx-dashboard/bupar/{cohort}/{age_band}/plots/{local_file.name}"
    
    # Upload with tracking
    success = upload_file_to_s3(
        local_path=local_file,
        s3_path=s3_path,
        logger=logger,
        check_exists=True,
        tracker=tracker,
        viz_type="bupar",
        cohort=cohort,
        age_band=age_band
    )
    
    print(f"\nUpload {'successful' if success else 'failed'}: {local_file.name}")
    
    # Print summary
    tracker.print_summary()
    
    # Query specific uploads
    print("\n" + "="*80)
    print("QUERYING BUPAR UPLOADS FOR OPIOID_ED:")
    print("="*80)
    bupar_uploads = tracker.get_uploads(
        visualization_type="bupar",
        cohort="opioid_ed",
        success_only=True
    )
    for upload in bupar_uploads:
        print(f"  ✓ {upload['age_band']}: {Path(upload['local_path']).name}")
    
    # Check for missing uploads
    tracker.print_missing_uploads(
        expected_cohorts=["opioid_ed", "non_opioid_ed"],
        expected_age_bands=["1-0-12", "1-13-24", "1-25-44", "1-45-54", 
                           "1-55-64", "1-65-74", "1-75-84", "1-85-114"],
        expected_viz_types=["bupar", "dtw", "fpgrowth"],
        expected_item_types=["drug_name", "icd_code", "cpt_code", "medical_code"]
    )


if __name__ == "__main__":
    main()
