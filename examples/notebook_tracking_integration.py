# S3 Upload Tracking - Dashboard Visuals Integration

# Add this cell at the beginning of 4_dashboard_visuals.ipynb to enable tracking

# Initialize S3 Upload Tracker
from py_helpers.s3_upload_tracker import S3UploadTracker

tracker = S3UploadTracker("status/s3_upload_tracker.json")
print("✓ S3 Upload Tracker initialized")

# ============================================================================
# Example: Track BupaR uploads
# ============================================================================

from pathlib import Path
from py_helpers.checkpoint_utils import upload_file_to_s3
import logging

logger = logging.getLogger(__name__)

cohort = "opioid_ed"
age_band = "1-0-12"
age_band_fname = age_band.replace("-", "_")

# Upload activity frequency with tracking
local_file = Path(f"9_dashboard_visuals/bupar/outputs/{cohort}/{age_band}/plots/"
                  f"{cohort}_{age_band_fname}_activity_frequency_interactive.html")
s3_path = f"s3://pgx-dashboard/bupar/{cohort}/{age_band}/plots/{local_file.name}"

success = upload_file_to_s3(
    local_path=local_file,
    s3_path=s3_path,
    logger=logger,
    check_exists=True,
    tracker=tracker,         # Enable tracking
    viz_type="bupar",        # Track as BupaR visualization
    cohort=cohort,           # Track cohort
    age_band=age_band        # Track age band
)

print(f"Upload {'✓' if success else '✗'}: {local_file.name}")

# ============================================================================
# Example: Track DTW uploads
# ============================================================================

local_file = Path(f"9_dashboard_visuals/dtw/outputs/{cohort}/{age_band}/plots/"
                  f"dtw_trajectory_cluster_interactive_{cohort}_{age_band_fname}.html")
s3_path = f"s3://pgx-dashboard/dtw/{cohort}/{age_band}/plots/{local_file.name}"

success = upload_file_to_s3(
    local_path=local_file,
    s3_path=s3_path,
    logger=logger,
    tracker=tracker,
    viz_type="dtw",
    cohort=cohort,
    age_band=age_band
)

# ============================================================================
# Example: Track FP-Growth uploads (with item_type)
# ============================================================================

item_type = "drug_name"
local_file = Path(f"outputs/{cohort}_{age_band_fname}_{item_type}_itemsets_interactive.html")
s3_path = f"s3://pgx-dashboard/fpgrowth/{cohort}/{age_band}/plots/{local_file.name}"

success = upload_file_to_s3(
    local_path=local_file,
    s3_path=s3_path,
    logger=logger,
    tracker=tracker,
    viz_type="fpgrowth",
    cohort=cohort,
    age_band=age_band,
    item_type=item_type      # Track item type for FP-Growth
)

# ============================================================================
# Batch Upload with Tracking
# ============================================================================

def upload_all_bupar_visualizations(tracker):
    """Upload all BupaR visualizations with tracking."""
    
    cohorts = ["opioid_ed", "non_opioid_ed"]
    age_bands = ["1-0-12", "1-13-24", "1-25-44", "1-45-54", 
                 "1-55-64", "1-65-74", "1-75-84", "1-85-114"]
    
    viz_files = [
        "activity_frequency_interactive.html",
        "trace_explorer_interactive.html",
        "process_matrix_interactive.html"
    ]
    
    total_uploaded = 0
    total_failed = 0
    
    for cohort in cohorts:
        for age_band in age_bands:
            age_band_fname = age_band.replace("-", "_")
            
            for viz_file in viz_files:
                local_file = Path(f"9_dashboard_visuals/bupar/outputs/{cohort}/{age_band}/plots/"
                                f"{cohort}_{age_band_fname}_{viz_file}")
                s3_path = f"s3://pgx-dashboard/bupar/{cohort}/{age_band}/plots/{local_file.name}"
                
                success = upload_file_to_s3(
                    local_path=local_file,
                    s3_path=s3_path,
                    logger=logger,
                    tracker=tracker,
                    viz_type="bupar",
                    cohort=cohort,
                    age_band=age_band
                )
                
                if success:
                    total_uploaded += 1
                else:
                    total_failed += 1
    
    print(f"\nBupaR Upload Summary:")
    print(f"  ✓ Uploaded: {total_uploaded}")
    print(f"  ✗ Failed: {total_failed}")
    
    return total_uploaded, total_failed

# Execute batch upload
uploaded, failed = upload_all_bupar_visualizations(tracker)

# ============================================================================
# View Upload Status
# ============================================================================

# Print summary report
tracker.print_summary()

# Check for missing uploads
tracker.print_missing_uploads(
    expected_cohorts=["opioid_ed", "non_opioid_ed"],
    expected_age_bands=["1-0-12", "1-13-24", "1-25-44", "1-45-54", 
                       "1-55-64", "1-65-74", "1-75-84", "1-85-114"],
    expected_viz_types=["bupar", "dtw", "fpgrowth"],
    expected_item_types=["drug_name", "icd_code", "cpt_code", "medical_code"]
)

# ============================================================================
# Query Specific Uploads
# ============================================================================

# Get all successful BupaR uploads for opioid_ed
bupar_opioid = tracker.get_uploads(
    visualization_type="bupar",
    cohort="opioid_ed",
    success_only=True
)

print(f"\nSuccessful BupaR uploads for opioid_ed: {len(bupar_opioid)}")

# Get all failed uploads
all_uploads = tracker.get_uploads()
failed_uploads = [u for u in all_uploads if not u["success"]]

print(f"Total failed uploads: {len(failed_uploads)}")
for upload in failed_uploads:
    print(f"  ✗ {upload['visualization_type']}/{upload['cohort']}/{upload['age_band']}")
    print(f"    Error: {upload['error']}")

# ============================================================================
# Export tracking data for analysis
# ============================================================================

import pandas as pd

# Convert to DataFrame for analysis
df_uploads = pd.DataFrame(tracker.uploads["uploads"])

# Successful uploads by visualization type
success_by_viz = df_uploads[df_uploads["success"]].groupby("visualization_type").size()
print("\nSuccessful uploads by visualization type:")
print(success_by_viz)

# Failed uploads by error
if len(failed_uploads) > 0:
    df_failed = pd.DataFrame(failed_uploads)
    print("\nFailed uploads by error:")
    print(df_failed.groupby("error").size())

# Total size uploaded
total_size = df_uploads[df_uploads["success"]]["file_size_mb"].sum()
print(f"\nTotal data uploaded: {total_size:.2f} MB")
