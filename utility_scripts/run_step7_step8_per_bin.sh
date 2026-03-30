#!/usr/bin/env bash
# Run Step 7 (SHAP) and Step 8 (FFA) for each density bin that has Step 6 artifacts.
# Uses the same discovery as notebook 3 (list_trained_density_bins via Python).
#
# Usage:
#   export PGX_REPO_ROOT=/home/ec2-user/pgx-analysis   # or /mnt/nvme/.../pgx-analysis
#   ./utility_scripts/run_step7_step8_per_bin.sh opioid_ed 13-24
#
set -euo pipefail
COHORT="${1:?usage: $0 <cohort> <age_band e.g. 13-24>}"
AGE="${2:?usage: $0 <cohort> <age_band>}"
REPO="${PGX_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
PY="${PGX_PYTHON:-$(command -v python3 || command -v python)}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

bins="$("$PY" -c "
from pathlib import Path
import sys
sys.path.insert(0, r'$REPO')
from py_helpers.event_density_utils import list_trained_density_bins
bins = list_trained_density_bins(Path(r'$REPO'), '$COHORT', '$AGE')
print(' '.join(bins))
")"

if [[ -z "${bins// }" ]]; then
  echo "No per-bin Step 6 artifacts found for $COHORT / $AGE (check PROJECT_ROOT and PGX_DATA_ROOT)."
  exit 1
fi

echo "Trained bins: $bins"
for b in $bins; do
  echo "======== Step 7 SHAP: bin=$b ========"
  "$PY" "$REPO/7_shap_analysis/run_shap_analysis.py" --cohort "$COHORT" --age_band "$AGE" --bin "$b"
  echo "======== Step 8 FFA: bin=$b ========"
  "$PY" "$REPO/10_risk_dashboard/data_preparation/run_shap_ffa_workflow.py" \
    --cohort "$COHORT" --age-band "$AGE" --bin "$b" --skip-shap --skip-combine
done
echo "Done. Run Combine from notebook 3 or combine_shap_ffa_results per bin."
