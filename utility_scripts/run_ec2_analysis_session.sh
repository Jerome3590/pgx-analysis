#!/usr/bin/env bash
# Compatibility wrapper. Canonical script:
#   aws-pgx-setup/ec2/scripts/bash/run_ec2_analysis_session.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/aws-pgx-setup/ec2/scripts/bash/run_ec2_analysis_session.sh" "$@"
