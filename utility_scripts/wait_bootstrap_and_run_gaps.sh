#!/usr/bin/env bash
# Compatibility wrapper. Canonical script:
#   aws-pgx-setup/ec2/scripts/bash/wait_bootstrap_and_run_gaps.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/aws-pgx-setup/ec2/scripts/bash/wait_bootstrap_and_run_gaps.sh" "$@"
