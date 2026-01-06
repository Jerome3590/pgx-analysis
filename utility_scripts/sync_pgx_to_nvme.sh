#!/usr/bin/env bash
set -euo pipefail

##############################################
# Configuration
##############################################

# 1) GitHub repo URL for this project
REPO_URL="https://github.com/Jerome3590/pgx-analysis.git"

# 2) Where to keep the git clone on EC2 (code stays on $HOME)
CLONE_DIR="$HOME/pgx-analysis"

# 3) Where the fast NVMe volume is mounted (for data, temp, staging)
NVME_ROOT="/mnt/nvme"
NVME_DEVICE="/dev/nvme1n1"  # Adjust if your device is different

# 4) Optional: where to write a small env file to source in your shell
ENV_FILE="$HOME/.pgx_env"

# 5) S3 bucket and prefixes to sync data locally (to NVMe)
S3_BUCKET="pgxdatalake"
S3_GOLD_PREFIX="s3://${S3_BUCKET}/gold"

##############################################
# Functions
##############################################

log() {
  printf '[pgx-sync] %s\n' "$*"
}

detect_os_and_resources() {
  OS_TYPE="$(uname -s || echo unknown)"
  CORES="$(python3 - <<'EOF'
import os
print(os.cpu_count() or 1)
EOF
)"

  TOTAL_RAM_GB=16
  if [ "$OS_TYPE" = "Linux" ] && [ -r /proc/meminfo ]; then
    mem_kb=$(grep -i '^MemTotal:' /proc/meminfo | awk '{print $2}')
    if [ -n "$mem_kb" ]; then
      TOTAL_RAM_GB=$((mem_kb / 1024 / 1024))
      if [ "$TOTAL_RAM_GB" -lt 1 ]; then
        TOTAL_RAM_GB=1
      fi
    fi
  else
    TOTAL_RAM_GB="$(python3 - <<'EOF'
try:
    import psutil
    print(int(psutil.virtual_memory().total / (1024**3)))
except Exception:
    print(16)
EOF
)"
  fi

  export PGX_OS_TYPE="$OS_TYPE"
  export PGX_CPU_CORES="$CORES"
  export PGX_TOTAL_RAM_GB="$TOTAL_RAM_GB"

  # Derive default worker and memory settings
  if [ -z "${PGX_WORKERS_MEDICAL:-}" ]; then
    if [ "$TOTAL_RAM_GB" -ge 512 ]; then
      PGX_WORKERS_MEDICAL=28
    elif [ "$TOTAL_RAM_GB" -ge 128 ]; then
      PGX_WORKERS_MEDICAL=18
    elif [ "$TOTAL_RAM_GB" -ge 64 ]; then
      PGX_WORKERS_MEDICAL=12
    else
      PGX_WORKERS_MEDICAL=8
    fi
    export PGX_WORKERS_MEDICAL
  fi

  if [ -z "${PGX_DUCKDB_MEMORY_LIMIT:-}" ]; then
    if [ "$TOTAL_RAM_GB" -ge 256 ]; then
      PGX_DUCKDB_MEMORY_LIMIT="3GB"
    elif [ "$TOTAL_RAM_GB" -ge 64 ]; then
      PGX_DUCKDB_MEMORY_LIMIT="2GB"
    else
      PGX_DUCKDB_MEMORY_LIMIT="1GB"
    fi
    export PGX_DUCKDB_MEMORY_LIMIT
  fi

  if [ -z "${PGX_THREADS_PER_WORKER:-}" ]; then
    PGX_THREADS_PER_WORKER=1
    export PGX_THREADS_PER_WORKER
  fi

  if [ -z "${PGX_SKLEARN_N_JOBS:-}" ]; then
    if [ "$TOTAL_RAM_GB" -ge 256 ]; then
      PGX_SKLEARN_N_JOBS=8
    elif [ "$TOTAL_RAM_GB" -ge 64 ]; then
      PGX_SKLEARN_N_JOBS=4
    else
      PGX_SKLEARN_N_JOBS=2
    fi
    export PGX_SKLEARN_N_JOBS
  fi

  if [ -z "${PGX_XGB_CPU_NTHREAD:-}" ]; then
    PGX_XGB_CPU_NTHREAD="$PGX_SKLEARN_N_JOBS"
    export PGX_XGB_CPU_NTHREAD
  fi

  if [ -z "${PGX_MC_CV_WORKERS:-}" ]; then
    if [ "$TOTAL_RAM_GB" -ge 256 ]; then
      PGX_MC_CV_WORKERS=8
    elif [ "$TOTAL_RAM_GB" -ge 64 ]; then
      PGX_MC_CV_WORKERS=4
    else
      PGX_MC_CV_WORKERS=2
    fi
    export PGX_MC_CV_WORKERS
  fi

  if [ -z "${PGX_MC_CV_N_SPLITS:-}" ]; then
    if [ "$TOTAL_RAM_GB" -ge 256 ] && [ "$OS_TYPE" = "Linux" ]; then
      PGX_MC_CV_N_SPLITS=50
    else
      PGX_MC_CV_N_SPLITS=200
    fi
    export PGX_MC_CV_N_SPLITS
  fi

  if [ -z "${PGX_MC_CV_N_RUNS:-}" ]; then
    if [ "$TOTAL_RAM_GB" -ge 256 ] && [ "$OS_TYPE" = "Linux" ]; then
      PGX_MC_CV_N_RUNS=3
    else
      PGX_MC_CV_N_RUNS=1
    fi
    export PGX_MC_CV_N_RUNS
  fi
}

mount_nvme() {
  # Check if already mounted
  if mountpoint -q "$NVME_ROOT" 2>/dev/null; then
    log "NVMe volume already mounted at $NVME_ROOT"
    # Verify write access
    if touch "$NVME_ROOT/.test_write" 2>/dev/null && rm "$NVME_ROOT/.test_write" 2>/dev/null; then
      log "NVMe mount verified and writable"
      return 0
    else
      log "Warning: NVMe mounted but not writable. Fixing permissions..."
    fi
  fi

  # Only proceed on Linux
  if [ "$OS_TYPE" != "Linux" ]; then
    log "Skipping NVMe mount (non-Linux system)"
    return 0
  fi

  # Check if device exists
  if [ ! -b "$NVME_DEVICE" ]; then
    log "Warning: NVMe device $NVME_DEVICE not found. Skipping mount."
    log "Available devices:"
    lsblk 2>/dev/null || true
    return 0
  fi

  # Check if we have root/sudo privileges
  if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    log "Need sudo privileges to mount NVMe. Please run:"
    log "  sudo $0"
    return 1
  fi

  log "Setting up NVMe device $NVME_DEVICE..."

  # Check if device has a filesystem
  if ! blkid "$NVME_DEVICE" >/dev/null 2>&1; then
    log "Formatting $NVME_DEVICE with ext4..."
    if [ "$EUID" -eq 0 ]; then
      mkfs.ext4 -F "$NVME_DEVICE"
    else
      sudo mkfs.ext4 -F "$NVME_DEVICE"
    fi
  fi

  # Create mount point
  log "Creating mount point at $NVME_ROOT..."
  if [ "$EUID" -eq 0 ]; then
    mkdir -p "$NVME_ROOT"
  else
    sudo mkdir -p "$NVME_ROOT"
  fi

  # Mount the device
  log "Mounting $NVME_DEVICE to $NVME_ROOT..."
  if [ "$EUID" -eq 0 ]; then
    mount "$NVME_DEVICE" "$NVME_ROOT"
  else
    sudo mount "$NVME_DEVICE" "$NVME_ROOT"
  fi

  # Set permissions
  log "Setting permissions for user $USER..."
  if [ "$EUID" -eq 0 ]; then
    chmod 755 "$NVME_ROOT"
    chown "${SUDO_USER:-$USER}:${SUDO_USER:-$USER}" "$NVME_ROOT" 2>/dev/null || true
  else
    sudo chmod 755 "$NVME_ROOT"
    sudo chown "$USER:$USER" "$NVME_ROOT" 2>/dev/null || true
  fi

  # Verify mount and write access
  log "Verifying mount..."
  if mountpoint -q "$NVME_ROOT" 2>/dev/null; then
    if touch "$NVME_ROOT/.test_write" 2>/dev/null && rm "$NVME_ROOT/.test_write" 2>/dev/null; then
      log "Mount successful and writable!"
    else
      log "Warning: Mount successful but write test failed"
    fi
  else
    log "Warning: Mount may have failed"
  fi

  # Add to /etc/fstab for persistence
  UUID=$(blkid -s UUID -o value "$NVME_DEVICE" 2>/dev/null || echo "")
  if [ -n "$UUID" ] && ! grep -q "$NVME_ROOT" /etc/fstab 2>/dev/null; then
    log "Adding $NVME_DEVICE to /etc/fstab for automatic mounting..."
    FSTAB_ENTRY="UUID=$UUID $NVME_ROOT ext4 defaults,nofail 0 2"
    if [ "$EUID" -eq 0 ]; then
      echo "$FSTAB_ENTRY" >> /etc/fstab
    else
      echo "$FSTAB_ENTRY" | sudo tee -a /etc/fstab >/dev/null
    fi
    log "Added to /etc/fstab. Device will auto-mount on reboot."
  fi

  # Show mount status
  df -h "$NVME_ROOT" 2>/dev/null || true
}

ensure_clone() {
  if [ -d "$CLONE_DIR/.git" ]; then
    log "Existing clone found at $CLONE_DIR, pulling latest..."
    git -C "$CLONE_DIR" fetch --all --prune
    git -C "$CLONE_DIR" checkout main
    git -C "$CLONE_DIR" pull --ff-only origin main
  else
    log "No clone found at $CLONE_DIR, cloning from $REPO_URL..."
    mkdir -p "$(dirname "$CLONE_DIR")"
    git clone "$REPO_URL" "$CLONE_DIR"
  fi
  
  # Initialize submodules if they exist
  if [ -f "$CLONE_DIR/.gitmodules" ]; then
    log "Initializing submodules..."
    git -C "$CLONE_DIR" submodule update --init --recursive || log "Warning: Submodule initialization failed (may need manual setup)"
  fi
}

sync_gold_data() {
  if ! command -v aws >/dev/null 2>&1; then
    log "aws CLI not found on PATH; skipping S3 data sync."
    return 0
  fi

  if [ ! -d "$NVME_ROOT" ]; then
    log "NVME_ROOT $NVME_ROOT does not exist. Did you mount the NVMe volume?"
    return 1
  fi

  if ! mountpoint -q "$NVME_ROOT" 2>/dev/null; then
    log "Warning: $NVME_ROOT is not a mount point. Skipping S3 sync."
    return 1
  fi

  GOLD_DIR="${NVME_ROOT}/gold"
  
  # Create gold directory structure
  log "Creating gold directory structure..."
  mkdir -p "${GOLD_DIR}"/{cohorts,medical,pharmacy}

  log "=========================================="
  log "Syncing Gold Data from S3 to NVMe"
  log "=========================================="
  echo ""

  # Function to sync with progress
  sync_with_progress() {
    local source=$1
    local dest=$2
    local name=$3
    
    log "Syncing ${name}..."
    log "  Source: ${source}"
    log "  Dest: ${dest}"
    
    if aws s3 sync "${source}" "${dest}" --no-progress 2>&1 | tee "/tmp/sync_${name}.log"; then
      log "✓ ${name} sync completed"
      
      # Show summary
      local files_synced=$(grep -c "s3://" "/tmp/sync_${name}.log" 2>/dev/null || echo "0")
      local size_info=$(du -sh "${dest}" 2>/dev/null | awk '{print $1}' || echo "unknown")
      log "  Files synced: ${files_synced}"
      log "  Size: ${size_info}"
    else
      log "⚠ ${name} sync had warnings (check logs)"
    fi
    echo ""
  }

  # 1. Sync Gold Cohorts
  log "[1/3] Syncing Gold Cohorts"
  sync_with_progress \
    "${S3_GOLD_PREFIX}/cohorts/" \
    "${GOLD_DIR}/cohorts/" \
    "Gold Cohorts"

  # 2. Sync Gold Medical
  log "[2/3] Syncing Gold Medical"
  sync_with_progress \
    "${S3_GOLD_PREFIX}/medical/" \
    "${GOLD_DIR}/medical/" \
    "Gold Medical"

  # 3. Sync Gold Pharmacy
  log "[3/3] Syncing Gold Pharmacy"
  sync_with_progress \
    "${S3_GOLD_PREFIX}/pharmacy/" \
    "${GOLD_DIR}/pharmacy/" \
    "Gold Pharmacy"

  log "=========================================="
  log "Gold Data Sync Complete!"
  log "=========================================="
  echo ""
  log "Summary:"
  log "  Cohorts: ${GOLD_DIR}/cohorts/"
  log "  Medical: ${GOLD_DIR}/medical/"
  log "  Pharmacy: ${GOLD_DIR}/pharmacy/"
  echo ""
  log "Note: Gold medical/pharmacy data is the SOURCE data used to CREATE controls."
  log "      It does not contain a 'target' column - controls are sampled from this"
  log "      data in Step 4a (create_model_data.py)."
}

setup_paths_and_env() {
  log "Ensuring convenience symlink at \$HOME/pgx-analysis ..."
  ln -sfn "$CLONE_DIR" "$HOME/pgx-analysis" 2>/dev/null || true

  log "Writing env file to $ENV_FILE ..."
  cat > "$ENV_FILE" <<EOF
# PGx project environment
export PGX_REPO_ROOT="$CLONE_DIR"
export PGX_DATA_ROOT="$NVME_ROOT"

# Use fast NVMe for local staging and DuckDB temp
export PGX_USE_LOCAL_STAGING=1
export PGX_LOCAL_STAGING_DIR="/mnt/nvme/pgx_staging"

# Auto-detected system characteristics
export PGX_OS_TYPE="${PGX_OS_TYPE:-unknown}"
export PGX_CPU_CORES="${PGX_CPU_CORES:-1}"
export PGX_TOTAL_RAM_GB="${PGX_TOTAL_RAM_GB:-16}"

# Recommended defaults for heavy jobs
export PGX_WORKERS_MEDICAL="${PGX_WORKERS_MEDICAL:-12}"
export PGX_DUCKDB_MEMORY_LIMIT="${PGX_DUCKDB_MEMORY_LIMIT:-2GB}"
export PGX_THREADS_PER_WORKER="${PGX_THREADS_PER_WORKER:-1}"

# Recommended defaults for feature engineering / model training
export PGX_SKLEARN_N_JOBS="${PGX_SKLEARN_N_JOBS:-2}"
export PGX_XGB_CPU_NTHREAD="${PGX_XGB_CPU_NTHREAD:-2}"
export PGX_MC_CV_WORKERS="${PGX_MC_CV_WORKERS:-2}"

# MC-CV configuration
export PGX_MC_CV_N_SPLITS="${PGX_MC_CV_N_SPLITS:-50}"
export PGX_MC_CV_N_RUNS="${PGX_MC_CV_N_RUNS:-3}"

# Ensure Python can import helper modules
export PYTHONPATH="\$PGX_REPO_ROOT:\$PYTHONPATH"
EOF

  # Add a one-time source hook to .bashrc (idempotent)
  if [ -f "$HOME/.bashrc" ] && ! grep -q 'source ~/.pgx_env' "$HOME/.bashrc" 2>/dev/null; then
    log "Adding 'source ~/.pgx_env' to ~/.bashrc ..."
    printf '\n# PGx project env\n[ -f "$HOME/.pgx_env" ] && source "$HOME/.pgx_env"\n' >> "$HOME/.bashrc"
  fi

  log "Environment configured. New shells will source $ENV_FILE automatically."
}

##############################################
# Main
##############################################

log "Starting PGx repo sync to NVMe..."

detect_os_and_resources
mount_nvme
ensure_clone
sync_gold_data
setup_paths_and_env

log ""
log "=========================================="
log "Setup Complete!"
log "=========================================="
log "Repo root: $CLONE_DIR"
log "Symlink  : $HOME/pgx-analysis"
log "Data NVMe root: $NVME_ROOT"
log "Gold data: $NVME_ROOT/gold/"
log "Env file: $ENV_FILE (auto-sourced in new shells)"
log ""
log "Next steps:"
log "  1. Source the environment: source $ENV_FILE"
log "  2. Navigate to repo: cd $CLONE_DIR"
log "  3. Run workflows: ./utility_scripts/run_cohort_workflow.sh <cohort> <age_band>"
