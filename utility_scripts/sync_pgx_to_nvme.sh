#!/usr/bin/env bash
set -euo pipefail

##############################################
# Configuration
##############################################

# 1) GitHub repo URL for this project
#    Use HTTPS or SSH depending on your EC2 auth setup.
REPO_URL="https://github.com/Jerome3590/pgx-analysis.git"
# REPO_URL="git@github.com:Jerome3590/pgx-analysis.git"  # Alternative: SSH URL

# 2) Where to keep the git clone on EC2 (code stays on $HOME)
CLONE_DIR="$HOME/pgx-analysis"

# 3) Where the fast NVMe volume is mounted (for data, temp, staging).
#    Adjust if your device is mounted differently (e.g. /mnt/nvme0n1).
NVME_ROOT="/mnt/nvme"

# 4) Optional: where to write a small env file to source in your shell
ENV_FILE="$HOME/.pgx_env"

# 5) Optional: S3 bucket and prefixes to sync data locally (to NVMe).
#    Adjust these to match your environment, or set S3_BUCKET to empty to skip.
S3_BUCKET="s3://pgxdatalake"
# Gold-tier medical/pharmacy and related aggregates
S3_SYNC_GOLD_PREFIX="gold"
# Cohort-level exports
S3_SYNC_COHORT_PREFIX="gold/cohorts"
# Model-ready datasets (current location)
S3_SYNC_MODEL_PREFIX="gold/cohorts_model_data"

##############################################
# Functions
##############################################

log() {
  # Simple logger; avoids non-ASCII characters for safety.
  printf '[pgx-sync] %s\n' "$*"
}

detect_os_and_resources() {
  # Detect OS, cores, and RAM to derive sane defaults.
  OS_TYPE="$(uname -s || echo unknown)"
  CORES="$(python3 - <<'EOF'
import os
print(os.cpu_count() or 1)
EOF
)"

  # Default fallbacks
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
    # Fallback using Python (works on macOS/Windows if needed)
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

  # Derive default worker and memory settings if not already set.
  # These are intentionally conservative and can be overridden by the user.
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
    # Aim for ~3GB on large EC2, smaller on dev machines.
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

  # Feature engineering / model training parallelism defaults.
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

  # MC-CV configuration: 50 splits and 3 runs on EC2 (>=256GB RAM), 200 splits and 1 run on smaller systems
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

mount_nvme() {
  # Check if already mounted
  if mountpoint -q "$NVME_ROOT" 2>/dev/null; then
    log "NVMe volume already mounted at $NVME_ROOT"
    return 0
  fi

  # Only proceed on Linux
  if [ "$OS_TYPE" != "Linux" ]; then
    log "Skipping NVMe mount (non-Linux system)"
    return 0
  fi

  # Check if we have root/sudo privileges
  if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    log "Warning: Cannot mount NVMe (need root/sudo). Please mount manually:"
    log "  sudo mkdir -p $NVME_ROOT"
    log "  sudo mount /dev/nvme1n1 $NVME_ROOT  # or appropriate device"
    return 0
  fi

  # Detect NVMe devices (typically nvme1n1, nvme2n1, etc. on EC2)
  # Skip nvme0n1 as it's usually the root EBS volume
  NVME_DEVICE=""
  for dev in /dev/nvme*n1; do
    # Skip if device doesn't exist (glob didn't match)
    [ -e "$dev" ] || continue
    
    # Skip if device is already mounted
    if mount | grep -q "^$dev"; then
      log "Skipping $dev (already mounted)"
      continue
    fi
    
    # Found an unmounted NVMe device
    NVME_DEVICE="$dev"
    log "Found available NVMe device: $NVME_DEVICE"
    break
  done

  if [ -z "$NVME_DEVICE" ]; then
    log "No available NVMe device found. Skipping mount."
    log "If you have an NVMe device, mount it manually:"
    log "  sudo mkdir -p $NVME_ROOT"
    log "  sudo mount /dev/nvme1n1 $NVME_ROOT"
    return 0
  fi

  # Create mount point
  if [ "$EUID" -eq 0 ]; then
    mkdir -p "$NVME_ROOT"
  else
    sudo mkdir -p "$NVME_ROOT"
  fi

  # Check if device has a filesystem
  if ! blkid "$NVME_DEVICE" >/dev/null 2>&1; then
    log "NVMe device $NVME_DEVICE appears unformatted. Formatting with ext4..."
    if [ "$EUID" -eq 0 ]; then
      mkfs.ext4 -F "$NVME_DEVICE"
    else
      sudo mkfs.ext4 -F "$NVME_DEVICE"
    fi
  fi

  # Mount the device
  log "Mounting $NVME_DEVICE to $NVME_ROOT..."
  if [ "$EUID" -eq 0 ]; then
    mount "$NVME_DEVICE" "$NVME_ROOT"
  else
    sudo mount "$NVME_DEVICE" "$NVME_ROOT"
  fi

  # Set permissions (make it accessible to current user)
  if [ "$EUID" -eq 0 ]; then
    chmod 755 "$NVME_ROOT"
    chown "$SUDO_USER:${SUDO_USER:-$USER}" "$NVME_ROOT" 2>/dev/null || true
  else
    sudo chmod 755 "$NVME_ROOT"
    sudo chown "$USER:$USER" "$NVME_ROOT" 2>/dev/null || true
  fi

  # Add to /etc/fstab for persistence (if not already there)
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

  log "NVMe volume mounted successfully at $NVME_ROOT"
}

sync_s3_data() {
  # Optionally sync S3 gold/cohort/model data down to NVMe for faster local access.
  if [ -z "${S3_BUCKET:-}" ]; then
    log "S3_BUCKET is empty; skipping S3 data sync."
    return 0
  fi

  if ! command -v aws >/dev/null 2>&1; then
    log "aws CLI not found on PATH; skipping S3 data sync."
    return 0
  fi

  if [ ! -d "$NVME_ROOT" ]; then
    log "NVME_ROOT $NVME_ROOT does not exist. Did you mount the NVMe volume?"
    return 0
  fi

  local data_root="$NVME_ROOT"
  mkdir -p "$data_root"

  if [ -n "${S3_SYNC_GOLD_PREFIX:-}" ]; then
    log "Syncing gold-tier data from $S3_BUCKET/$S3_SYNC_GOLD_PREFIX to $data_root/gold ..."
    mkdir -p "$data_root/gold"
    aws s3 sync \
      "$S3_BUCKET/$S3_SYNC_GOLD_PREFIX/" \
      "$data_root/gold/" \
      --only-show-errors
  fi

  if [ -n "${S3_SYNC_COHORT_PREFIX:-}" ]; then
    log "Syncing cohort data from $S3_BUCKET/$S3_SYNC_COHORT_PREFIX to $data_root/cohorts ..."
    mkdir -p "$data_root/cohorts"
    aws s3 sync \
      "$S3_BUCKET/$S3_SYNC_COHORT_PREFIX/" \
      "$data_root/cohorts/" \
      --only-show-errors
  fi

  if [ -n "${S3_SYNC_MODEL_PREFIX:-}" ]; then
    log "Syncing model data from $S3_BUCKET/$S3_SYNC_MODEL_PREFIX to $data_root/model_data ..."
    mkdir -p "$data_root/model_data"
    aws s3 sync \
      "$S3_BUCKET/$S3_SYNC_MODEL_PREFIX/" \
      "$data_root/model_data/" \
      --only-show-errors
  fi

  log "S3 data sync complete (if prefixes existed)."
}

setup_paths_and_env() {
  log "Ensuring convenience symlink at \$HOME/pgx-analysis ..."
  ln -sfn "$CLONE_DIR" "$HOME/pgx-analysis"

  log "Writing env file to $ENV_FILE ..."
  cat > "$ENV_FILE" <<EOF
# PGx project environment
export PGX_REPO_ROOT="$CLONE_DIR"
export PGX_DATA_ROOT="$NVME_ROOT"

# Use fast NVMe for local staging and DuckDB temp
export PGX_USE_LOCAL_STAGING=1
export PGX_LOCAL_STAGING_DIR="/mnt/nvme/pgx_staging"

# Auto-detected system characteristics (set by sync_pgx_to_nvme.sh)
export PGX_OS_TYPE="${PGX_OS_TYPE:-unknown}"
export PGX_CPU_CORES="${PGX_CPU_CORES:-1}"
export PGX_TOTAL_RAM_GB="${PGX_TOTAL_RAM_GB:-16}"

# Recommended defaults for heavy jobs (can be overridden in shell)
export PGX_WORKERS_MEDICAL="${PGX_WORKERS_MEDICAL:-12}"
export PGX_DUCKDB_MEMORY_LIMIT="${PGX_DUCKDB_MEMORY_LIMIT:-2GB}"
export PGX_THREADS_PER_WORKER="${PGX_THREADS_PER_WORKER:-1}"

# Recommended defaults for feature engineering / model training
export PGX_SKLEARN_N_JOBS="${PGX_SKLEARN_N_JOBS:-2}"
export PGX_XGB_CPU_NTHREAD="${PGX_XGB_CPU_NTHREAD:-2}"
export PGX_MC_CV_WORKERS="${PGX_MC_CV_WORKERS:-2}"

# MC-CV configuration (50 splits, 3 runs on EC2; 200 splits, 1 run on Windows)
export PGX_MC_CV_N_SPLITS="${PGX_MC_CV_N_SPLITS:-50}"
export PGX_MC_CV_N_RUNS="${PGX_MC_CV_N_RUNS:-3}"

# Ensure Python can import helper modules (py_helpers, etc.)
export PYTHONPATH="\$PGX_REPO_ROOT:\$PYTHONPATH"

# Optionally default working directory for interactive shells
# cd "\$PGX_REPO_ROOT" 2>/dev/null || true
EOF

  # Add a one-time source hook to .bashrc (idempotent).
  if ! grep -q 'source ~/.pgx_env' "$HOME/.bashrc" 2>/dev/null; then
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
sync_s3_data
setup_paths_and_env

log "Done. Use this path for scripts and data:"
log "  Repo root: $CLONE_DIR"
log "  Symlink  : $HOME/pgx-analysis"
log "  Data NVMe root: $NVME_ROOT"
log "  Env file: $ENV_FILE (auto-sourced in new shells)"

