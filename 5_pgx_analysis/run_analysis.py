#!/usr/bin/env python3
"""
Module-style orchestration script for PGx feature engineering.

This script runs the complete PGx workflow:
1. Map drugs to genes (cohort-level, reused across age bands)
2. Create PGx features (using global/cohort-level allele frequency cache)
3. Add PGx features to model data

Usage:
    python 5_pgx_analysis/run_analysis.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from py_helpers.fe_monitor import (  # noqa: E402
    detect_runtime_environment,
    function_block,
    module_block,
    step_block,
    mirror_log_to_s3,
)


def _get_logger(cohort_name: str, age_band: str) -> tuple[logging.Logger, Path]:
    """Create a module-level logger with both console and file handlers."""
    logs_dir = PROJECT_ROOT / "logs" / "5_pgx_analysis"
    logs_dir.mkdir(parents=True, exist_ok=True)

    age_band_fname = age_band.replace("-", "_")
    log_path = logs_dir / f"pgx_{cohort_name}_{age_band_fname}.log"

    logger = logging.getLogger(f"pgx.{cohort_name}.{age_band_fname}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
        )

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger, log_path


def _download_from_s3_if_exists(s3_path: str, local_path: Path, logger: logging.Logger) -> bool:
    """Download file from S3 if it exists, return True if downloaded or already exists locally."""
    if local_path.exists():
        return True
    
    try:
        from py_helpers.checkpoint_utils import check_s3_output_exists
        import boto3
        
        if not check_s3_output_exists(s3_path):
            return False
        
        # Parse S3 path
        if s3_path.startswith("s3://"):
            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            logger.warning(f"Invalid S3 path format: {s3_path}")
            return False
        
        # Download file
        s3_client = boto3.client("s3")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        logger.info(f"Downloaded from S3: {s3_path} -> {local_path}")
        return True
    except Exception as e:
        logger.warning(f"Could not download {s3_path}: {e}")
        return False


def map_drugs_to_genes(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> None:
    """
    Step 1: Map cohort drugs to genes (if mappings do not already exist).

    NOTE: Drug→gene mappings are **not age-band specific**. We therefore create
    ONE mapping file per cohort and reuse it across all age bands.
    
    This function is idempotent: it checks local files first, then S3 (global cache,
    cohort-level, legacy age-band), and only generates if none exist.
    """
    with step_block("5_pgx_analysis", "map_drugs_to_genes", logger=logger):
        script_path = PROJECT_ROOT / "5_pgx_analysis" / "map_drugs_to_genes.py"
        cohort_out_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs" / cohort_name
        cohort_out_dir.mkdir(parents=True, exist_ok=True)
        global_out_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "global"
        global_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Cohort-global mapping file (shared across all age bands)
        mapping_file = cohort_out_dir / f"{cohort_name}_drug_gene_mappings.csv"
        global_mapping_file = global_out_dir / "pgx_drug_gene_mappings_global.csv"

        # Check local files first
        if mapping_file.exists():
            logger.info(
                "Cohort-level drug-gene mapping already exists at %s; skipping",
                mapping_file,
            )
            return
        
        if global_mapping_file.exists():
            logger.info(
                "Global drug-gene mapping cache exists at %s; using it",
                global_mapping_file,
            )
            # Copy global to cohort-level for consistency
            shutil.copy2(global_mapping_file, mapping_file)
            return
        
        # Check S3 (prefer global cache, then cohort-level, then legacy age-band)
        s3_paths_to_try = [
            "s3://pgxdatalake/gold/pgx_features/global/pgx_drug_gene_mappings_global.csv",
            f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{cohort_name}_drug_gene_mappings.csv",
            f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/{cohort_name}_drug_gene_mappings.csv",
        ]
        
        downloaded = False
        for s3_path in s3_paths_to_try:
            if _download_from_s3_if_exists(s3_path, mapping_file, logger):
                downloaded = True
                logger.info(f"Using drug-gene mappings from S3: {s3_path}")
                break
        
        if downloaded:
            return

        logger.info("Mapping drugs to genes for %s / %s", cohort_name, age_band)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort",
                    cohort_name,
                    "--age_band",
                    age_band,
                    "--output",
                    str(mapping_file),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Drug-gene mapping completed")
            if result.stdout:
                logger.info("Mapping stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Mapping stderr:\n%s", result.stderr)

            # Update global PGx drug-gene mapping cache (shared across cohorts).
            try:
                global_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "global"
                global_dir.mkdir(parents=True, exist_ok=True)
                global_mapping_file = global_dir / "pgx_drug_gene_mappings_global.csv"

                if mapping_file.exists():
                    cohort_df = pd.read_csv(mapping_file)
                    if global_mapping_file.exists():
                        global_df = pd.read_csv(global_mapping_file)
                        combined = pd.concat([global_df, cohort_df], ignore_index=True)
                        # Deduplicate by drug_name + gene to keep table small and stable
                        if {"drug_name", "gene"}.issubset(combined.columns):
                            combined = combined.drop_duplicates(subset=["drug_name", "gene"])
                        else:
                            combined = combined.drop_duplicates()
                        combined.to_csv(global_mapping_file, index=False)
                    else:
                        shutil.copy2(mapping_file, global_mapping_file)
                    logger.info(
                        "Updated global PGx drug-gene mapping cache at %s",
                        global_mapping_file,
                    )
            except Exception as exc:  # pragma: no cover - best-effort
                logger.warning("Could not update global PGx drug-gene cache: %s", exc)
        except subprocess.CalledProcessError as exc:
            logger.error("Drug mapping failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Drug mapping failed with exception: %s", exc)
            raise


def create_pgx_features_step(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 3: Create patient-level PGx features."""
    with step_block("5_pgx_analysis", "create_pgx_features", logger=logger):
        logger.info("Creating PGx features for %s / %s", cohort_name, age_band)
        script_path = (
            PROJECT_ROOT
            / "5_pgx_analysis"
            / "create_pgx_features_patient_level.py"
        )

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort",
                    cohort_name,
                    "--age_band",
                    age_band,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("PGx features created")
            if result.stdout:
                logger.info("Create stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Create stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("PGx feature creation failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("PGx feature creation failed with exception: %s", exc)
            return False


def add_pgx_features_to_model_data(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 4: Merge PGx features into final PGx feature table."""
    with step_block("5_pgx_analysis", "add_pgx_features_to_model_data", logger=logger):
        logger.info(
            "Adding PGx features to model data for %s / %s",
            cohort_name,
            age_band,
        )
        script_path = (
            PROJECT_ROOT
            / "5_pgx_analysis"
            / "add_pgx_features_to_model_data.py"
        )

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort-name",
                    cohort_name,
                    "--age-band",
                    age_band,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("PGx features added to model data")
            if result.stdout:
                logger.info("Merge stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Merge stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("PGx feature merge failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("PGx feature merge failed with exception: %s", exc)
            return False


def run_pgx_analysis(
    cohort_name: str,
    age_band: str,
    skip_drug_mapping: bool = False,
    skip_feature_engineering: bool = False,
) -> bool:
    """
    Run complete PGx analysis workflow as a module-style function.

    This function is idempotent with respect to its outputs: rerunning will
    regenerate the same CSVs, and downstream code reads the most recent files.
    """
    logger, log_path = _get_logger(cohort_name, age_band)

    env = detect_runtime_environment(PROJECT_ROOT)
    logger.info(
        "Runtime environment: os=%s logical_cores=%s ram_gb=%s fast_root=%s",
        env.os_name,
        env.logical_cores,
        env.ram_gb,
        env.fast_root,
    )

    # Check S3 for existing outputs (idempotency)
    try:
        from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists

        age_band_fname = age_band.replace("-", "_")
        s3_output_paths = [
            f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/pgx_added_features_{cohort_name}_{age_band_fname}.csv",
            f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/{cohort_name}_drug_gene_mappings.csv",
            f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/{cohort_name}_allele_frequencies.csv",
        ]

        if check_step_outputs_exist(s3_output_paths, logger) or check_step_checkpoint_exists("5_pgx_analysis", cohort_name, age_band, logger):
            logger.info(f"Step 5 outputs already exist in S3 for {cohort_name}/{age_band}; skipping.")
            return True
    except ImportError:
        pass  # Fallback to local-only if checkpoint_utils not available

    with function_block("5_pgx_analysis", "run_pgx_analysis", logger=logger):
        logger.info("Starting PGx analysis for %s / %s", cohort_name, age_band)

        if not skip_drug_mapping:
            try:
                # Drug-to-gene mappings are cohort-level and reused across age bands.
                map_drugs_to_genes(cohort_name, age_band, logger=logger)
            except Exception:
                mirror_log_to_s3("5_pgx_analysis", cohort_name, age_band, log_path, logger)
                return False
        else:
            logger.info(
                "Skipping drug mapping (using existing cohort/global mappings if present)"
            )

        if not skip_feature_engineering:
            if not create_pgx_features_step(cohort_name, age_band, logger=logger):
                logger.error("PGx feature creation failed; aborting module")
                mirror_log_to_s3("5_pgx_analysis", cohort_name, age_band, log_path, logger)
                return False

            if not add_pgx_features_to_model_data(cohort_name, age_band, logger=logger):
                logger.error("PGx feature merge failed; aborting module")
                mirror_log_to_s3("5_pgx_analysis", cohort_name, age_band, log_path, logger)
                return False
        else:
            logger.info("Skipping PGx feature engineering (using existing features)")

        logger.info("PGx analysis completed for %s / %s", cohort_name, age_band)

    # Upload outputs to S3 and save checkpoint
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint

        age_band_fname = age_band.replace("-", "_")
        s3_outputs = []

        # Upload PGx features
        pgx_features_path = (
            PROJECT_ROOT
            / "5_feature_engineering"
            / "feature_engineering_outputs"
            / "5_pgx_analysis"
            / cohort_name
            / age_band
            / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
        )
        if pgx_features_path.exists():
            s3_pgx_path = f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/pgx_added_features_{cohort_name}_{age_band_fname}.csv"
            if upload_file_to_s3(pgx_features_path, s3_pgx_path, logger):
                s3_outputs.append(s3_pgx_path)

        # Upload drug-gene mappings
        cohort_out_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs" / cohort_name
        mappings_path = cohort_out_dir / f"{cohort_name}_drug_gene_mappings.csv"
        if mappings_path.exists():
            s3_mappings_path = f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/{cohort_name}_drug_gene_mappings.csv"
            if upload_file_to_s3(mappings_path, s3_mappings_path, logger):
                s3_outputs.append(s3_mappings_path)

        # Upload allele frequencies
        freq_path = cohort_out_dir / f"{cohort_name}_allele_frequencies.csv"
        if freq_path.exists():
            s3_freq_path = f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/{cohort_name}_allele_frequencies.csv"
            if upload_file_to_s3(freq_path, s3_freq_path, logger):
                s3_outputs.append(s3_freq_path)

        # Save checkpoint
        save_step_checkpoint(
            step_name="5_pgx_analysis",
            cohort=cohort_name,
            age_band=age_band,
            metadata={},
            output_paths=s3_outputs,
            logger=logger,
        )
    except ImportError:
        pass  # Checkpoint saving is optional

    mirror_log_to_s3("5_pgx_analysis", cohort_name, age_band, log_path, logger)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete PGx analysis workflow"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        required=True,
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 0-12)",
    )
    parser.add_argument(
        "--skip-drug-mapping",
        action="store_true",
        help="Skip drug-to-gene mapping step (assumes mappings already exist)",
    )
    parser.add_argument(
        "--skip-feature-engineering",
        action="store_true",
        help="Skip PGx feature engineering steps",
    )

    args = parser.parse_args()

    with module_block("5_pgx_analysis"):
        success = run_pgx_analysis(
            cohort_name=args.cohort_name,
            age_band=args.age_band,
            skip_drug_mapping=args.skip_drug_mapping,
            skip_feature_engineering=args.skip_feature_engineering,
        )

    sys.exit(0 if success else 1)
