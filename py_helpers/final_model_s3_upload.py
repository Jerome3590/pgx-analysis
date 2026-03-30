"""
Idempotent upload of Step 6 local outputs to S3 gold/final_model.

Uses ``upload_file_to_s3`` (skip if object already exists) with **explicit**
(local_path, s3_uri) pairs matching ``6_final_model/run_final_model.py`` — no
``aws s3 sync`` of whole trees, so keys cannot drift to the wrong prefix.

CLI: ``python -m py_helpers.final_model_s3_upload --cohort X --age-band Y``
or ``--all`` to discover cohort/age pairs under ``6_final_model/outputs/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from py_helpers.checkpoint_utils import save_step_checkpoint, upload_file_to_s3
from py_helpers.constants import age_band_to_fname
from py_helpers.event_density_utils import DENSITY_BINS, resolve_step6_cohort_age_dir


def _s3_bucket() -> str:
    return os.environ.get("PGX_S3_BUCKET", "pgxdatalake")


def _outputs_scan_roots(project_root: Path) -> List[Path]:
    """Repo and optional DATA_ROOT ``6_final_model/outputs`` (deduped)."""
    roots: List[Path] = [project_root / "6_final_model" / "outputs"]
    try:
        from py_helpers.env_utils import get_data_root

        p2 = Path(get_data_root()) / "6_final_model" / "outputs"
        if p2.resolve() != roots[0].resolve():
            roots.append(p2)
    except Exception:
        pass
    return roots


def discover_cohort_age_pairs(project_root: Path) -> List[Tuple[str, str]]:
    """Unique (cohort, age_band) with at least one output directory on disk."""
    seen: set[Tuple[str, str]] = set()
    for root in _outputs_scan_roots(project_root):
        if not root.is_dir():
            continue
        for cohort_dir in sorted(root.iterdir()):
            if not cohort_dir.is_dir():
                continue
            cohort = cohort_dir.name
            for abf_dir in sorted(cohort_dir.iterdir()):
                if not abf_dir.is_dir():
                    continue
                age_band = abf_dir.name.replace("_", "-")
                seen.add((cohort, age_band))
    return sorted(seen)


def build_step6_gold_upload_pairs(
    project_root: Path,
    cohort: str,
    age_band: str,
    *,
    upload_bins: bool,
) -> List[Tuple[Path, str]]:
    """
    (local file, s3://bucket/gold/final_model/...) pairs for Step 6.

    ``upload_bins``: include ``bin_models/{bin}/…`` joblibs and native binaries
    (same layout as ``run_final_model`` idempotent repair block).
    """
    out_base = resolve_step6_cohort_age_dir(project_root, cohort, age_band)
    abf = age_band_to_fname(age_band)
    bucket = _s3_bucket()
    s3_root = f"s3://{bucket}/gold/final_model/{cohort}/{age_band}"

    pairs: List[Tuple[Path, str]] = []

    def add(local: Path, s3_uri: str) -> None:
        if local.is_file():
            pairs.append((local, s3_uri))

    add(
        out_base / f"{cohort}_{abf}_model_selection_metadata.json",
        f"{s3_root}/{cohort}_{abf}_model_selection_metadata.json",
    )
    add(
        out_base / "final_model_json" / f"{cohort}_{abf}_best_xgboost_model.json",
        f"{s3_root}/{cohort}_{abf}_best_xgboost_model.json",
    )
    add(
        out_base / "final_model_json" / f"{cohort}_{abf}_best_catboost_model.cbm",
        f"{s3_root}/{cohort}_{abf}_best_catboost_model.cbm",
    )
    add(out_base / "models" / "xgboost.joblib", f"{s3_root}/xgboost.joblib")
    add(out_base / "models" / "catboost.joblib", f"{s3_root}/catboost.joblib")
    add(
        out_base / f"{cohort}_{abf}_xgboost_feature_importance.csv",
        f"{s3_root}/{cohort}_{abf}_xgboost_feature_importance.csv",
    )
    add(
        out_base / f"{cohort}_{abf}_train_final_features_no_leakage.csv",
        f"{s3_root}/{cohort}_{abf}_train_final_features_no_leakage.csv",
    )
    # Native binaries (SHAP) — same keys as train_and_evaluate save_model_idempotent
    add(out_base / "models" / "xgboost_model.ubj", f"{s3_root}/xgboost_model.ubj")
    add(out_base / "models" / "catboost_model.cbm", f"{s3_root}/catboost_model.cbm")

    if upload_bins and (out_base / "bin_models").is_dir():
        for b in DENSITY_BINS:
            mdir = out_base / "bin_models" / b / "models"
            add(mdir / "xgboost.joblib", f"{s3_root}/bin_models/{b}/xgboost.joblib")
            add(mdir / "catboost.joblib", f"{s3_root}/bin_models/{b}/catboost.joblib")
            add(mdir / "xgboost_model.ubj", f"{s3_root}/bin_models/{b}/xgboost_model.ubj")
            add(mdir / "catboost_model.cbm", f"{s3_root}/bin_models/{b}/catboost_model.cbm")

    return pairs


def upload_step6_outputs_to_s3(
    cohort: str,
    age_band: str,
    project_root: Path,
    *,
    logger: Optional[logging.Logger] = None,
    train_mode: Optional[str] = None,
    check_exists: bool = True,
    save_checkpoint: bool = True,
) -> List[str]:
    """
    Upload Step 6 artifacts for one cohort/age_band using explicit keys.

    ``train_mode``: ``aggregate`` skips per-bin uploads; ``per_bin`` / ``both``
    include them; ``None`` uploads bins if ``bin_models/`` exists.
    """
    if train_mode is None:
        out_base = resolve_step6_cohort_age_dir(project_root, cohort, age_band)
        upload_bins = (out_base / "bin_models").is_dir()
    else:
        upload_bins = train_mode in ("per_bin", "both")

    pairs = build_step6_gold_upload_pairs(
        project_root, cohort, age_band, upload_bins=upload_bins
    )
    s3_done: List[str] = []
    for local_path, s3_uri in pairs:
        if upload_file_to_s3(local_path, s3_uri, logger=logger, check_exists=check_exists):
            s3_done.append(s3_uri)

    if save_checkpoint and s3_done:
        try:
            save_step_checkpoint(
                step_name="6_final_model",
                cohort=cohort,
                age_band=age_band,
                metadata={"n_outputs": len(s3_done), "upload": "final_model_s3_upload"},
                output_paths=s3_done,
            )
        except Exception as e:
            if logger:
                logger.warning("Could not save S3 checkpoint: %s", e)

    return s3_done


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Idempotent Step 6 upload to s3://…/gold/final_model/ (explicit keys, no sync)."
    )
    p.add_argument("--cohort", help="Cohort name (e.g. opioid_ed)")
    p.add_argument("--age-band", help="Age band with hyphens (e.g. 13-24)")
    p.add_argument(
        "--all",
        action="store_true",
        help="Upload every cohort/age_band found under 6_final_model/outputs (repo + DATA_ROOT).",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root (default: env PGX_REPO_ROOT or parent of py_helpers).",
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Do not write pgx-repository step checkpoint after upload.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Upload even when the object already exists in S3 (check_exists=False).",
    )
    p.add_argument(
        "--train-mode",
        choices=["per_bin", "aggregate", "both"],
        default=None,
        help="Override bin upload rule (default: infer from bin_models/ directory).",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    root = args.project_root or (
        Path(os.environ["PGX_REPO_ROOT"]).resolve()
        if os.environ.get("PGX_REPO_ROOT")
        else Path(__file__).resolve().parents[1]
    )
    log = logging.getLogger("final_model_s3_upload")
    if not log.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.all:
        pairs_list = discover_cohort_age_pairs(root)
        if not pairs_list:
            log.error("No cohort/age_band directories under %s", root / "6_final_model" / "outputs")
            return 1
        total = 0
        for cohort, age_band in pairs_list:
            log.info("Uploading Step 6 → S3: %s / %s", cohort, age_band)
            n = upload_step6_outputs_to_s3(
                cohort,
                age_band,
                root,
                logger=log,
                train_mode=args.train_mode,
                check_exists=not args.force,
                save_checkpoint=not args.no_checkpoint,
            )
            log.info("  → %d object(s) present or uploaded", len(n))
            total += len(n)
        log.info("Done. Total uploads/skips: %d", total)
        return 0

    if not args.cohort or not args.age_band:
        log.error("Provide --cohort and --age-band, or use --all")
        return 2

    n = upload_step6_outputs_to_s3(
        args.cohort,
        args.age_band,
        root,
        logger=log,
        train_mode=args.train_mode,
        check_exists=not args.force,
        save_checkpoint=not args.no_checkpoint,
    )
    log.info("Done. %d object(s) present or uploaded.", len(n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
