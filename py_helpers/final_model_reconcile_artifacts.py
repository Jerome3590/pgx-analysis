from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

from py_helpers.constants import REQUIRED_COHORTS, age_band_to_fname
from py_helpers.event_density_utils import DENSITY_BINS, resolve_step6_cohort_age_dir
from py_helpers.final_model_s3_upload import upload_step6_outputs_to_s3

MODEL_FILES = ("xgboost.joblib", "catboost.joblib", "xgboost_model.ubj", "catboost_model.cbm")


def _s3_bucket() -> str:
    return os.environ.get("PGX_S3_BUCKET", "pgxdatalake")


def _iter_scope(cohorts: Optional[Iterable[str]], age_bands: Optional[Iterable[str]]):
    cohort_filter = set(cohorts) if cohorts else None
    age_filter = set(age_bands) if age_bands else None
    for cohort, bands in REQUIRED_COHORTS.items():
        if cohort_filter is not None and cohort not in cohort_filter:
            continue
        for age_band in bands:
            if age_filter is not None and age_band not in age_filter:
                continue
            yield cohort, age_band


def _copy_if_needed(src: Path, dest: Path, *, apply: bool, force: bool, logger: logging.Logger) -> bool:
    if not src.is_file():
        return False
    if dest.exists() and not force:
        return False
    logger.info("%s %s -> %s", "COPY" if apply else "WOULD COPY", src, dest)
    if apply:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return True


def _remove_if_requested(path: Path, *, apply: bool, logger: logging.Logger) -> bool:
    if not path.is_file():
        return False
    logger.info("%s stale local file %s", "DELETE" if apply else "WOULD DELETE", path)
    if apply:
        path.unlink()
    return True


def _read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_source_marker(
    bin_root: Path,
    *,
    cohort: str,
    age_band: str,
    bin_name: str,
    model_source: str,
    fallback_used: Optional[bool],
    fallback_reason: Optional[str],
    apply: bool,
    logger: logging.Logger,
) -> bool:
    source_path = bin_root / "PER_BIN_TRAINING_SOURCE.json"
    existing = _read_json(source_path) if source_path.exists() else {}
    payload = {
        "cohort": cohort,
        "age_band": age_band,
        "n_event_bin": bin_name,
        "model_source": existing.get("model_source", model_source),
        "fallback_used": existing.get("fallback_used", fallback_used),
        "fallback_reason": existing.get("fallback_reason", fallback_reason),
        "train_years": existing.get("train_years", "2016-2018"),
        "holdout_year": existing.get("holdout_year", 2019),
        "n_train_total": existing.get("n_train_total"),
        "n_train_cases": existing.get("n_train_cases"),
        "n_train_controls": existing.get("n_train_controls"),
        "reconciled_without_retraining": True,
    }
    if existing == payload:
        return False
    logger.info("%s source marker %s", "WRITE" if apply else "WOULD WRITE", source_path)
    if apply:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with open(source_path, "w") as f:
            json.dump(payload, f, indent=2)
    return True


def _patch_holdout_source_fields(bin_root: Path, *, apply: bool, logger: logging.Logger) -> bool:
    source_path = bin_root / "PER_BIN_TRAINING_SOURCE.json"
    source = _read_json(source_path)
    if not source:
        return False
    changed = False
    for holdout_path in bin_root.glob("*_holdout_2019_metrics.json"):
        payload = _read_json(holdout_path)
        if not payload:
            continue
        for key in (
            "model_source",
            "fallback_used",
            "fallback_reason",
            "n_train_total",
            "n_train_cases",
            "n_train_controls",
        ):
            value = source.get(key)
            if payload.get(key) != value:
                payload[key] = value
                changed = True
        if changed:
            logger.info("%s holdout source fields %s", "PATCH" if apply else "WOULD PATCH", holdout_path)
            if apply:
                with open(holdout_path, "w") as f:
                    json.dump(payload, f, indent=2)
    return changed


def reconcile_one(
    project_root: Path,
    cohort: str,
    age_band: str,
    *,
    apply: bool,
    force_copy: bool,
    delete_local_stale: bool,
    upload: bool,
    delete_s3_stale: bool,
    logger: logging.Logger,
) -> dict:
    out_base = resolve_step6_cohort_age_dir(project_root, cohort, age_band)
    abf = age_band_to_fname(age_band)
    result = {"cohort": cohort, "age_band": age_band, "local_dir": str(out_base), "actions": 0, "uploaded": 0, "s3_deleted": 0}
    if not out_base.is_dir():
        logger.warning("Missing Step 6 output directory for %s/%s: %s", cohort, age_band, out_base)
        return result

    aggregate_models = out_base / "models"
    for bin_name in DENSITY_BINS:
        bin_root = out_base / "bin_models" / bin_name
        models_dir = bin_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True) if apply else None
        legacy_present = any((bin_root / fname).is_file() for fname in MODEL_FILES)
        active_present = any((models_dir / fname).is_file() for fname in MODEL_FILES)
        inference_source = bin_root / "INFERENCE_SOURCE.txt"
        fallback_used = inference_source.exists() or (not active_present and any((aggregate_models / fname).is_file() for fname in MODEL_FILES))
        model_source = "aggregate_fallback" if fallback_used else "bin_specific"
        fallback_reason = "reconciled from aggregate or legacy per-bin artifacts without retraining" if fallback_used else None

        for fname in MODEL_FILES:
            legacy = bin_root / fname
            active = models_dir / fname
            aggregate = aggregate_models / fname
            if legacy.is_file():
                result["actions"] += int(_copy_if_needed(legacy, active, apply=apply, force=force_copy, logger=logger))
                if delete_local_stale:
                    result["actions"] += int(_remove_if_requested(legacy, apply=apply, logger=logger))
            elif fallback_used and aggregate.is_file() and not active.is_file():
                result["actions"] += int(_copy_if_needed(aggregate, active, apply=apply, force=force_copy, logger=logger))

        result["actions"] += int(
            _write_source_marker(
                bin_root,
                cohort=cohort,
                age_band=age_band,
                bin_name=bin_name,
                model_source=model_source,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                apply=apply,
                logger=logger,
            )
        )
        result["actions"] += int(_patch_holdout_source_fields(bin_root, apply=apply, logger=logger))

    if upload and apply:
        uploaded = upload_step6_outputs_to_s3(cohort, age_band, project_root, train_mode="per_bin", check_exists=False)
        result["uploaded"] = len(uploaded)
    elif upload:
        logger.info("WOULD upload reconciled Step 6 outputs for %s/%s", cohort, age_band)

    if delete_s3_stale:
        result["s3_deleted"] = _delete_stale_s3_bin_root_models(cohort, age_band, apply=apply, logger=logger)

    return result


def _delete_stale_s3_bin_root_models(cohort: str, age_band: str, *, apply: bool, logger: logging.Logger) -> int:
    import boto3

    bucket = _s3_bucket()
    s3 = boto3.client("s3")
    deleted = 0
    for bin_name in DENSITY_BINS:
        for fname in MODEL_FILES:
            key = f"gold/final_model/{cohort}/{age_band}/bin_models/{bin_name}/{fname}"
            try:
                s3.head_object(Bucket=bucket, Key=key)
            except Exception:
                continue
            logger.info("%s stale S3 object s3://%s/%s", "DELETE" if apply else "WOULD DELETE", bucket, key)
            if apply:
                s3.delete_object(Bucket=bucket, Key=key)
            deleted += 1
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Step 6 per-bin final-model artifacts without retraining.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cohort", action="append", choices=sorted(REQUIRED_COHORTS))
    parser.add_argument("--age-band", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force-copy", action="store_true")
    parser.add_argument("--delete-local-stale", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--delete-s3-stale", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("final_model_reconcile_artifacts")

    if not args.all and not args.cohort:
        raise SystemExit("Specify --all or at least one --cohort")

    project_root = args.project_root.resolve()
    total = []
    for cohort, age_band in _iter_scope(args.cohort, args.age_band):
        total.append(
            reconcile_one(
                project_root,
                cohort,
                age_band,
                apply=args.apply,
                force_copy=args.force_copy,
                delete_local_stale=args.delete_local_stale,
                upload=args.upload,
                delete_s3_stale=args.delete_s3_stale,
                logger=logger,
            )
        )

    print(json.dumps(total, indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply to make changes.")


if __name__ == "__main__":
    main()
