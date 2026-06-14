#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _top_features(df: pd.DataFrame, n: int) -> set[str]:
    if "mean_abs_shap" in df.columns:
        df = df.sort_values("mean_abs_shap", ascending=False)
    return set(df.head(n)["feature"].astype(str))


def _read_ffa_features(path: Path) -> set[str]:
    import duckdb

    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT feature FROM read_parquet('{str(path)}')").df()
    finally:
        con.close()
    return set(df["feature"].astype(str)) if "feature" in df.columns else set()


def audit_cohort_level(project_root: Path) -> pd.DataFrame:
    shap_root = project_root / "7_shap_analysis" / "outputs"
    ffa_root = project_root / "8_ffa_analysis" / "outputs"
    rows: list[dict] = []

    for xgb_path in sorted(shap_root.rglob("*_shap_global_importance_xgboost.csv")):
        cb_path = xgb_path.with_name(xgb_path.name.replace("_xgboost.csv", "_catboost.csv"))
        if not cb_path.exists():
            continue
        cohort, age = xgb_path.relative_to(shap_root).parts[:2]
        xgb = pd.read_csv(xgb_path)
        cat = pd.read_csv(cb_path)
        xgb_features = set(xgb["feature"].astype(str))
        cat_features = set(cat["feature"].astype(str))
        overlap = xgb_features & cat_features
        xgb_top20, cat_top20 = _top_features(xgb, 20), _top_features(cat, 20)
        xgb_top50, cat_top50 = _top_features(xgb, 50), _top_features(cat, 50)
        xgb_top100, cat_top100 = _top_features(xgb, 100), _top_features(cat, 100)

        ffa_candidates = [
            ffa_root / cohort / age / "xgboost" / "feature_importance_axp.parquet",
            ffa_root / cohort / age.replace("_", "-") / "xgboost" / "feature_importance_axp.parquet",
        ]
        ffa_path = next((p for p in ffa_candidates if p.exists()), None)
        ffa_features = _read_ffa_features(ffa_path) if ffa_path else set()

        rows.append(
            {
                "cohort": cohort,
                "age": age,
                "xgb_n": len(xgb_features),
                "cat_n": len(cat_features),
                "overlap_n": len(overlap),
                "overlap_xgb_pct": len(overlap) / len(xgb_features) if xgb_features else 0.0,
                "overlap_cat_pct": len(overlap) / len(cat_features) if cat_features else 0.0,
                "top20_jaccard": len(xgb_top20 & cat_top20) / len(xgb_top20 | cat_top20) if xgb_top20 | cat_top20 else 0.0,
                "top50_jaccard": len(xgb_top50 & cat_top50) / len(xgb_top50 | cat_top50) if xgb_top50 | cat_top50 else 0.0,
                "top100_jaccard": len(xgb_top100 & cat_top100) / len(xgb_top100 | cat_top100) if xgb_top100 | cat_top100 else 0.0,
                "ffa_exists": bool(ffa_path),
                "ffa_n": len(ffa_features),
                "ffa_cat_overlap_pct": len(ffa_features & cat_features) / len(ffa_features) if ffa_features else None,
                "cat_top50_in_ffa_pct": len(cat_top50 & ffa_features) / len(cat_top50) if cat_top50 else None,
                "dual_top50_in_ffa_pct": len((xgb_top50 & cat_top50) & ffa_features) / len(xgb_top50 & cat_top50) if (xgb_top50 & cat_top50) else None,
            }
        )
    return pd.DataFrame(rows)


def audit_per_bin(project_root: Path) -> pd.DataFrame:
    shap_root = project_root / "7_shap_analysis" / "outputs"
    ffa_root = project_root / "8_ffa_analysis" / "outputs"
    rows: list[dict] = []

    for ffa_path in sorted(ffa_root.rglob("bin_models/*/xgboost/feature_importance_axp.parquet")):
        rel = ffa_path.relative_to(ffa_root).parts
        cohort, age, bin_name = rel[0], rel[1], rel[3]
        age_fname = age.replace("-", "_")
        xgb_path = shap_root / cohort / age / f"{cohort}_{age_fname}_shap_global_importance_xgboost.csv"
        cb_path = shap_root / cohort / age / f"{cohort}_{age_fname}_shap_global_importance_catboost.csv"
        if not xgb_path.exists() or not cb_path.exists():
            rows.append({"cohort": cohort, "age": age, "bin": bin_name, "missing_shap": True})
            continue

        xgb = pd.read_csv(xgb_path)
        cat = pd.read_csv(cb_path)
        xgb_features = set(xgb["feature"].astype(str))
        cat_features = set(cat["feature"].astype(str))
        xgb_top50, cat_top50 = _top_features(xgb, 50), _top_features(cat, 50)
        dual_top50 = xgb_top50 & cat_top50
        ffa_features = _read_ffa_features(ffa_path)

        rows.append(
            {
                "cohort": cohort,
                "age": age,
                "bin": bin_name,
                "missing_shap": False,
                "ffa_n": len(ffa_features),
                "ffa_in_cat_pct": len(ffa_features & cat_features) / len(ffa_features) if ffa_features else 0.0,
                "ffa_in_xgb_pct": len(ffa_features & xgb_features) / len(ffa_features) if ffa_features else 0.0,
                "cat_top50_in_ffa_pct": len(cat_top50 & ffa_features) / len(cat_top50) if cat_top50 else 0.0,
                "xgb_top50_in_ffa_pct": len(xgb_top50 & ffa_features) / len(xgb_top50) if xgb_top50 else 0.0,
                "dual_top50_in_ffa_pct": len(dual_top50 & ffa_features) / len(dual_top50) if dual_top50 else None,
                "dual_top50_n": len(dual_top50),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit existing XGBoost/CatBoost SHAP overlap and FFA feature coverage.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cohort-output", type=Path, default=Path("audit_shap_ffa_existing_coverage.csv"))
    parser.add_argument("--per-bin-output", type=Path, default=Path("audit_per_bin_ffa_vs_shap_coverage.csv"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cohort_df = audit_cohort_level(project_root)
    per_bin_df = audit_per_bin(project_root)

    cohort_out = args.cohort_output if args.cohort_output.is_absolute() else project_root / args.cohort_output
    per_bin_out = args.per_bin_output if args.per_bin_output.is_absolute() else project_root / args.per_bin_output
    cohort_df.to_csv(cohort_out, index=False)
    per_bin_df.to_csv(per_bin_out, index=False)

    print(f"Saved {cohort_out}")
    print(f"Saved {per_bin_out}")
    if not per_bin_df.empty:
        for col in ["ffa_in_cat_pct", "ffa_in_xgb_pct", "cat_top50_in_ffa_pct", "xgb_top50_in_ffa_pct", "dual_top50_in_ffa_pct"]:
            print(f"{col}: mean={per_bin_df[col].mean():.4f} min={per_bin_df[col].min():.4f}")


if __name__ == "__main__":
    main()
