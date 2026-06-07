#!/usr/bin/env python3
"""
Regenerate explicit multi-drug combination summaries from scenario SHAP/FFA outputs.

This is a lightweight regeneration path for manuscript/dashboard review when full
FFA interaction_analysis.parquet artifacts are absent. It does not claim causal
synergy. It summarizes multi-drug medication profiles that co-occur in top
combined SHAP/FFA features and patient-level FFA rule explanations.
"""

import argparse
import ast
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

DRUG_PREFIX = "item_drug_"
FEATURE_RE = re.compile(r"(item_drug_[A-Za-z0-9_\-]+|pgx_[A-Za-z0-9_\-]+|pgx_num_drugs|pgx_num_cpic_drugs)\s*(?:[<>=!]+)")


def _drug_name(feature: str) -> str:
    return str(feature).replace(DRUG_PREFIX, "")


def _parse_listish(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return [text]


def _extract_drug_features_from_rule_text(text: str) -> set[str]:
    return {m.group(1) for m in FEATURE_RE.finditer(str(text)) if m.group(1).startswith(DRUG_PREFIX)}


def _iter_bin_dirs(scenario_root: Path) -> Iterable[Path]:
    for combined in scenario_root.rglob("combined_shap_importance.csv"):
        yield combined.parent


def summarize_bin(bin_dir: Path, scenario_root: Path, top_k: int, max_size: int) -> tuple[list[dict], list[dict]]:
    rel = bin_dir.relative_to(scenario_root).parts
    if len(rel) < 3:
        return [], []
    cohort, age_band, bin_name = rel[:3]
    combined_path = bin_dir / "combined_shap_importance.csv"
    patient_path = bin_dir / "patient_explanations.csv"

    combined = pd.read_csv(combined_path)
    if "feature" not in combined.columns:
        return [], []
    combined = combined.reset_index(drop=True)
    top = combined.head(top_k).copy()
    top["rank"] = top.index + 1
    drug_rows = top[top["feature"].astype(str).str.startswith(DRUG_PREFIX)].copy()
    if drug_rows.empty:
        return [], []

    score_by_feature = {
        str(row.feature): float(getattr(row, "combined_importance", 0.0) or 0.0)
        for row in drug_rows.itertuples(index=False)
    }
    rank_by_feature = {str(row.feature): int(row.rank) for row in drug_rows.itertuples(index=False)}
    top_drugs = sorted(score_by_feature)

    rule_support = Counter()
    patient_support = Counter()
    if patient_path.exists():
        for chunk in pd.read_csv(patient_path, chunksize=5000):
            for _, row in chunk.iterrows():
                features = set()
                for col in ("ffa_matched_rules", "ffa_features", "consensus_features", "shap_top_positive"):
                    if col in chunk.columns:
                        for item in _parse_listish(row.get(col)):
                            features.update(_extract_drug_features_from_rule_text(item))
                            if str(item).startswith(DRUG_PREFIX):
                                features.add(str(item))
                features = features & set(top_drugs)
                if len(features) >= 2:
                    for size in range(2, min(max_size, len(features)) + 1):
                        for combo in itertools.combinations(sorted(features), size):
                            patient_support[combo] += 1
                if "ffa_matched_rules" in chunk.columns:
                    for rule in _parse_listish(row.get("ffa_matched_rules")):
                        rf = _extract_drug_features_from_rule_text(rule) & set(top_drugs)
                        if len(rf) >= 2:
                            for size in range(2, min(max_size, len(rf)) + 1):
                                for combo in itertools.combinations(sorted(rf), size):
                                    rule_support[combo] += 1

    combo_rows = []
    for size in range(2, min(max_size, len(top_drugs)) + 1):
        for combo in itertools.combinations(top_drugs, size):
            scores = [score_by_feature[c] for c in combo]
            ranks = [rank_by_feature[c] for c in combo]
            combo_rows.append({
                "cohort": cohort,
                "age_band": age_band,
                "bin": bin_name,
                "combination": "|".join(combo),
                "drug_names": " + ".join(_drug_name(c) for c in combo),
                "interaction_size": size,
                "mean_combined_importance": sum(scores) / len(scores),
                "sum_combined_importance": sum(scores),
                "best_rank": min(ranks),
                "worst_rank": max(ranks),
                "patient_rule_profile_support": int(patient_support.get(combo, 0)),
                "strict_rule_support": int(rule_support.get(combo, 0)),
                "evidence_type": "scenario_topk_rule_profile",
                "interpretation_limit": "co-occurring top SHAP/FFA drug profile; not explicit causal synergy",
            })

    single_rows = []
    for feature in top_drugs:
        single_rows.append({
            "cohort": cohort,
            "age_band": age_band,
            "bin": bin_name,
            "feature": feature,
            "drug_name": _drug_name(feature),
            "rank": rank_by_feature[feature],
            "combined_importance": score_by_feature[feature],
        })
    return combo_rows, single_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate multi-drug scenario profile summaries.")
    parser.add_argument("--scenario-root", type=Path, default=Path("10_risk_dashboard/visualizations/scenario"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/scenario_audit/multidrug_interactions"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-size", type=int, default=3)
    args = parser.parse_args()

    if not args.scenario_root.exists():
        raise FileNotFoundError(f"Scenario root not found: {args.scenario_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_combos = []
    all_singles = []
    for bin_dir in _iter_bin_dirs(args.scenario_root):
        combos, singles = summarize_bin(bin_dir, args.scenario_root, args.top_k, args.max_size)
        all_combos.extend(combos)
        all_singles.extend(singles)

    combo_df = pd.DataFrame(all_combos)
    single_df = pd.DataFrame(all_singles)
    combo_path = args.output_dir / "multidrug_scenario_profiles.csv"
    single_path = args.output_dir / "top_drug_features_by_bin.csv"
    combo_df.to_csv(combo_path, index=False)
    single_df.to_csv(single_path, index=False)

    if combo_df.empty:
        summary = {"total_profiles": 0}
    else:
        recurrent = (
            combo_df.groupby(["combination", "drug_names", "interaction_size"], as_index=False)
            .agg(
                n_bins=("combination", "size"),
                mean_combined_importance=("mean_combined_importance", "mean"),
                max_combined_importance=("mean_combined_importance", "max"),
                total_patient_rule_profile_support=("patient_rule_profile_support", "sum"),
                total_strict_rule_support=("strict_rule_support", "sum"),
            )
            .sort_values(["n_bins", "mean_combined_importance"], ascending=[False, False])
        )
        recurrent.to_csv(args.output_dir / "recurrent_multidrug_profiles.csv", index=False)
        for cohort, g in combo_df.groupby("cohort"):
            out = (
                g.groupby(["combination", "drug_names", "interaction_size"], as_index=False)
                .agg(
                    n_bins=("combination", "size"),
                    mean_combined_importance=("mean_combined_importance", "mean"),
                    total_patient_rule_profile_support=("patient_rule_profile_support", "sum"),
                    total_strict_rule_support=("strict_rule_support", "sum"),
                )
                .sort_values(["n_bins", "mean_combined_importance"], ascending=[False, False])
            )
            out.to_csv(args.output_dir / f"recurrent_multidrug_profiles_{cohort}.csv", index=False)
        summary = {
            "total_profiles": int(len(combo_df)),
            "total_bins": int(combo_df[["cohort", "age_band", "bin"]].drop_duplicates().shape[0]),
            "top_k": args.top_k,
            "max_size": args.max_size,
            "outputs": [str(p.name) for p in args.output_dir.glob("*.csv")],
        }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {combo_path}")


if __name__ == "__main__":
    main()
