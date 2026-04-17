#!/usr/bin/env python3
"""
Check that EC2 paths and artifacts required for the dashboard exist before Lambda update / S3 sync.

Aligns with 10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md and
RESEARCH_QUESTIONS_ARTIFACTS.md. Run from repo root (e.g. before Step 6 in 5_build_and_deploy.ipynb).

Checks both cohort-level paths and per-bin paths (causal, BupaR, FP-Growth, PGx). Per-bin checks
are informational; --strict only fails on required cohort-level/metadata paths.

Usage:
  python 10_risk_dashboard/data_preparation/check_dashboard_artifact_paths.py
  python 10_risk_dashboard/data_preparation/check_dashboard_artifact_paths.py --strict   # exit 1 if any required missing
  python 10_risk_dashboard/data_preparation/check_dashboard_artifact_paths.py --project-root /path/to/pgx-analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root(project_root: Path | None) -> Path:
    if project_root is not None and project_root.exists():
        return project_root.resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "10_risk_dashboard").exists():
        return cwd
    if (cwd / "5_build_and_deploy.ipynb").exists():
        return cwd
    # Assume we're in 10_risk_dashboard or data_preparation
    for parent in [cwd, cwd.parent, cwd.parent.parent]:
        if (parent / "10_risk_dashboard").exists():
            return parent
    return cwd


_BINS = ("low", "medium", "high", "extreme")


def _list_trained_bins(root: Path, cohort: str, age_band: str) -> list[str]:
    """Return bins that have Step 6 artifacts (per-bin models trained)."""
    try:
        import sys
        sys.path.insert(0, str(root))
        from py_helpers.event_density_utils import list_trained_density_bins
        return list_trained_density_bins(root, cohort, age_band)
    except ImportError:
        return []


def _get_cohorts_and_bands(root: Path) -> list[tuple[str, str]]:
    try:
        sys.path.insert(0, str(root))
        from py_helpers.constants import REQUIRED_COHORTS
        combos = []
        for cohort, bands in REQUIRED_COHORTS.items():
            for age_band in bands:
                combos.append((cohort, age_band))
        return combos
    except ImportError:
        # Fallback
        cohorts = ["opioid_ed", "non_opioid_ed"]
        bands = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
        return [(c, b) for c in cohorts for b in bands]


def check_feature_importance(root: Path) -> list[tuple[str, bool, str]]:
    """Feature Importance tab: aggregated heatmaps per cohort and combined."""
    results = []
    fi_base = root / "3a_feature_importance" / "outputs"
    for cohort in ("opioid_ed", "non_opioid_ed"):
        png = fi_base / cohort / "plots" / f"{cohort}_aggregated_fi_heatmap.png"
        json_p = fi_base / cohort / "plots" / f"{cohort}_aggregated_fi_heatmap.json"
        results.append((f"FI {cohort} (PNG)", png.exists(), str(png)))
        results.append((f"FI {cohort} (JSON)", json_p.exists(), str(json_p)))
    combined_png = fi_base / "plots" / "combined_cohorts_feature_importance_heatmap.png"
    combined_json = fi_base / "plots" / "combined_cohorts_aggregated_fi_heatmap.json"
    if not combined_json.exists():
        combined_json = fi_base / "combined" / "aggregated_fi_heatmap.json"
    results.append(("FI combined (PNG)", combined_png.exists(), str(combined_png)))
    results.append(("FI combined (JSON)", combined_json.exists(), str(combined_json)))
    return results


def check_causal(root: Path, combos: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """Causal Analysis tab: dashboard_data.json per cohort/age_band (visualizations/causal)."""
    results = []
    out = root / "10_risk_dashboard" / "visualizations" / "causal"
    for cohort, age_band in combos:
        ab_fname = age_band.replace("-", "_")
        path = out / cohort / ab_fname / "dashboard_data.json"
        results.append((f"Causal {cohort}/{age_band}", path.exists(), str(path)))
    return results


def check_bupar(root: Path, combos: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """BupaR Process Mining tab: RQ artifact files in plots/ per cohort/age_band."""
    results = []
    base_dir = root / "10_risk_dashboard" / "visualizations" / "bupar"
    for cohort, age_band in combos:
        ab_fname = age_band.replace("-", "_")
        base_name = f"{cohort}_{ab_fname}"
        pre = "pre_f1120" if cohort == "opioid_ed" else "pre_hcg"
        plots_dir = base_dir / cohort / ab_fname / "plots"
        # At least one RQ artifact must exist
        rq_files = [
            plots_dir / f"{base_name}_activity_frequency.json",
            plots_dir / f"{base_name}_pre_target_activity_frequency.json",
            plots_dir / f"{base_name}_process_matrix_drug_drug.png",
            plots_dir / f"{base_name}_trace_explorer_{pre}.png",
        ]
        any_exists = plots_dir.exists() and any(p.exists() for p in rq_files)
        results.append((f"BupaR {cohort}/{age_band}", any_exists, str(plots_dir)))
    return results


def check_dtw(root: Path, combos: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """DTW Trajectories tab: chart_data.json, sequence_heatmap.json per cohort/age_band."""
    results = []
    base_dir = root / "10_risk_dashboard" / "visualizations" / "dtw"
    for cohort, age_band in combos:
        ab_fname = age_band.replace("-", "_")
        out_dir = base_dir / cohort / ab_fname
        chart = out_dir / "chart_data.json"
        heatmap = out_dir / "sequence_heatmap.json"
        results.append((f"DTW chart_data {cohort}/{age_band}", chart.exists(), str(chart)))
        results.append((f"DTW sequence_heatmap {cohort}/{age_band}", heatmap.exists(), str(heatmap)))
    return results


def check_fpgrowth(root: Path, combos: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """FP-Growth Patterns tab: combined_rules_network.html or itemsets per cohort/age_band."""
    results = []
    base_dir = root / "10_risk_dashboard" / "visualizations" / "fpgrowth"
    for cohort, age_band in combos:
        ab_fname = age_band.replace("-", "_")
        plots_dir = base_dir / cohort / ab_fname / "plots"
        data_dir = base_dir / cohort / ab_fname / "data"
        network = plots_dir / f"{cohort}_{ab_fname}_combined_rules_network.html"
        itemsets_png = plots_dir / f"{cohort}_{ab_fname}_drug_name_combined_top_itemsets.png"
        itemsets_json = data_dir / "drug_name_itemsets.json"
        any_exists = (
            network.exists() or itemsets_png.exists() or itemsets_json.exists()
        )
        results.append((f"FP-Growth {cohort}/{age_band}", any_exists, str(plots_dir)))
    return results


def check_pgx_cohort(root: Path, combos: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """PGx Cohort tab: network_topology.html per cohort/age_band."""
    results = []
    base_dir = root / "10_risk_dashboard" / "visualizations" / "cohort_pgx" / "networks"
    for cohort, age_band in combos:
        ab_fname = age_band.replace("-", "_")
        path = base_dir / cohort / ab_fname / "network_topology.html"
        results.append((f"PGx Cohort {cohort}/{age_band}", path.exists(), str(path)))
    return results


def check_causal_per_bin(root: Path, combos: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """Per-bin causal dashboard_data.json (for bins with Step 6 models). Informational only."""
    results = []
    out = root / "10_risk_dashboard" / "visualizations" / "causal"
    for cohort, age_band in combos:
        ab_fname = age_band.replace("-", "_")
        trained = _list_trained_bins(root, cohort, age_band)
        for bin_name in trained:
            path = out / cohort / ab_fname / bin_name / "dashboard_data.json"
            results.append((f"Causal (bin) {cohort}/{age_band}/{bin_name}", path.exists(), str(path)))
    return results


def check_bupar_per_bin(root: Path, combos: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """Per-bin BupaR plots under density/{bin}/. Informational only."""
    results = []
    base_dir = root / "10_risk_dashboard" / "visualizations" / "bupar"
    for cohort, age_band in combos:
        ab_fname = age_band.replace("-", "_")
        base_name = f"{cohort}_{ab_fname}"
        for bin_name in _BINS:
            plots_dir = base_dir / cohort / ab_fname / "density" / bin_name / "plots"
            probe = plots_dir / f"{base_name}_activity_frequency.json"
            results.append((f"BupaR (bin) {cohort}/{age_band}/{bin_name}", probe.exists(), str(plots_dir)))
    return results


def check_fpgrowth_per_bin(root: Path, combos: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """Per-bin FP-Growth itemsets/rules under density/{bin}/. Informational only."""
    results = []
    base_dir = root / "10_risk_dashboard" / "visualizations" / "fpgrowth"
    for cohort, age_band in combos:
        ab_fname = age_band.replace("-", "_")
        for bin_name in _BINS:
            density_dir = base_dir / cohort / ab_fname / "density" / bin_name
            itemsets = density_dir / "drug_name_itemsets.json"
            rules = density_dir / "drug_name_rules.json"
            any_exists = itemsets.exists() or rules.exists()
            results.append((f"FP-Growth (bin) {cohort}/{age_band}/{bin_name}", any_exists, str(density_dir)))
    return results


def check_pgx_per_bin(root: Path, combos: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """Per-bin PGx network_topology.html under density/{bin}/. Informational only."""
    results = []
    base_dir = root / "10_risk_dashboard" / "visualizations" / "cohort_pgx" / "networks"
    for cohort, age_band in combos:
        ab_fname = age_band.replace("-", "_")
        for bin_name in _BINS:
            path = base_dir / cohort / ab_fname / "density" / bin_name / "network_topology.html"
            results.append((f"PGx (bin) {cohort}/{age_band}/{bin_name}", path.exists(), str(path)))
    return results


def check_metadata_frontend(root: Path) -> list[tuple[str, bool, str]]:
    """Frontend, metadata (required for deploy)."""
    dash = root / "10_risk_dashboard"
    results = [
        ("Frontend index.html", (dash / "frontend" / "index.html").exists(), str(dash / "frontend" / "index.html")),
        ("Metadata model_performance_metrics.json", (dash / "outputs" / "metadata" / "model_performance_metrics.json").exists(), ""),
        ("Metadata opioid_ed.json", (dash / "outputs" / "metadata" / "metadata_opioid_ed.json").exists(), ""),
        ("Metadata non_opioid_ed.json", (dash / "outputs" / "metadata" / "metadata_non_opioid_ed.json").exists(), ""),
    ]
    return results


def _print_section(title: str, results: list[tuple[str, bool, str]]) -> int:
    ok_count = sum(1 for _, ok, _ in results if ok)
    print(f"--- {title} ---")
    for name, ok, path in results:
        sym = "[OK]" if ok else "[--]"
        print(f"  {sym} {name}")
    print(f"  ({ok_count}/{len(results)} present)")
    print()
    return ok_count


def _build_available_json(
    combos: list[tuple[str, str]],
    meta: list, fi: list,
    causal: list, bupar: list, dtw: list, fpg: list, pgx: list,
    causal_bin: list, bupar_bin: list, fpg_bin: list, pgx_bin: list,
) -> dict:
    """Build structured availability dict from check results for available.json."""
    import datetime

    def _cohort_bands(results: list[tuple[str, bool, str]], prefix: str) -> dict:
        """Return {cohort: [age_band, ...]} for OK results matching 'prefix cohort/age_band'."""
        from collections import defaultdict
        out: dict = defaultdict(list)
        for name, ok, _ in results:
            if not ok:
                continue
            label = name[len(prefix):].strip()  # "opioid_ed/13-24"
            parts = label.split("/")
            if len(parts) == 2:
                out[parts[0]].append(parts[1])
        return dict(out)

    def _cohort_band_bins(results: list[tuple[str, bool, str]], prefix: str) -> dict:
        """Return {cohort: {age_band: [bin, ...]}} for OK per-bin results."""
        from collections import defaultdict
        out: dict = defaultdict(lambda: defaultdict(list))
        for name, ok, _ in results:
            if not ok:
                continue
            label = name[len(prefix):].strip()  # "opioid_ed/13-24/low"
            parts = label.split("/")
            if len(parts) == 3:
                out[parts[0]][parts[1]].append(parts[2])
        return {c: dict(v) for c, v in out.items()}

    # Risk availability: infer from causal_per_bin (same training pipeline).
    # Any cohort/age_band with at least one trained bin has risk models.
    risk_map: dict = {}
    cbb = _cohort_band_bins(causal_bin, "Causal (bin) ")
    for cohort, band_bins in cbb.items():
        risk_map[cohort] = sorted(band_bins.keys())

    return {
        "generated": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "risk":          risk_map,
        "causal":        _cohort_bands(causal,    "Causal "),
        "bupar":         _cohort_bands(bupar,      "BupaR "),
        "dtw":           _cohort_bands(
                             [r for r in dtw if "chart_data" in r[0]], "DTW chart_data "),
        "fpgrowth":      _cohort_bands(fpg,        "FP-Growth "),
        "pgx_cohort":    _cohort_bands(pgx,        "PGx Cohort "),
        "causal_per_bin":  _cohort_band_bins(causal_bin, "Causal (bin) "),
        "bupar_per_bin":   _cohort_band_bins(bupar_bin,  "BupaR (bin) "),
        "fpgrowth_per_bin": _cohort_band_bins(fpg_bin,   "FP-Growth (bin) "),
        "pgx_per_bin":     _cohort_band_bins(pgx_bin,    "PGx (bin) "),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check dashboard artifact paths (EC2) before Lambda/S3 sync. See README_dashboard_visual_artifact_paths.md."
    )
    ap.add_argument("--project-root", type=Path, default=None, help="Repo root (default: auto-detect)")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if any required path is missing")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Write structured available.json to this path (for S3 upload and Lambda /available endpoint)")
    args = ap.parse_args()

    root = _repo_root(args.project_root)
    combos = _get_cohorts_and_bands(root)

    print("Dashboard artifact path check (README_dashboard_visual_artifact_paths.md)")
    print("=" * 72)
    print(f"Project root: {root}")
    print(f"Cohort/age_band combinations: {len(combos)}")
    print()

    required_fail = 0

    # Required for deploy
    meta = check_metadata_frontend(root)
    _print_section("Metadata & Frontend (required for deploy)", meta)
    for _, ok, _ in meta:
        if not ok:
            required_fail += 1

    # Feature Importance (at least one heatmap required for FI tab)
    fi = check_feature_importance(root)
    _print_section("Feature Importance", fi)
    if sum(1 for _, ok, _ in fi if ok) == 0:
        required_fail += 1

    # Causal
    causal = check_causal(root, combos)
    _print_section("Causal Analysis", causal)

    # BupaR (at least one combo required for BupaR tab)
    bupar = check_bupar(root, combos)
    _print_section("BupaR Process Mining", bupar)
    if not any(ok for _, ok, _ in bupar):
        required_fail += 1

    # DTW (at least one combo required for DTW tab)
    dtw = check_dtw(root, combos)
    _print_section("DTW Trajectories", dtw)
    dtw_any = any(ok for _, ok, _ in dtw)
    if not dtw_any:
        required_fail += 1

    # FP-Growth (at least one combo required for FP-Growth tab)
    fpg = check_fpgrowth(root, combos)
    _print_section("FP-Growth Patterns", fpg)
    if not any(ok for _, ok, _ in fpg):
        required_fail += 1

    # PGx Cohort (optional)
    pgx = check_pgx_cohort(root, combos)
    _print_section("PGx Cohort", pgx)

    # Per-bin paths (informational; Lambda/dashboard use these when density filter is active)
    causal_bin = check_causal_per_bin(root, combos)
    _print_section("Causal Analysis (per-bin)", causal_bin)
    bupar_bin = check_bupar_per_bin(root, combos)
    _print_section("BupaR (per-bin)", bupar_bin)
    fpg_bin = check_fpgrowth_per_bin(root, combos)
    _print_section("FP-Growth (per-bin)", fpg_bin)
    pgx_bin = check_pgx_per_bin(root, combos)
    _print_section("PGx Cohort (per-bin)", pgx_bin)

    total_results = meta + fi + causal + bupar + dtw + fpg + pgx
    total_ok = sum(1 for _, ok, _ in total_results if ok)
    print("=" * 72)
    print(f"Total: {total_ok}/{len(total_results)} paths present")
    if required_fail > 0:
        print(f"Required missing: {required_fail} (fix before Step 6 / Lambda deploy)")
        print("  See 10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md for EC2 paths.")
        if args.strict:
            print("Exiting with code 1 (--strict).")
            return 1
    else:
        print("All required dashboard artifact paths present.")

    if args.json_out:
        import json
        avail = _build_available_json(
            combos, meta, fi, causal, bupar, dtw, fpg, pgx,
            causal_bin, bupar_bin, fpg_bin, pgx_bin,
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(avail, indent=2))
        print(f"\navailable.json written → {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
