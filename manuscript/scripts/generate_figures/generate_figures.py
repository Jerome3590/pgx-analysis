#!/usr/bin/env python3
"""
Master figure generation script for all manuscript chapters.
Calls generate_figures_ch3, ch4, and ch5 in sequence.

Usage:
    cd C:/Projects/pgx-analysis
    python manuscript/generate_figures.py
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

print("=" * 60)
print("Manuscript Figure Generator")
print("=" * 60)

print("\n=== CH_3: Opioid ED (25-44 band) ===")
import generate_figures_ch3 as ch3
ch3.fig_attrition()
ch3.fig_curves()
ch3.fig_shap()
ch3.fig_shap_pdp()
ch3.fig_trajectories()

print("\n=== CH_4: Non-Opioid ED / Polypharmacy (65-74 band) ===")
import generate_figures_ch4 as ch4
ch4.fig_network()
ch4.fig_ir()
ch4.fig_zcode()
ch4.fig_shap_pdp()

print("\n=== CH_5: PGx Risk Dashboard ===")
import generate_figures_ch5 as ch5
ch5.fig_architecture()
ch5.fig_imputation()
ch5.fig_dashboard()
ch5.fig_latency()

print("\n" + "=" * 60)
print("All figures generated successfully.")
print("=" * 60)

# Verify outputs
from pathlib import Path
expected = {
    "ch03": ["fig_attrition.pdf","fig_curves.pdf","fig_shap.pdf",
             "fig_shap_pdp.pdf","fig_trajectories.pdf"],
    "ch04": ["fig_network.pdf","fig_ir.pdf","fig_zcode.pdf","fig_shap_pdp.pdf"],
    "ch02": ["pgx_dashboard_architecture.pdf"],
    "ch05": ["pgx_dashboard_architecture.pdf","fig_imputation.pdf",
             "pgx_dashboard.pdf","fig_latency.pdf"],
}
fig_root = SCRIPT_DIR.parent.parent / "figures"  # manuscript/figures (same paths Quarto uses)
missing = []
for ch, figs in expected.items():
    for f in figs:
        p = fig_root / ch / f
        if not p.exists() or p.stat().st_size < 1000:
            missing.append(str(p.relative_to(fig_root.parent)))

if missing:
    print("\n⚠ Missing or empty figures:")
    for m in missing:
        print(f"  {m}")
else:
    print("\n✓ All expected figure files present and non-empty.")
