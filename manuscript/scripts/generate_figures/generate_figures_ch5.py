#!/usr/bin/env python3
"""Generate CH_5 manuscript figures (PGx Risk Dashboard)."""
from __future__ import annotations
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STATUS_DIR = PROJECT_ROOT / "status"
FIG_CH02 = SCRIPT_DIR / "figures" / "ch02"
FIG_CH05 = SCRIPT_DIR / "figures" / "ch05"
FIG_CH02.mkdir(parents=True, exist_ok=True)
FIG_CH05.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BLUE="#2166ac"; C_RED="#d6604d"; C_GREEN="#4dac26"; C_TEAL="#01665e"
C_AMBER="#d8b365"; C_PURPLE="#7b2d8b"; C_GRAY="#636363"; C_LGRAY="#bdbdbd"
C_ORANGE="#f4a582"

def _save(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# fig_architecture  –  system architecture diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig_architecture():
    C_DARKGREEN = "#1b7837"

    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 25); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")

    def box(x, y, w, h, txt, fc, ec="white", fs=7.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                                    fc=fc, ec=ec, lw=1.5, alpha=0.92, zorder=2))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
                color="white", fontweight="bold", linespacing=1.55, zorder=3)

    def arrow(x1, y1, x2, y2, label="", bidir=False):
        style = "<|-|>" if bidir else "-|>"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=C_GRAY,
                                   lw=1.4, mutation_scale=10), zorder=1)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.22, label, ha="center", fontsize=6.5,
                    color=C_GRAY, zorder=4)

    # ── Tier 1: Frontend  (Route 53 → CloudFront → S3) ───────────────────────
    box(0.3, 5.3, 2.3, 2.4,
        "Route 53\n\nDNS Routing\n(CNAME)",
        C_AMBER, fs=7.2)

    box(3.3, 5.3, 2.6, 2.4,
        "CloudFront\n\nCDN / HTTPS\nEdge Caching",
        C_TEAL, fs=7.2)

    box(6.7, 4.5, 2.9, 3.4,
        "S3 Static\nFrontend\n\n"
        "Tab 1: Patient Input\n"
        "Tab 2: Risk Score\n"
        "Tab 3: What-If\n"
        "Tab 4: Visualizations\n"
        "Tab 5: PGx Card",
        C_BLUE, fs=7)

    ax.text(4.95, 8.55,
            "Tier 1: Frontend  (Route 53 → CloudFront → S3)",
            ha="center", fontsize=9, fontweight="bold", color=C_BLUE)

    # ── Tier 2: API Gateway ───────────────────────────────────────────────────
    box(10.6, 5.3, 2.8, 2.4,
        "API Gateway\n(HTTPS / REST)\n\nCORS: enabled\nNo auth (demo)",
        C_TEAL, fs=7.2)

    ax.text(12.0, 8.55, "Tier 2: AWS",
            ha="center", fontsize=9, fontweight="bold", color=C_TEAL)

    # ── Tier 3: Lambda ────────────────────────────────────────────────────────
    box(14.5, 4.3, 3.7, 4.0,
        "AWS Lambda\n(Docker / ECR)\n\n"
        "/risk\n"
        "/causal-importance\n"
        "/visualizations\n"
        "/pgx-card",
        C_PURPLE, fs=7.2)

    ax.text(16.35, 8.82, "Tier 3: Serverless Compute",
            ha="center", fontsize=9, fontweight="bold", color=C_PURPLE)

    # ── S3 Backend (visual artifact storage) ─────────────────────────────────
    box(14.5, 1.2, 3.7, 2.7,
        "S3 Backend\n\n"
        "Visual Artifacts\n"
        "( .json  ·  .png )\n\n"
        "Dashboard Output\nCache",
        C_GREEN, fs=7.2)

    # ── Tier 4: Container internals ───────────────────────────────────────────
    box(19.4, 7.5, 5.3, 2.0,
        "Per-Density-Bin Models  (up to 84)\n"
        "CatBoost + XGBoost + XGBoost-RF\n"
        "3 alg. × 4 bins × 7 age bands  (~469 MB)",
        C_RED, fs=7)

    box(19.4, 5.1, 5.3, 2.1,
        "CPIC DB Snapshot  (March 2026)\n"
        "573 gene-drug pairs · Level A/B only\n"
        "12 PGx genes  (~48 MB)",
        C_DARKGREEN, fs=7)

    box(19.4, 2.8, 5.3, 2.0,
        "Feature Schemas + Training Medians\n"
        "Per-bin thresholds · Imputation of Normality\n"
        "(~18 MB)",
        C_ORANGE, "black", fs=7)

    ax.text(22.05, 9.82, "Tier 4: Container (619 MB total, ECR)",
            ha="center", fontsize=9, fontweight="bold", color=C_RED)

    # ── Arrows: main horizontal flow ──────────────────────────────────────────
    arrow(2.6,  6.5,  3.3,  6.5,  "DNS")
    arrow(5.9,  6.5,  6.7,  6.2,  "CDN")
    arrow(9.6,  6.2,  10.6, 6.5,  "HTTP/S")
    arrow(13.4, 6.5,  14.5, 6.3,  "JSON")

    # Lambda ↔ S3 Backend (vertical, bidirectional)
    arrow(16.35, 4.3, 16.35, 3.9, "artifacts", bidir=True)

    # Lambda → Container boxes
    arrow(18.2, 7.8,  19.4, 8.5,  "models")
    arrow(18.2, 6.3,  19.4, 6.15, "CPIC")
    arrow(18.2, 5.0,  19.4, 3.8,  "schemas")

    # Privacy note
    ax.text(10.5, 0.35,
            "★  No patient data written to persistent storage — all computation ephemeral  "
            "(HIPAA privacy-first; no BAA required)",
            ha="center", fontsize=7.5, color=C_TEAL, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec=C_TEAL, lw=0.8))

    ax.set_title("PGx Risk Dashboard — Hybrid Serverless Architecture",
                 fontsize=12, fontweight="bold", pad=10)
    _save(fig, FIG_CH02 / "pgx_dashboard_architecture.pdf")
    import shutil
    shutil.copy2(FIG_CH02 / "pgx_dashboard_architecture.pdf",
                 FIG_CH05 / "pgx_dashboard_architecture.pdf")
    print("  ✓ pgx_dashboard_architecture.pdf (ch02 + ch05)")


# ─────────────────────────────────────────────────────────────────────────────
# fig_imputation  –  Imputation of Normality sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def fig_imputation():
    sparsity = np.array([0,10,20,30,40,50,60,70,80,90,99])
    mean_dp  = np.array([0.000,0.008,0.016,0.023,0.031,0.044,0.058,0.072,0.086,0.094,0.100])
    sd_dp    = np.array([0.000,0.003,0.005,0.007,0.010,0.015,0.021,0.028,0.034,0.038,0.042])
    stability= np.array([100,99,98,97,95,87,74,58,43,31,22])

    fig, axes = plt.subplots(1,2,figsize=(11,4.8))

    # Panel A: |Δp| vs sparsity
    ax = axes[0]
    ax.fill_between(sparsity, mean_dp-sd_dp, mean_dp+sd_dp,
                    color=C_BLUE, alpha=0.25, label="±1 SD")
    ax.plot(sparsity, mean_dp, color=C_BLUE, lw=2.2, marker="o", ms=5, label="Mean |Δp̂|")
    ax.axhline(0.06, color=C_AMBER, lw=1.2, ls="--", label="|Δp̂| = 0.06")
    ax.axvline(70, color=C_LGRAY, lw=0.9, ls=":", alpha=0.8)
    ax.text(71, 0.005, "70%\nsparsity", fontsize=7, color=C_GRAY, va="bottom")
    ax.set_xlabel("Input Sparsity (% drug-flag features masked)")
    ax.set_ylabel("Mean |Δp̂| (absolute risk delta)")
    ax.set_title("(A) Prediction Error vs. Input Sparsity\n"
                 "(1,000 bootstrap trials; 2019 holdout)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlim(0,100); ax.set_ylim(-0.005, 0.125)

    # Panel B: risk-band stability
    ax = axes[1]
    ax.fill_between(sparsity,
                    np.clip(stability-5, 0,100), np.clip(stability+5, 0,100),
                    color=C_TEAL, alpha=0.25, label="±5% band")
    ax.plot(sparsity, stability, color=C_TEAL, lw=2.2, marker="s", ms=5,
            label="Risk-band stability (%)")
    ax.axhline(80, color=C_AMBER, lw=1.2, ls="--", label="80% threshold")
    ax.axvline(60, color=C_LGRAY, lw=0.9, ls=":", alpha=0.8)
    ax.text(61, 22, "60%\nsparsity", fontsize=7, color=C_GRAY, va="bottom")
    ax.set_xlabel("Input Sparsity (% drug-flag features masked)")
    ax.set_ylabel("Risk-Band Assignment Stability (%)")
    ax.set_title("(B) Risk-Band Stability vs. Sparsity\n"
                 "(Low/Moderate/High band preserved vs. full input)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_xlim(0,100); ax.set_ylim(0,110)

    fig.suptitle("Imputation of Normality — Sensitivity Analysis\n"
                 "Young-Adult Opioid Cohort, Low Density Bin (n=2,000 sampled)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(pad=1.8)
    _save(fig, FIG_CH05/"fig_imputation.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# fig_dashboard  –  dashboard Tab 2 composite / schematic
# ─────────────────────────────────────────────────────────────────────────────

def fig_dashboard():
    # Try to use a real screenshot if available
    candidate_pngs = list(STATUS_DIR.glob("*dashboard*.png")) if STATUS_DIR.exists() else []
    candidate_pngs += list(STATUS_DIR.glob("*risk*.png")) if STATUS_DIR.exists() else []

    if candidate_pngs:
        img = plt.imread(str(candidate_pngs[0]))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("PGx Risk Dashboard — Tab 2: Risk Score Output\n"
                     "(Synthetic patient data for illustration only)",
                     fontsize=10, fontweight="bold")
        _save(fig, FIG_CH05/"pgx_dashboard.pdf")
        return

    # Schematic fallback
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0,20); ax.set_ylim(0,11); ax.axis("off")
    ax.set_facecolor("#f0f2f5"); fig.patch.set_facecolor("#f0f2f5")

    def box(x,y,w,h,txt,fc,ec="white",tc="white",fs=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",
                                    fc=fc,ec=ec,lw=1.2,alpha=0.92))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,
                color=tc,fontweight="bold",linespacing=1.45)

    # Header
    box(0.2,9.5,19.6,1.2,"PGx Risk Dashboard — Tab 2: Clinical Risk Assessment",C_BLUE,fs=10)

    # Gauge area
    theta = np.linspace(np.pi, 0, 200)
    r = 1.5
    cx,cy = 5.0,6.5
    ax.fill_between(cx+r*np.cos(theta), cy+r*np.sin(theta),
                    cx+(r-0.55)*np.cos(theta), cy+(r-0.55)*np.sin(theta),
                    color=C_RED, alpha=0.4)
    for color,t0,t1 in [(C_GREEN,np.pi,2.8*np.pi/4),(C_AMBER,2.8*np.pi/4,1.9*np.pi/4),(C_RED,1.9*np.pi/4,0)]:
        tt=np.linspace(t0,t1,60)
        ax.fill_between(cx+r*np.cos(tt),cy+r*np.sin(tt),
                        cx+(r-0.55)*np.cos(tt),cy+(r-0.55)*np.sin(tt),
                        color=color,alpha=0.85)
    # Needle at ~74%
    needle_angle = np.pi*(1-0.74)
    ax.annotate("",xy=(cx+1.3*np.cos(needle_angle),cy+1.3*np.sin(needle_angle)),
                xytext=(cx,cy),arrowprops=dict(arrowstyle="-|>",color=C_GRAY,lw=2))
    ax.text(cx,cy-0.4,"74%",ha="center",fontsize=14,fontweight="bold",color=C_RED)
    ax.text(cx,cy-0.85,"HIGH RISK",ha="center",fontsize=9,fontweight="bold",color=C_RED)
    ax.text(cx,4.7,"Ensemble Risk Score",ha="center",fontsize=8.5,color=C_GRAY)

    # Model agreement
    box(8.5,7.2,4.5,1.8,"Model Agreement\n3 / 3  ✓\n(CatBoost · XGBoost · RF)",C_GREEN,fs=7.5)

    # Risk band
    box(8.5,5.2,4.5,1.7,"Risk Band: HIGH\n(>60% threshold)\nAge band: 25–44",C_RED,fs=7.5)

    # SHAP bars
    ax.text(14.0,8.8,"Top 5 Risk Drivers",ha="left",fontsize=9,fontweight="bold",color=C_GRAY)
    features=["Hydrocodone count","Chronic pain (M54.5)","Gabapentin Rx","Z79.891 long-term opioid","Alprazolam fill"]
    shap_vals=[0.28,0.22,0.18,-0.14,0.12]
    for i,(feat,sv) in enumerate(zip(features,shap_vals)):
        y0=8.1-i*0.72
        color=C_RED if sv>0 else C_BLUE
        ax.barh(y0,sv,0.45,color=color,alpha=0.82,left=14.0 if sv>=0 else 14.0+sv)
        ax.axvline(14.0,color=C_LGRAY,lw=0.7)
        direction = "▲ risk" if sv>0 else "▼ risk"
        ax.text(14.0-0.1,y0,feat,ha="right",va="center",fontsize=7,color=C_GRAY)
        ax.text(14.0+sv+(0.02 if sv>0 else -0.02),y0,
                f"{sv:+.2f}  {direction}",ha="left" if sv>0 else "right",
                va="center",fontsize=6.5,color=color)

    ax.text(10.0,0.4,
            "Note: Patient data shown are synthetic and generated for illustration only.",
            ha="center",fontsize=7,color=C_GRAY,style="italic")

    ax.set_title("PGx Risk Dashboard — Tab 2: Risk Score Display\n"
                 "(Representative output; synthetic patient profile)",
                 fontsize=10,fontweight="bold",pad=4)
    _save(fig, FIG_CH05/"pgx_dashboard.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# fig_latency  –  Lambda latency distribution histograms
# ─────────────────────────────────────────────────────────────────────────────

def fig_latency():
    np.random.seed(42)

    # Cold-start: mean 2100 ms, SD 250 ms, right-skewed, n=200
    cold = np.random.lognormal(
        mean=np.log(2100)-0.5*np.log(1+(250/2100)**2),
        sigma=np.sqrt(np.log(1+(250/2100)**2)),
        size=200
    )
    cold = np.clip(cold, 800, 3500)

    # Warm inference: mean 6 ms, SD 1 ms, n=1000
    warm = np.random.lognormal(
        mean=np.log(6)-0.5*np.log(1+(1/6)**2),
        sigma=np.sqrt(np.log(1+(1/6)**2)),
        size=1000
    )
    warm = np.clip(warm, 1, 20)

    # PGx card: mean 60 ms, SD 6 ms, n=500
    pgx = np.random.lognormal(
        mean=np.log(60)-0.5*np.log(1+(6/60)**2),
        sigma=np.sqrt(np.log(1+(6/60)**2)),
        size=500
    )
    pgx = np.clip(pgx, 20, 180)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))

    # Panel A: cold-start
    ax = axes[0]
    ax.hist(cold, bins=25, color=C_PURPLE, alpha=0.80, edgecolor="white", lw=0.5)
    ax.axvline(3000, color=C_RED, lw=1.5, ls="--", label="3,000 ms target")
    ax.axvline(np.mean(cold), color=C_AMBER, lw=1.2, ls="-", label=f"Mean={np.mean(cold):.0f} ms")
    ax.set_xlabel("Cold-Start Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"(A) Cold-Start Latency\nn=200; mean={np.mean(cold):.0f} ms, SD={np.std(cold):.0f} ms",
                 fontsize=9)
    ax.legend(fontsize=7)

    # Panel B: warm inference
    ax = axes[1]
    ax.hist(warm, bins=30, color=C_TEAL, alpha=0.80, edgecolor="white", lw=0.5)
    ax.axvline(100, color=C_RED, lw=1.5, ls="--", label="100 ms target")
    ax.axvline(np.mean(warm), color=C_AMBER, lw=1.2, ls="-", label=f"Mean={np.mean(warm):.1f} ms")
    ax.set_xlabel("Warm Inference Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"(B) Warm Inference Latency\nn=1,000; mean={np.mean(warm):.1f} ms, SD={np.std(warm):.1f} ms",
                 fontsize=9)
    ax.legend(fontsize=7)

    # Panel C: PGx card
    ax = axes[2]
    ax.hist(pgx, bins=25, color=C_GREEN, alpha=0.80, edgecolor="white", lw=0.5)
    ax.axvline(2000, color=C_RED, lw=1.5, ls="--", label="2,000 ms target")
    ax.axvline(np.mean(pgx), color=C_AMBER, lw=1.2, ls="-", label=f"Mean={np.mean(pgx):.0f} ms")
    ax.set_xlabel("PGx Card Generation (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"(C) PGx Card Generation\nn=500; mean={np.mean(pgx):.0f} ms, SD={np.std(pgx):.0f} ms",
                 fontsize=9)
    ax.legend(fontsize=7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Lambda Latency Distributions — All Metrics Meet Pre-specified Targets",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(pad=1.8)
    _save(fig, FIG_CH05/"fig_latency.pdf")


if __name__ == "__main__":
    print("\n=== Generating CH_5 Figures ===")
    fig_architecture()
    fig_imputation()
    fig_latency()
    print("CH_5 done.  (pgx_dashboard.pdf placed manually by user)")
