"""
FP-Growth visualization helpers.

This module reads FP-Growth JSON outputs (itemsets and rules) and creates:
- Top-N itemset support bar charts (combined cohort) — PNG and optional Plotly HTML
- Network-style graphs from target-only rules (targets only) — PNG and Plotly HTML

Outputs are written to a local output directory and can be uploaded to S3
(same bucket as dashboard, under fpgrowth/{cohort}/{age_band}/plots/) for the dashboard to serve by cohort.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def _top_itemset_plot(
    df_itemsets: pd.DataFrame,
    cohort_name: str,
    age_band: str,
    item_type: str,
    top_n: int,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    if df_itemsets.empty or "support" not in df_itemsets.columns:
        return None

    # Derive a simple label for each itemset
    def _label(items) -> str:
        if not isinstance(items, list):
            return str(items)
        return ", ".join(str(x) for x in items)

    df = df_itemsets.copy()
    df["label"] = df["itemsets"].apply(_label)
    df = df.sort_values("support", ascending=False).head(top_n)

    if df.empty:
        return None

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=df,
        x="support",
        y="label",
        color="steelblue",
    )
    ax.set_xlabel("Support")
    ax.set_ylabel("Itemset")
    title = f"{cohort_name} {age_band} {item_type} top {len(df)} itemsets (combined)"
    ax.set_title(title)
    plt.tight_layout()

    fname = f"{cohort_name}_{age_band.replace('-', '_')}_{item_type}_combined_top_itemsets.png"
    out_path = output_dir / fname
    _ensure_output_dir(output_dir)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    if logger:
        logger.info("Saved top itemset plot to %s", out_path)

    return out_path


def _network_from_rules(
    df_rules: pd.DataFrame,
    cohort_name: str,
    age_band: str,
    item_type: str,
    min_rules: int,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    Build a simple directed network graph from association rules and save as PNG.

    Expected columns in df_rules:
      - antecedents: list of items
      - consequents: list of items
      - support
      - confidence
    """
    if df_rules.empty:
        return None

    if "antecedents" not in df_rules.columns or "consequents" not in df_rules.columns:
        return None

    if len(df_rules) < min_rules:
        # Too few rules for a meaningful network
        return None

    # Build directed graph
    G = nx.DiGraph()
    for _, row in df_rules.iterrows():
        ants = row["antecedents"]
        cons = row["consequents"]
        support = float(row.get("support", 0.0) or 0.0)
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if not isinstance(ants, list) or not isinstance(cons, list):
            continue
        for a in ants:
            for c in cons:
                if not a or not c:
                    continue
                if G.has_edge(a, c):
                    # Aggregate support and confidence by averaging
                    data = G[a][c]
                    data["support"] = (data["support"] + support) / 2.0
                    data["confidence"] = (data["confidence"] + confidence) / 2.0
                else:
                    G.add_edge(a, c, support=support, confidence=confidence)

    if G.number_of_edges() == 0:
        return None

    # Compute simple centrality for node sizing
    centrality = nx.degree_centrality(G)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.5)

    node_sizes = [300 + 2000 * centrality.get(n, 0.0) for n in G.nodes()]
    edge_widths = [1.0 + 5.0 * G[u][v].get("support", 0.0) for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", arrows=True, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)

    title = f"{cohort_name} {age_band} {item_type} target rules network"
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    fname = f"{cohort_name}_{age_band.replace('-', '_')}_{item_type}_target_rules_network.png"
    out_path = output_dir / fname
    _ensure_output_dir(output_dir)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    if logger:
        logger.info(
            "Saved target rules network (%d nodes, %d edges) to %s",
            G.number_of_nodes(),
            G.number_of_edges(),
            out_path,
        )

    return out_path


def _network_from_rules_plotly(
    df_rules: pd.DataFrame,
    cohort_name: str,
    age_band: str,
    item_type: str,
    min_rules: int,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    Build an interactive Plotly directed network from association rules and save as HTML.
    Same data contract as _network_from_rules; returns path to HTML file.
    """
    if not PLOTLY_AVAILABLE:
        if logger:
            logger.warning("Plotly not available; skipping network HTML")
        return None
    if df_rules.empty or "antecedents" not in df_rules.columns or "consequents" not in df_rules.columns:
        return None
    if len(df_rules) < min_rules:
        return None

    G = nx.DiGraph()
    for _, row in df_rules.iterrows():
        ants = row["antecedents"]
        cons = row["consequents"]
        support = float(row.get("support", 0.0) or 0.0)
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if not isinstance(ants, list) or not isinstance(cons, list):
            continue
        for a in ants:
            for c in cons:
                if not a or not c:
                    continue
                if G.has_edge(a, c):
                    data = G[a][c]
                    data["support"] = (data["support"] + support) / 2.0
                    data["confidence"] = (data["confidence"] + confidence) / 2.0
                else:
                    G.add_edge(a, c, support=support, confidence=confidence)

    if G.number_of_edges() == 0:
        return None

    pos = nx.spring_layout(G, seed=42, k=0.5)
    centrality = nx.degree_centrality(G)

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_sizes = [15 + 35 * centrality.get(n, 0.0) for n in G.nodes()]
    node_text = list(G.nodes())

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(size=node_sizes, color="lightblue", line=dict(width=1, color="darkblue")),
        hoverinfo="text",
        hovertext=node_text,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"{cohort_name} {age_band} {item_type} — target rules network",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        ),
    )

    fname = f"{cohort_name}_{age_band.replace('-', '_')}_{item_type}_target_rules_network.html"
    out_path = output_dir / fname
    _ensure_output_dir(output_dir)
    fig.write_html(str(out_path), config={"responsive": True})
    if logger:
        logger.info("Saved Plotly network HTML to %s", out_path)
    return out_path


def _s3_public_url(bucket: str, key: str, region: Optional[str] = None) -> str:
    """Return HTTPS URL for an S3 object (public-read style)."""
    if region:
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def create_all_fpgrowth_plots(
    base_dir: str,
    cohort_name: str,
    age_band: str,
    event_year: str,
    split_type: str = "combined",
    item_types: Optional[List[str]] = None,
    output_dir: str = "",
    s3_upload: bool = False,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "fpgrowth",
    top_n: int = 30,
) -> Dict[str, Any]:
    """
    Create standard FP-Growth plots for a cohort / age_band.
    Writes PNG and Plotly HTML (network) to output_dir; optionally uploads to S3.

    Implemented:
      - Combined itemsets: top-N itemset support bar chart (PNG)
      - Target-only rules: network PNG and Plotly interactive HTML

    Returns:
      Dict with:
        "plots": mapping item_type -> { plot_name: Path }
        "s3_urls": (if s3_upload) mapping item_type -> { plot_name: str URL }
    """
    # Logging is optional; use a basic logger so messages can be seen when run via CLI.
    logger = logging.getLogger("fpgrowth_plots")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    if not item_types:
        item_types = ["drug_name", "icd_code", "cpt_code", "medical_code"]

    base_path = Path(base_dir)
    age_band_fname = age_band.replace("-", "_")
    plots_root = Path(output_dir) if output_dir else base_path / "plots"

    results: Dict[str, Dict[str, Path]] = {}

    for item_type in item_types:
        logger.info(
            "Creating FP-Growth plots for %s / %s / %s (%s)",
            cohort_name,
            age_band,
            event_year,
            item_type,
        )
        item_results: Dict[str, Path] = {}

        # Combined itemsets (top-N support plot)
        combined_dir = (
            base_path / cohort_name / "combined" / age_band_fname / str(event_year)
        )
        combined_itemsets_path = combined_dir / f"{item_type}_itemsets.json"
        df_itemsets = _load_json_df(combined_itemsets_path)
        top_plot = _top_itemset_plot(
            df_itemsets=df_itemsets,
            cohort_name=cohort_name,
            age_band=age_band,
            item_type=item_type,
            top_n=top_n,
            output_dir=plots_root,
            logger=logger,
        )
        if top_plot is not None:
            item_results["combined_top_itemsets"] = top_plot

        # Target-only rules (network plot)
        target_dir = (
            base_path / cohort_name / "target" / age_band_fname / str(event_year)
        )
        target_rules_path = target_dir / f"{item_type}_rules_target_only.json"
        df_rules = _load_json_df(target_rules_path)
        net_plot = _network_from_rules(
            df_rules=df_rules,
            cohort_name=cohort_name,
            age_band=age_band,
            item_type=item_type,
            min_rules=5,
            output_dir=plots_root,
            logger=logger,
        )
        if net_plot is not None:
            item_results["target_rules_network"] = net_plot

        # Plotly interactive network HTML (for dashboard iframe by cohort)
        net_html = _network_from_rules_plotly(
            df_rules=df_rules,
            cohort_name=cohort_name,
            age_band=age_band,
            item_type=item_type,
            min_rules=5,
            output_dir=plots_root,
            logger=logger,
        )
        if net_html is not None:
            item_results["target_rules_network_html"] = net_html

        if item_results:
            results[item_type] = item_results

    out: Dict[str, Any] = {"plots": results}

    # Upload to S3 when requested (same bucket as dashboard; prefix e.g. fpgrowth -> cohort/age_band/plots/)
    if s3_upload and s3_bucket and results:
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
        except ImportError:
            logger = logging.getLogger("fpgrowth_plots")
            if logger.handlers:
                logger.warning("checkpoint_utils not available; skipping S3 upload")
            s3_upload = False
    if s3_upload and s3_bucket and results:
        age_band_fname = age_band.replace("-", "_")
        s3_plot_prefix = f"{s3_prefix.rstrip('/')}/{cohort_name}/{age_band}/plots"
        s3_urls: Dict[str, Dict[str, str]] = {}
        logger = logging.getLogger("fpgrowth_plots")
        for itype, paths in results.items():
            s3_urls[itype] = {}
            for plot_name, local_path in paths.items():
                if not isinstance(local_path, Path) or not local_path.exists():
                    continue
                key = f"{s3_plot_prefix}/{local_path.name}"
                s3_path = f"s3://{s3_bucket}/{key}"
                if upload_file_to_s3(local_path, s3_path, logger=logger, check_exists=True):
                    s3_urls[itype][plot_name] = _s3_public_url(s3_bucket, key)
            if not s3_urls[itype]:
                del s3_urls[itype]
        out["s3_urls"] = s3_urls

    return out


