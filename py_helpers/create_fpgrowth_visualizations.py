"""
FP-Growth visualization helpers.

This module reads FP-Growth JSON outputs (itemsets and rules) and creates:
- Top-N itemset support bar charts (combined cohort) — PNG and optional Plotly HTML
- Network-style graphs from target-only rules (targets only) — PNG and Plotly HTML

Optional code mapping table (code -> description) makes node and itemset labels
human-readable; see 9_dashboard_visuals/fpgrowth/code_mappings/README.md.

Outputs are written to a local output directory and can be uploaded to S3
(same bucket as dashboard, under fpgrowth/{cohort}/{age_band}/plots/) for the dashboard to serve by cohort.
"""

import csv
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

# Production HTML pattern: single-file, embedded Plotly.js, so HTML renders in iframes/S3 without external deps.
# BupaR uses the same pattern via R htmlwidgets::saveWidget(..., selfcontained = TRUE).
PLOTLY_HTML_CONFIG = {"responsive": True, "displayModeBar": True}
PLOTLY_INCLUDE_JS = True  # embed full Plotly.js for production rendering

# Item types for FP-Growth: one combined graph with nodes colored/filterable by type.
# medical_code is the union of these three (allowed_codes); not a separate graph.
FPGROWTH_GRAPH_ITEM_TYPES = ["drug_name", "icd_code", "cpt_code"]
# Display labels and colors for filter dropdown and node coloring
ITEM_TYPE_LABELS = {"drug_name": "Drug", "icd_code": "ICD", "cpt_code": "CPT"}
ITEM_TYPE_COLORS = {"drug_name": "#3b82f6", "icd_code": "#22c55e", "cpt_code": "#f59e0b"}


def write_plotly_html_for_production(fig: "go.Figure", out_path: Path, config: Optional[Dict[str, Any]] = None) -> None:
    """Write Plotly figure to a single self-contained HTML file for production (iframes, S3, local)."""
    cfg = config if config is not None else PLOTLY_HTML_CONFIG
    fig.write_html(str(out_path), config=cfg, include_plotlyjs=PLOTLY_INCLUDE_JS)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_code_mapping(mapping_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load code -> description mapping from CSV (columns: code, description).
    Used to show human-readable labels in network graph and itemset plots.
    If path is None, tries default: 9_dashboard_visuals/fpgrowth/code_mappings/fpgrowth_code_descriptions.csv.
    """
    if mapping_path is None:
        # Default: repo root is parent of py_helpers
        repo_root = Path(__file__).resolve().parent.parent
        mapping_path = (
            repo_root
            / "9_dashboard_visuals"
            / "fpgrowth"
            / "code_mappings"
            / "fpgrowth_code_descriptions.csv"
        )
    if not mapping_path.exists():
        return {}
    out: Dict[str, str] = {}
    try:
        with mapping_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "code" not in (reader.fieldnames or []):
                return {}
            for row in reader:
                code = (row.get("code") or "").strip()
                desc = (row.get("description") or "").strip()
                if code:
                    out[code] = desc
    except Exception:
        return {}
    return out


def _load_json_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def _load_rules_all_types(
    base_path: Path,
    cohort_name: str,
    age_band: str,
    split_type: str,
    event_year: str,
) -> List[tuple]:
    """
    Load rules for drug_name, icd_code, cpt_code from cohort/age_band (visualization artifacts = cohort then age_band only).
    Returns list of (item_type, df_rules). Tries rules_target_only.json and rules_lift_filtered.json.
    split_type and event_year are ignored; paths are base_path/cohort_name/age_band_fname/.
    """
    age_band_fname = age_band.replace("-", "_")
    dir_path = base_path / cohort_name / age_band_fname
    out: List[tuple] = []
    for item_type in FPGROWTH_GRAPH_ITEM_TYPES:
        for fname in (f"{item_type}_rules_target_only.json", f"{item_type}_rules_lift_filtered.json"):
            path = dir_path / fname
            df = _load_json_df(path)
            if not df.empty and "antecedents" in df.columns and "consequents" in df.columns:
                out.append((item_type, df))
                break
    return out


def _build_combined_rules_graph(
    rules_by_type: List[tuple],
    min_rules_per_type: int = 2,
    code_mapping: Optional[Dict[str, str]] = None,
) -> Optional[Any]:
    """
    Build one DiGraph from rules for drug_name, icd_code, cpt_code.
    Node ids are prefixed by type so the same label in different types stays distinct.
    Each node has attributes: node_type (str), display_label (str).
    If code_mapping is provided, display_label uses description when available for viewable labels.
    """
    if not rules_by_type:
        return None
    G = nx.DiGraph()
    for item_type, df_rules in rules_by_type:
        if len(df_rules) < min_rules_per_type:
            continue
        for _, row in df_rules.iterrows():
            ants = row["antecedents"]
            cons = row["consequents"]
            support = float(row.get("support", 0.0) or 0.0)
            confidence = float(row.get("confidence", 0.0) or 0.0)
            lift = float(row.get("lift", 0.0) or 0.0)
            if not isinstance(ants, list) or not isinstance(cons, list):
                continue
            for a in ants:
                for c in cons:
                    if not a or not c:
                        continue
                    u = f"{item_type}::{a}"
                    v = f"{item_type}::{c}"
                    label_a = (code_mapping.get(str(a).strip(), str(a))) if code_mapping else str(a)
                    label_c = (code_mapping.get(str(c).strip(), str(c))) if code_mapping else str(c)
                    if not G.has_node(u):
                        G.add_node(u, node_type=item_type, display_label=label_a)
                    if not G.has_node(v):
                        G.add_node(v, node_type=item_type, display_label=label_c)
                    if G.has_edge(u, v):
                        data = G[u][v]
                        data["support"] = (data["support"] + support) / 2.0
                        data["confidence"] = (data["confidence"] + confidence) / 2.0
                        data["lift"] = (data.get("lift", 0) + lift) / 2.0
                    else:
                        G.add_edge(u, v, support=support, confidence=confidence, lift=lift)
    if G.number_of_edges() == 0:
        return None
    return G


def _write_empty_network_html(
    cohort_name: str,
    age_band: str,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Write a minimal HTML that states 'No rules for this cohort' so the dashboard
    can show a placeholder instead of a missing asset. Same filename as the real network.
    """
    age_band_fname = age_band.replace("-", "_")
    fname = f"{cohort_name}_{age_band_fname}_combined_rules_network.html"
    out_path = output_dir / fname
    _ensure_output_dir(output_dir)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>FP-Growth network — {cohort_name} {age_band}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 80vh; margin: 0; background: #f8fafc; }}
    .message {{ text-align: center; padding: 2rem; color: #64748b; font-size: 1.1rem; }}
  </style>
</head>
<body>
  <div class="message">No rules for this cohort.</div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    if logger:
        logger.info("Saved empty network placeholder (no rules) to %s", out_path)
    return out_path


def _network_combined_plotly_with_filter(
    G: Any,
    cohort_name: str,
    age_band: str,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    max_nodes: int = 80,
) -> Optional[Path]:
    """
    Build one Plotly network with nodes colored by type (drug_name, icd_code, cpt_code)
    and a dropdown to filter by type: All | Drug | ICD | CPT.
    """
    if not PLOTLY_AVAILABLE or G is None or G.number_of_edges() == 0:
        return None
    if G.number_of_nodes() > max_nodes:
        centrality = nx.degree_centrality(G)
        top = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
        G = G.subgraph(top).copy()
    pos = nx.spring_layout(G, seed=42, k=0.6, iterations=50)
    centrality = nx.degree_centrality(G)

    # Group nodes and edges by type(s) for filter
    nodes_by_type: Dict[str, List[str]] = {t: [] for t in FPGROWTH_GRAPH_ITEM_TYPES}
    for n in G.nodes():
        t = G.nodes[n].get("node_type", "drug_name")
        if t in nodes_by_type:
            nodes_by_type[t].append(n)
    edges_by_type_pair: Dict[tuple, List[tuple]] = {}
    for u, v in G.edges():
        tu = G.nodes[u].get("node_type", "drug_name")
        tv = G.nodes[v].get("node_type", "drug_name")
        key = (tu, tv)
        edges_by_type_pair.setdefault(key, []).append((u, v))

    # One node trace per type (for filter visibility)
    node_traces = []
    for item_type in FPGROWTH_GRAPH_ITEM_TYPES:
        nodes = nodes_by_type.get(item_type, [])
        if not nodes:
            continue
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        node_sizes = [15 + 35 * centrality.get(n, 0.0) for n in nodes]
        labels = [G.nodes[n].get("display_label", n.split("::", 1)[-1]) for n in nodes]
        hover = [
            f"<b>{ITEM_TYPE_LABELS.get(item_type, item_type)}:</b> {lb}<br>"
            f"<b>Centrality:</b> {centrality.get(n, 0):.4f}"
            for n, lb in zip(nodes, labels)
        ]
        node_traces.append(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=node_sizes,
                    color=ITEM_TYPE_COLORS.get(item_type, "#94a3b8"),
                    line=dict(width=1, color="#1e293b"),
                ),
                hoverinfo="text",
                hovertext=hover,
                name=ITEM_TYPE_LABELS.get(item_type, item_type),
                legendgrouptitle=dict(text="Filter by type"),
            )
        )

    # One edge trace per edge so width can be based on support (frequency)
    # Width = 0.5 + 6*support, clamped to [0.5, 4]
    edge_traces = []
    edge_trace_type: List[tuple] = []  # (tu, tv) per trace for filter
    for (tu, tv), pairs in edges_by_type_pair.items():
        for u, v in pairs:
            support = G[u][v].get("support", 0.0) or 0.0
            width = max(0.5, min(4.0, 0.5 + 6.0 * support))
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=width, color="#94a3b8"),
                    hoverinfo="text",
                    hovertext=f"Support: {support:.3f}",
                    mode="lines",
                    name=f"{ITEM_TYPE_LABELS.get(tu, tu)} → {ITEM_TYPE_LABELS.get(tv, tv)}",
                    showlegend=False,
                )
            )
            edge_trace_type.append((tu, tv))

    all_traces = edge_traces + node_traces
    n_edge_traces = len(edge_traces)
    n_node_traces = len(node_traces)
    type_to_node_idx = {FPGROWTH_GRAPH_ITEM_TYPES[i]: i for i in range(n_node_traces) if i < len(FPGROWTH_GRAPH_ITEM_TYPES)}
    # Which edge traces belong to which (tu, tv) for filter
    type_to_edge_indices: Dict[tuple, List[int]] = {}
    for i, tt in enumerate(edge_trace_type):
        type_to_edge_indices.setdefault(tt, []).append(i)

    # Dropdown: All, Drug, ICD, CPT
    buttons = [
        dict(
            label="All",
            method="update",
            args=[
                {"visible": [True] * len(all_traces)},
                {"title": f"{cohort_name} {age_band} — Association rules (All types)"},
            ],
        )
    ]
    for item_type in FPGROWTH_GRAPH_ITEM_TYPES:
        if item_type not in type_to_node_idx:
            continue
        vis = [False] * len(all_traces)
        ni = type_to_node_idx[item_type]
        vis[n_edge_traces + ni] = True
        for ei in type_to_edge_indices.get((item_type, item_type), []):
            vis[ei] = True
        buttons.append(
            dict(
                label=ITEM_TYPE_LABELS.get(item_type, item_type),
                method="update",
                args=[
                    {"visible": vis},
                    {"title": f"{cohort_name} {age_band} — Association rules ({ITEM_TYPE_LABELS.get(item_type, item_type)} only)"},
                ],
            )
        )

    layout = go.Layout(
        title=f"{cohort_name} {age_band} — Association rules (All types)",
        showlegend=True,
        legend=dict(orientation="h", y=1.12, xanchor="center", x=0.5),
        hovermode="closest",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.18,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#888",
                borderwidth=1,
            )
        ],
        annotations=[
            dict(
                text="<b>Filter by type:</b>",
                showarrow=False,
                x=0.02,
                y=1.18,
                xref="paper",
                yref="paper",
                align="left",
                font=dict(size=12),
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800,
        margin=dict(l=20, r=20, t=100, b=20),
    )
    fig = go.Figure(data=all_traces, layout=layout)
    out_path = output_dir / f"{cohort_name}_{age_band.replace('-', '_')}_combined_rules_network.html"
    _ensure_output_dir(output_dir)
    write_plotly_html_for_production(fig, out_path)
    if logger:
        logger.info("Saved combined rules network (filter by type) to %s", out_path)
    return out_path


def _load_multi_year_data(
    base_path: Path,
    cohort_name: str,
    age_band: str,
    item_type: str,
    split_type: str = "combined",
    logger: Optional[logging.Logger] = None,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Load itemsets and rules from multiple year directories (train/, 2016/, 2017/, 2018/).
    
    Returns:
        Dict mapping year -> {"itemsets": df, "rules": df}
        Year 0 represents "All Years" (train/ directory)
        Individual years 2016, 2017, 2018 map to their respective directories
    """
    age_band_fname = age_band.replace("-", "_")
    years_to_load = [
        (0, "train"),       # Year 0 = "All Years" from train/ directory
        (2016, "2016"),
        (2017, "2017"),
        (2018, "2018"),
    ]
    
    multi_year_data: Dict[int, Dict[str, pd.DataFrame]] = {}
    
    for year, year_dir in years_to_load:
        year_path = base_path / cohort_name / split_type / age_band_fname / year_dir
        
        # Load itemsets
        itemsets_file = f"{item_type}_itemsets_lift_filtered.json"
        itemsets_path = year_path / itemsets_file
        df_itemsets = _load_json_df(itemsets_path)
        
        # Load rules (for target split type)
        df_rules = pd.DataFrame()
        if split_type == "target":
            rules_file = f"{item_type}_rules_lift_filtered.json"
            rules_path = year_path / rules_file
            df_rules = _load_json_df(rules_path)
        
        # Only store if we have data
        if not df_itemsets.empty or not df_rules.empty:
            multi_year_data[year] = {
                "itemsets": df_itemsets,
                "rules": df_rules
            }
            if logger:
                logger.info(
                    "Loaded year %s (%s): %d itemsets, %d rules",
                    year if year != 0 else "All",
                    year_dir,
                    len(df_itemsets),
                    len(df_rules)
                )
        else:
            if logger:
                logger.warning("No data found for year %s (%s) at %s", year, year_dir, year_path)
    
    return multi_year_data


def _top_itemset_plot(
    df_itemsets: pd.DataFrame,
    cohort_name: str,
    age_band: str,
    item_type: str,
    top_n: int,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    code_mapping: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    if df_itemsets.empty or "support" not in df_itemsets.columns:
        return None

    # Derive a simple label for each itemset; use description when mapping exists for viewable labels
    def _label(items) -> str:
        if not isinstance(items, list):
            x = items
            return (code_mapping.get(str(x).strip(), str(x))) if code_mapping else str(x)
        parts = [
            (code_mapping.get(str(x).strip(), str(x)) if code_mapping else str(x))
            for x in items
        ]
        return ", ".join(parts)

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


def _top_itemsets_interactive(
    multi_year_data: Dict[int, Dict[str, pd.DataFrame]],
    cohort_name: str,
    age_band: str,
    item_type: str,
    top_n: int,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    Create an interactive Plotly horizontal bar chart showing top itemsets with year dropdown.
    
    Args:
        multi_year_data: Dict mapping year -> {"itemsets": df, "rules": df}
        cohort_name: Name of the cohort (opioid_ed, non_opioid_ed)
        age_band: Age band (e.g., "1-0-12", "1-13-24")
        item_type: Type of items for this graph (drug_name, icd_code, or cpt_code)
        top_n: Number of top itemsets to display
        output_dir: Directory to save the output HTML file
        logger: Logger instance
        
    Returns:
        Path to the generated HTML file, or None if no data
    """
    if not PLOTLY_AVAILABLE:
        if logger:
            logger.warning("Plotly not available; skipping interactive itemsets visualization")
        return None
    
    if not multi_year_data:
        return None
    
    # Helper function to create label for itemset
    def _label(items) -> str:
        if not isinstance(items, list):
            return str(items)
        return ", ".join(str(x) for x in items[:3])  # Limit to first 3 items for readability
    
    # Prepare data for each year
    traces = []
    years = sorted(multi_year_data.keys())
    
    for year in years:
        df_itemsets = multi_year_data[year]["itemsets"]
        if df_itemsets.empty or "support" not in df_itemsets.columns:
            continue
        
        df = df_itemsets.copy()
        df["label"] = df["itemsets"].apply(_label)
        df = df.sort_values("support", ascending=False).head(top_n)
        
        if df.empty:
            continue
        
        # Create hover text with full itemset details
        def _hover_text(row):
            items = row["itemsets"] if isinstance(row["itemsets"], list) else [row["itemsets"]]
            support = row["support"]
            return f"<b>Itemset:</b> {', '.join(str(x) for x in items)}<br><b>Support:</b> {support:.4f}"
        
        df["hover"] = df.apply(_hover_text, axis=1)
        
        year_label = "All Years (2016-2018)" if year == 0 else str(year)
        
        trace = go.Bar(
            x=df["support"],
            y=df["label"],
            orientation="h",
            name=year_label,
            visible=(year == 0),  # Only "All Years" visible by default
            marker=dict(color="steelblue"),
            hovertext=df["hover"],
            hoverinfo="text"
        )
        traces.append(trace)
    
    if not traces:
        if logger:
            logger.warning("No itemsets data available for interactive visualization")
        return None
    
    # Create dropdown menu buttons
    buttons = []
    for i, year in enumerate(years):
        year_label = "All Years (2016-2018)" if year == 0 else str(year)
        visible_list = [False] * len(traces)
        visible_list[i] = True
        
        button = dict(
            label=year_label,
            method="update",
            args=[
                {"visible": visible_list},
                {"title": f"{cohort_name} {age_band} {item_type} — Top {top_n} Itemsets ({year_label})"}
            ]
        )
        buttons.append(button)
    
    # Create layout with dropdown
    layout = go.Layout(
        title=f"{cohort_name} {age_band} {item_type} — Top {top_n} Itemsets (All Years)",
        xaxis=dict(title="Support", showgrid=True),
        yaxis=dict(title="Itemset", autorange="reversed"),  # Top itemset at top
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.08,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#888",
                borderwidth=1
            )
        ],
        annotations=[
            dict(
                text="<b>Year:</b>",
                showarrow=False,
                x=0.03,
                y=1.08,
                xref="paper",
                yref="paper",
                align="left",
                xanchor="left",
                yanchor="top",
                font=dict(size=12)
            )
        ],
        height=800,
        margin=dict(l=200, r=50, t=100, b=50),
        hovermode="closest"
    )
    
    fig = go.Figure(data=traces, layout=layout)
    
    # Save to HTML (production: single file, embedded Plotly.js — same pattern as BupaR selfcontained=TRUE)
    fname = f"{cohort_name}_{age_band.replace('-', '_')}_{item_type}_itemsets_interactive.html"
    out_path = output_dir / fname
    _ensure_output_dir(output_dir)
    write_plotly_html_for_production(fig, out_path)
    
    if logger:
        logger.info("Saved interactive itemsets visualization to %s", out_path)
    
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
    write_plotly_html_for_production(fig, out_path)
    if logger:
        logger.info("Saved Plotly network HTML to %s", out_path)
    return out_path


def _network_interactive_multi_year(
    multi_year_data: Dict[int, Dict[str, pd.DataFrame]],
    cohort_name: str,
    age_band: str,
    item_type: str,
    min_rules: int,
    max_nodes: int,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    Create an interactive Plotly network graph from association rules with year dropdown.
    
    Args:
        multi_year_data: Dict mapping year -> {"itemsets": df, "rules": df}
        cohort_name: Name of the cohort (opioid_ed, non_opioid_ed)
        age_band: Age band (e.g., "1-0-12", "1-13-24")
        item_type: Type of items for this graph (drug_name, icd_code, or cpt_code)
        min_rules: Minimum number of rules required to generate network
        max_nodes: Maximum number of nodes to include in the network
        output_dir: Directory to save the output HTML file
        logger: Logger instance
        
    Returns:
        Path to the generated HTML file, or None if no data
    """
    if not PLOTLY_AVAILABLE:
        if logger:
            logger.warning("Plotly not available; skipping interactive network visualization")
        return None
    
    if not multi_year_data:
        return None
    
    # Build network graphs for each year
    year_graphs = {}
    years = sorted(multi_year_data.keys())
    
    for year in years:
        df_rules = multi_year_data[year]["rules"]
        if df_rules.empty or len(df_rules) < min_rules:
            continue
        
        if "antecedents" not in df_rules.columns or "consequents" not in df_rules.columns:
            continue
        
        # Build directed graph
        G = nx.DiGraph()
        for _, row in df_rules.iterrows():
            ants = row["antecedents"]
            cons = row["consequents"]
            support = float(row.get("support", 0.0) or 0.0)
            confidence = float(row.get("confidence", 0.0) or 0.0)
            lift = float(row.get("lift", 0.0) or 0.0)
            
            if not isinstance(ants, list) or not isinstance(cons, list):
                continue
            
            for a in ants:
                for c in cons:
                    if not a or not c:
                        continue
                    if G.has_edge(a, c):
                        # Aggregate metrics by averaging
                        data = G[a][c]
                        data["support"] = (data["support"] + support) / 2.0
                        data["confidence"] = (data["confidence"] + confidence) / 2.0
                        data["lift"] = (data["lift"] + lift) / 2.0
                    else:
                        G.add_edge(a, c, support=support, confidence=confidence, lift=lift)
        
        if G.number_of_edges() > 0:
            # Limit nodes if network too large
            if G.number_of_nodes() > max_nodes:
                # Keep top nodes by degree centrality
                centrality = nx.degree_centrality(G)
                top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
                G = G.subgraph(top_nodes).copy()
            
            year_graphs[year] = G
    
    if not year_graphs:
        if logger:
            logger.warning("No network data available for interactive visualization")
        return None
    
    # Create traces for each year
    traces = []
    years_with_data = sorted(year_graphs.keys())
    
    for year in years_with_data:
        G = year_graphs[year]
        
        # Compute layout once per year
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
        centrality = nx.degree_centrality(G)
        
        # Create edge traces
        edge_x, edge_y = [], []
        edge_hover = []
        
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            data = G[u][v]
            hover = (
                f"<b>Rule:</b> {u} → {v}<br>"
                f"<b>Support:</b> {data['support']:.4f}<br>"
                f"<b>Confidence:</b> {data['confidence']:.4f}<br>"
                f"<b>Lift:</b> {data['lift']:.4f}"
            )
            edge_hover.append(hover)
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.5, color="#888"),
            hoverinfo="text",
            hovertext=[h for h in edge_hover for _ in range(3)],  # Repeat for each edge segment
            mode="lines",
            name="Rules",
            showlegend=False,
            visible=(year == 0)  # Only "All Years" visible by default
        )
        
        # Create node traces
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_sizes = [15 + 35 * centrality.get(n, 0.0) for n in G.nodes()]
        node_text = list(G.nodes())
        
        node_hover = []
        for node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            hover = (
                f"<b>Item:</b> {node}<br>"
                f"<b>In-degree:</b> {in_degree}<br>"
                f"<b>Out-degree:</b> {out_degree}<br>"
                f"<b>Centrality:</b> {centrality[node]:.4f}"
            )
            node_hover.append(hover)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=node_sizes,
                color="lightblue",
                line=dict(width=1, color="darkblue")
            ),
            hoverinfo="text",
            hovertext=node_hover,
            name="Items",
            showlegend=False,
            visible=(year == 0)  # Only "All Years" visible by default
        )
        
        traces.extend([edge_trace, node_trace])
    
    # Create dropdown menu buttons
    buttons = []
    for i, year in enumerate(years_with_data):
        year_label = "All Years (2016-2018)" if year == 0 else str(year)
        visible_list = [False] * len(traces)
        
        # Each year has 2 traces (edges and nodes)
        trace_idx = i * 2
        visible_list[trace_idx] = True      # edge trace
        visible_list[trace_idx + 1] = True  # node trace
        
        G = year_graphs[year]
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        button = dict(
            label=year_label,
            method="update",
            args=[
                {"visible": visible_list},
                {"title": f"{cohort_name} {age_band} {item_type} — Association Rules Network ({year_label})<br><sub>{n_nodes} items, {n_edges} rules</sub>"}
            ]
        )
        buttons.append(button)
    
    # Initial year for title
    initial_year = years_with_data[0]
    G_initial = year_graphs[initial_year]
    initial_label = "All Years (2016-2018)" if initial_year == 0 else str(initial_year)
    
    # Create layout with dropdown
    layout = go.Layout(
        title=f"{cohort_name} {age_band} {item_type} — Association Rules Network ({initial_label})<br><sub>{G_initial.number_of_nodes()} items, {G_initial.number_of_edges()} rules</sub>",
        showlegend=False,
        hovermode="closest",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#888",
                borderwidth=1
            )
        ],
        annotations=[
            dict(
                text="<b>Year:</b>",
                showarrow=False,
                x=0.03,
                y=1.15,
                xref="paper",
                yref="paper",
                align="left",
                xanchor="left",
                yanchor="top",
                font=dict(size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800,
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    fig = go.Figure(data=traces, layout=layout)
    
    # Save to HTML (production: single file, embedded Plotly.js)
    fname = f"{cohort_name}_{age_band.replace('-', '_')}_{item_type}_network_interactive.html"
    out_path = output_dir / fname
    _ensure_output_dir(output_dir)
    write_plotly_html_for_production(fig, out_path)
    
    if logger:
        logger.info("Saved interactive network visualization to %s", out_path)
    
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
    code_mapping_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create standard FP-Growth plots for a cohort / age_band.
    Writes one combined rules network (filter by type: Drug / ICD / CPT) and optional per-type itemsets.

    One graph: drug_name, icd_code, cpt_code as node types in a single network; filter by type in the UI.
    medical_code is the union of these three (allowed_codes); not a separate graph.

    Implemented:
      - Combined rules network: one HTML with dropdown "Filter by type" (All | Drug | ICD | CPT)
      - Per-type itemsets: top-N bar chart PNG per type (optional)

    Returns:
      Dict with:
        "plots": "combined" -> { target_rules_network_combined: Path }, and per item_type -> itemsets if any
        "s3_urls": (if s3_upload) same structure
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
        item_types = FPGROWTH_GRAPH_ITEM_TYPES

    base_path = Path(base_dir)
    age_band_fname = age_band.replace("-", "_")
    plots_root = Path(output_dir) if output_dir else base_path / "plots"

    results: Dict[str, Dict[str, Path]] = {}

    # Optional code mapping for viewable labels (code -> description)
    code_mapping_path_val = Path(code_mapping_path) if code_mapping_path else None
    code_mapping = _load_code_mapping(code_mapping_path_val)
    if code_mapping:
        logger.info("Loaded %d code descriptions for viewable labels", len(code_mapping))

    # One combined network: load rules for all three types, build one graph, save HTML with filter by type
    rules_by_type = _load_rules_all_types(
        base_path, cohort_name, age_band, split_type="target", event_year=event_year
    )
    G = _build_combined_rules_graph(
        rules_by_type, min_rules_per_type=2, code_mapping=code_mapping
    )
    combined_net = _network_combined_plotly_with_filter(
        G, cohort_name, age_band, plots_root, logger=logger
    )
    if combined_net is not None:
        results["combined"] = {"target_rules_network_combined": combined_net}
        logger.info("Created combined rules network (filter by type) for %s / %s", cohort_name, age_band)
    else:
        # Return a blank network HTML that states 'No rules for this cohort' so the dashboard can show it
        empty_path = _write_empty_network_html(cohort_name, age_band, plots_root, logger=logger)
        results["combined"] = {"target_rules_network_combined": empty_path}

    # Per-type itemsets (top-N bar chart): load from cohort/age_band (target-only or combined)
    for item_type in item_types:
        artifact_dir = base_path / cohort_name / age_band_fname
        itemsets_path = artifact_dir / f"{item_type}_itemsets_target_only.json"
        df_itemsets = _load_json_df(itemsets_path)
        if df_itemsets.empty:
            itemsets_path = artifact_dir / f"{item_type}_itemsets.json"
            df_itemsets = _load_json_df(itemsets_path)
        top_plot = _top_itemset_plot(
            df_itemsets=df_itemsets,
            cohort_name=cohort_name,
            age_band=age_band,
            item_type=item_type,
            top_n=top_n,
            output_dir=plots_root,
            logger=logger,
            code_mapping=code_mapping,
        )
        if top_plot is not None:
            results.setdefault(item_type, {})["combined_top_itemsets"] = top_plot

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

    # Upload empty-state JSON for dashboard when itemsets/rules were empty (written by run_single_cohort_fpgrowth)
    if s3_upload and s3_bucket:
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
        except ImportError:
            pass
        else:
            empty_state_path = plots_root / "empty_state.json"
            if empty_state_path.exists():
                s3_plot_prefix = f"{s3_prefix.rstrip('/')}/{cohort_name}/{age_band}/plots"
                key = f"{s3_plot_prefix}/empty_state.json"
                s3_path = f"s3://{s3_bucket}/{key}"
                logger = logging.getLogger("fpgrowth_plots")
                if upload_file_to_s3(empty_state_path, s3_path, logger=logger, check_exists=True):
                    out.setdefault("s3_urls", {})["_empty_state"] = _s3_public_url(s3_bucket, key)

    # Upload itemsets JSON for dashboard (client-side Plotly rendering; Lambda returns itemsets_data)
    if s3_upload and s3_bucket:
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
        except ImportError:
            pass
        else:
            artifact_dir = base_path / cohort_name / age_band_fname
            s3_data_prefix = f"{s3_prefix.rstrip('/')}/{cohort_name}/{age_band}/data"
            logger = logging.getLogger("fpgrowth_plots")
            for itype in item_types:
                for fname in (f"{itype}_itemsets_target_only.json", f"{itype}_itemsets.json"):
                    local_path = artifact_dir / fname
                    if not local_path.exists():
                        continue
                    key = f"{s3_data_prefix}/{itype}_itemsets.json"
                    s3_path = f"s3://{s3_bucket}/{key}"
                    if upload_file_to_s3(local_path, s3_path, logger=logger, check_exists=True):
                        break

    return out



def create_all_fpgrowth_plots_multi_year(
    base_dir: str,
    cohort_name: str,
    age_band: str,
    item_types: Optional[List[str]] = None,
    output_dir: str = "",
    s3_upload: bool = False,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "fpgrowth",
    top_n: int = 30,
    max_nodes: int = 50,
) -> Dict[str, Any]:
    """
    Create interactive FP-Growth visualizations with multi-year support (train/, 2016/, 2017/, 2018/).

    One graph per item type: drug_name, icd_code, cpt_code. medical_code is the union
    of these three and is not used for the graph.

    Generates:
      - Interactive itemsets bar chart with year dropdown (combined split)
      - Interactive network graph with year dropdown (target split)

    Args:
        base_dir: Root directory containing FP-Growth results
        cohort_name: Name of the cohort (opioid_ed, non_opioid_ed)
        age_band: Age band (e.g., "1-0-12", "1-13-24")
        item_types: Item types to plot (default: drug_name, icd_code, cpt_code only)
        output_dir: Directory to save output files (defaults to base_dir/plots)
        s3_upload: Whether to upload results to S3
        s3_bucket: S3 bucket name for uploads
        s3_prefix: S3 key prefix for uploads
        top_n: Number of top itemsets to display
        max_nodes: Maximum number of nodes in network graph

    Returns:
        Dict with:
            "plots": mapping item_type -> { plot_name: Path }
            "s3_urls": (if s3_upload) mapping item_type -> { plot_name: str URL }
    """
    # Setup logger
    logger = logging.getLogger("fpgrowth_plots_multi_year")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    if not item_types:
        item_types = FPGROWTH_GRAPH_ITEM_TYPES

    base_path = Path(base_dir)
    plots_root = Path(output_dir) if output_dir else base_path / "plots"

    results: Dict[str, Dict[str, Path]] = {}

    for item_type in item_types:
        logger.info(
            "Creating multi-year FP-Growth plots for %s / %s (%s)",
            cohort_name,
            age_band,
            item_type,
        )
        item_results: Dict[str, Path] = {}
        
        # Load multi-year data for combined itemsets
        multi_year_combined = _load_multi_year_data(
            base_path=base_path,
            cohort_name=cohort_name,
            age_band=age_band,
            item_type=item_type,
            split_type="combined",
            logger=logger
        )
        
        # Create interactive itemsets visualization
        if multi_year_combined:
            itemsets_html = _top_itemsets_interactive(
                multi_year_data=multi_year_combined,
                cohort_name=cohort_name,
                age_band=age_band,
                item_type=item_type,
                top_n=top_n,
                output_dir=plots_root,
                logger=logger
            )
            if itemsets_html is not None:
                item_results["itemsets_interactive"] = itemsets_html
        
        # Load multi-year data for target rules network
        multi_year_target = _load_multi_year_data(
            base_path=base_path,
            cohort_name=cohort_name,
            age_band=age_band,
            item_type=item_type,
            split_type="target",
            logger=logger
        )
        
        # Create interactive network visualization
        if multi_year_target:
            network_html = _network_interactive_multi_year(
                multi_year_data=multi_year_target,
                cohort_name=cohort_name,
                age_band=age_band,
                item_type=item_type,
                min_rules=5,
                max_nodes=max_nodes,
                output_dir=plots_root,
                logger=logger
            )
            if network_html is not None:
                item_results["network_interactive"] = network_html
        
        if item_results:
            results[item_type] = item_results
    
    out: Dict[str, Any] = {"plots": results}
    
    # Upload to S3 when requested
    if s3_upload and s3_bucket and results:
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
        except ImportError:
            logger.warning("checkpoint_utils not available; skipping S3 upload")
            s3_upload = False
    
    if s3_upload and s3_bucket and results:
        age_band_fname = age_band.replace("-", "_")
        s3_plot_prefix = f"{s3_prefix.rstrip('/')}/{cohort_name}/{age_band}/plots"
        s3_urls: Dict[str, Dict[str, str]] = {}
        
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

    # Upload empty-state JSON when itemsets/rules were empty
    if s3_upload and s3_bucket:
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
        except ImportError:
            pass
        else:
            empty_state_path = plots_root / "empty_state.json"
            if empty_state_path.exists():
                s3_plot_prefix = f"{s3_prefix.rstrip('/')}/{cohort_name}/{age_band}/plots"
                key = f"{s3_plot_prefix}/empty_state.json"
                s3_path = f"s3://{s3_bucket}/{key}"
                if upload_file_to_s3(empty_state_path, s3_path, logger=logger, check_exists=True):
                    out.setdefault("s3_urls", {})["_empty_state"] = _s3_public_url(s3_bucket, key)

    return out
