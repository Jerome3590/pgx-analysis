#!/usr/bin/env python3
"""
Generate publication-oriented Cohort PGx drug network visuals for the dashboard.

Reads NetworkX exports from:
  10_risk_dashboard/visualizations/cohort_pgx/networks/{cohort}/{age_band_fname}/

Writes dashboard-ready HTML/PNG outputs to:
  10_risk_dashboard/visualizations/cohort_pgx/figure_pack/

The figure pack adds clinical context around:
- dynamics
- kinetics
- allergic response
- underappreciated pathway signaling
- kinetic pathways
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COHORT_LABELS = {
    "opioid_ed": "Opioid ED",
    "non_opioid_ed": "Polypharmacy",
    "falls": "Falls",
    "ed": "ED",
}

TIER_COLORS = {
    "Tier 1": "#C0392B",
    "Tier 2": "#F39C12",
    "Tier 3": "#D4AC0D",
    "Unknown": "#8E44AD",
    "Undefined": "#8E44AD",
    "Drug": "#5DADE2",
    "Phenotype": "#95A5A6",
}

NODE_SYMBOLS = {"gene": "circle", "drug": "diamond", "phenotype": "square"}
RELATION_COLORS = {
    "feature_importance_drug_gene": "#7F8C8D",
    "metabolizes": "#7F8C8D",
    "co_metabolizes": "#2E86C1",
    "affects_risk": "#D81B60",
    "metabolic": "#AF7AC5",
    "inhibition": "#DC143C",
    "induction": "#32CD32",
    "combination": "#D4AC0D",
    "enhancement": "#FF6347",
}

KINETICS_GENES = {"ABCB1", "CES1", "CYP2C19", "CYP2D6", "CYP3A4", "CYP3A5", "SLCO1B1"}
ALLERGIC_RESPONSE_GENES = {"HLA-A", "HLA-B", "HLA-C", "HLA-DQA1", "HLA-DQB1", "HLA-DRB1"}
UNDERAPPRECIATED_SIGNALING_GENES = {
    "ADD1",
    "ADRA2C",
    "CETP",
    "GRK4",
    "GRK5",
    "HMGCR",
    "LPA",
    "NEDD4L",
    "PRKCA",
    "PTGFR",
    "YEATS4",
}
DYNAMIC_ANCHOR_DRUGS = {"CARVEDILOL", "FUROSEMIDE", "HYDROCHLOROTHIAZIDE", "SIMVASTATIN", "OMEPRAZOLE"}
CONTEXT_DEFINITIONS = {
    "dynamics": "Where the signal changes by cohort/age stratum and implies a prevention target.",
    "kinetics": "ADME and transporter genes that alter exposure, clearance, or active metabolite burden.",
    "allergic_response": "Immune/hypersensitivity context to monitor when HLA or allergy-linked PGx edges emerge.",
    "underappreciated_signaling": "Peripheral or Undefined genes that make sparse modules clinically interpretable.",
    "kinetic_pathways": "Drug -> kinetics gene -> exposure/timing pathway connected to lead-time before event.",
}
FIGURE_CONTEXT_NOTE = (
    "<b>Context:</b> dynamics = cohort/age shifts; kinetics = ADME exposure and clearance; "
    "allergic response = hypersensitivity watch-list; underappreciated signaling = Undefined/peripheral genes; "
    "kinetic pathways = drug -> gene -> exposure/timing chain."
)

CLUSTER_RULES = {
    "Adrenergic / beta-blocker": {
        "drugs": {"CARVEDILOL", "ATENOLOL"},
        "genes": {"ADRA2C", "ADRB1", "ADRB2", "CYP2D6", "GRK4", "GRK5"},
    },
    "Diuretic / hypertension": {
        "drugs": {"FUROSEMIDE", "HYDROCHLOROTHIAZIDE"},
        "genes": {"ADD1", "NEDD4L", "PRKCA", "YEATS4"},
    },
    "Lipid / statin": {
        "drugs": {"SIMVASTATIN"},
        "genes": {"ABCB1", "CETP", "CYP3A4", "CYP3A5", "HMGCR", "LPA", "SLCO1B1"},
    },
    "GI / antiplatelet / ophthalmic": {
        "drugs": {"OMEPRAZOLE", "CLOPIDOGREL", "LATANOPROST"},
        "genes": {"CYP2C19", "CES1", "PTGFR"},
    },
}


@dataclass(frozen=True)
class FigurePaths:
    html: Path
    png: Path


def find_headless_browser() -> Path | None:
    for candidate in [
        os.environ.get("CHROME_BIN"),
        os.environ.get("CHROMIUM_BIN"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "google-chrome",
        "chromium",
        "chromium-browser",
        "msedge",
    ]:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved)
    return None


def write_figure(fig: go.Figure, paths: FigurePaths, width: int = 1600, height: int = 1100) -> None:
    paths.html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(paths.html), include_plotlyjs="cdn")
    browser = find_headless_browser()
    if not browser:
        print(f"Saved HTML only; no Chromium browser found for PNG: {paths.html}")
        return
    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--virtual-time-budget=5000",
        f"--screenshot={paths.png.resolve()}",
        paths.html.resolve().as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if result.returncode != 0:
        print(f"PNG screenshot failed for {paths.html}: {(result.stderr or '')[-500:]}")
    elif paths.png.exists():
        print(f"Saved {paths.png}")


def context_tags_for_pair(gene: str, drug: str) -> list[str]:
    gene = str(gene).upper()
    drug = str(drug).upper()
    tags = []
    if drug in DYNAMIC_ANCHOR_DRUGS:
        tags.append("dynamics")
    if gene in KINETICS_GENES:
        tags.append("kinetics")
    if gene in ALLERGIC_RESPONSE_GENES:
        tags.append("allergic_response")
    if gene in UNDERAPPRECIATED_SIGNALING_GENES:
        tags.append("underappreciated_signaling")
    if gene in KINETICS_GENES and drug in DYNAMIC_ANCHOR_DRUGS:
        tags.append("kinetic_pathways")
    return tags or ["context_review"]


def primary_context_for_pair(gene: str, drug: str) -> str:
    tags = context_tags_for_pair(gene, drug)
    for tag in ["kinetic_pathways", "kinetics", "underappreciated_signaling", "dynamics", "allergic_response"]:
        if tag in tags:
            return tag
    return tags[0]


def add_context_annotation(fig: go.Figure, y: float = -0.08) -> None:
    fig.add_annotation(
        text=FIGURE_CONTEXT_NOTE,
        x=0,
        y=y,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        font=dict(size=11, color="#34495E"),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#D5D8DC",
        borderwidth=1,
        borderpad=6,
    )


def load_network_tables(networks_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_frames = []
    edge_frames = []
    for edges_path in sorted(networks_root.glob("*/*/network_edges.csv")):
        cohort = edges_path.parents[1].name
        age_band = edges_path.parent.name.replace("_", "-")
        nodes_path = edges_path.with_name("network_nodes.csv")
        if not nodes_path.exists():
            continue
        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)
        nodes["network_cohort"] = cohort
        nodes["network_age_band"] = age_band
        edges["network_cohort"] = cohort
        edges["network_age_band"] = age_band
        node_frames.append(nodes)
        edge_frames.append(edges)
    if not node_frames or not edge_frames:
        raise FileNotFoundError(f"No network_nodes.csv/network_edges.csv files found under {networks_root}")
    nodes = pd.concat(node_frames, ignore_index=True)
    edges = pd.concat(edge_frames, ignore_index=True)
    return harmonize_nodes(nodes), harmonize_edges(edges)


def harmonize_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()
    out["id"] = out["id"].astype(str)
    out["label"] = out.get("label", out["id"]).fillna(out["id"]).astype(str)
    out["type"] = out.get("type", out.get("node_type", "unknown")).fillna("unknown").astype(str)
    out["tier"] = out.get("tier", pd.Series(index=out.index, dtype=object)).fillna("Unknown").astype(str)
    out.loc[out["type"].eq("drug"), "tier"] = "Drug"
    out.loc[out["type"].eq("phenotype"), "tier"] = "Phenotype"
    degree = out["degree"] if "degree" in out.columns else pd.Series(1, index=out.index)
    out["degree"] = pd.to_numeric(degree, errors="coerce").fillna(1)
    seed_gene = out["seed_gene"] if "seed_gene" in out.columns else pd.Series(False, index=out.index)
    seed_drug = out["seed_drug"] if "seed_drug" in out.columns else pd.Series(False, index=out.index)
    out["seed_gene"] = seed_gene.fillna(False).astype(bool)
    out["seed_drug"] = seed_drug.fillna(False).astype(bool)
    return out


def harmonize_edges(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges.copy()
    out["source"] = out["source"].astype(str)
    out["target"] = out["target"].astype(str)
    out["relation"] = out["relation"].fillna("related").astype(str)
    out["weight"] = pd.to_numeric(out.get("weight", 1.0), errors="coerce").fillna(1.0)
    out["feature_importance"] = pd.to_numeric(out.get("feature_importance", out["weight"]), errors="coerce")
    rank = out["rank"] if "rank" in out.columns else pd.Series(pd.NA, index=out.index)
    out["rank"] = pd.to_numeric(rank, errors="coerce")
    seed_edge = out["seed_edge"] if "seed_edge" in out.columns else out["relation"].isin(["feature_importance_drug_gene", "metabolizes"])
    out["seed_edge"] = seed_edge
    out["seed_edge"] = out["seed_edge"].fillna(False).astype(bool)
    out["cohort"] = out.get("cohort", out["network_cohort"]).fillna(out["network_cohort"])
    out["age_band"] = out.get("age_band", out["network_age_band"]).fillna(out["network_age_band"])
    out["outcome"] = out["cohort"].map(COHORT_LABELS).fillna(out["cohort"])
    out["panel"] = out["outcome"].astype(str) + " " + out["age_band"].astype(str)
    return out


def aggregate_node_table(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    node_ids = pd.unique(pd.concat([edges["source"], edges["target"]], ignore_index=True))
    base = (
        nodes.sort_values(["seed_gene", "seed_drug", "degree"], ascending=[False, False, False])
        .drop_duplicates("id")
        .set_index("id")
        .reindex(node_ids)
        .reset_index()
        .rename(columns={"index": "id"})
    )
    base["label"] = base["label"].fillna(base["id"])
    base["type"] = base["type"].fillna("unknown")
    base["tier"] = base["tier"].fillna("Unknown")
    base["degree"] = base["id"].map(edges["source"].value_counts().add(edges["target"].value_counts(), fill_value=0)).fillna(0)
    return base


def should_label_node(node_id: str, attrs: pd.Series) -> bool:
    highlight = DYNAMIC_ANCHOR_DRUGS | {"CYP2D6", "CYP3A4", "SLCO1B1", "ABCB1", "ADD1", "ADRA2C"}
    return node_id in highlight or attrs.get("tier") in {"Tier 1", "Undefined", "Unknown"} or bool(attrs.get("seed_gene", False))


def add_network_traces(fig: go.Figure, nodes: pd.DataFrame, edges: pd.DataFrame, row=None, col=None, showlegend=True, title_prefix="") -> None:
    graph = nx.Graph()
    for _, node in nodes.iterrows():
        graph.add_node(node["id"])
    for _, edge in edges.iterrows():
        graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
    if graph.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(graph, seed=42, k=1.2, iterations=80)
    for relation, relation_edges in edges.groupby("relation"):
        x_vals, y_vals = [], []
        edge_width = max(0.8, min(5.0, float(relation_edges["feature_importance"].fillna(0.1).max()) * 10))
        for _, edge in relation_edges.iterrows():
            if edge["source"] not in pos or edge["target"] not in pos:
                continue
            x0, y0 = pos[edge["source"]]
            x1, y1 = pos[edge["target"]]
            x_vals.extend([x0, x1, None])
            y_vals.extend([y0, y1, None])
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color=RELATION_COLORS.get(relation, "#BBBBBB"), width=edge_width),
                opacity=0.85 if relation in {"feature_importance_drug_gene", "metabolizes"} else 0.35,
                hoverinfo="skip",
                name=f"{title_prefix}{relation}",
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
    for (node_type, tier), group in nodes.groupby(["type", "tier"], dropna=False):
        ids = [node_id for node_id in group["id"] if node_id in pos]
        if not ids:
            continue
        group = group.set_index("id").loc[ids].reset_index()
        labels = [node_id if should_label_node(node_id, attrs) else "" for node_id, attrs in group.set_index("id").iterrows()]
        sizes = [14 + min(float(attrs.get("degree", 1) or 1), 12) * 2.5 for _, attrs in group.iterrows()]
        fig.add_trace(
            go.Scatter(
                x=[pos[node_id][0] for node_id in ids],
                y=[pos[node_id][1] for node_id in ids],
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=TIER_COLORS.get(tier, "#8E44AD"),
                    symbol=NODE_SYMBOLS.get(node_type, "circle"),
                    line=dict(width=1.5, color="white"),
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=10),
                hovertext=[f"{r.id}<br>type={r.type}<br>tier={r.tier}<br>degree={r.degree}" for r in group.itertuples()],
                hoverinfo="text",
                name=f"{title_prefix}{tier} {node_type}",
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )


def seed_edges(edges: pd.DataFrame) -> pd.DataFrame:
    seed = edges[edges["relation"].isin(["feature_importance_drug_gene", "metabolizes"])].copy()
    if seed.empty:
        seed = edges.copy()
    return seed.sort_values(["rank", "feature_importance", "weight"], ascending=[True, False, False])


def make_global_network(nodes: pd.DataFrame, edges: pd.DataFrame, out_dir: Path) -> None:
    graph_edges = seed_edges(edges).head(180)
    graph_nodes = aggregate_node_table(nodes, graph_edges)
    fig = go.Figure()
    add_network_traces(fig, graph_nodes, graph_edges)
    fig.update_layout(
        title="PGx Intervention-Weighted Global Network<br><sup>Drug-gene edges emphasize dynamics, kinetics, pathway signaling, and exposure context.</sup>",
        width=1500,
        height=1000,
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="v", x=1.02, y=1),
        margin=dict(l=20, r=260, t=90, b=95),
    )
    add_context_annotation(fig)
    write_figure(fig, FigurePaths(out_dir / "pgx_global_intervention_network.html", out_dir / "pgx_global_intervention_network.png"))


def make_cohort_small_multiples(nodes: pd.DataFrame, edges: pd.DataFrame, out_dir: Path) -> None:
    panels = sorted(edges[["cohort", "age_band"]].drop_duplicates().itertuples(index=False, name=None))
    panels = panels[:4] if len(panels) > 4 else panels
    rows = 2 if len(panels) > 2 else 1
    cols = 2 if len(panels) > 1 else 1
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{COHORT_LABELS.get(c, c)} {a}" for c, a in panels],
        horizontal_spacing=0.03,
        vertical_spacing=0.1,
    )
    for idx, (cohort, age_band) in enumerate(panels):
        row = idx // cols + 1
        col = idx % cols + 1
        panel_edges = seed_edges(edges[edges["cohort"].eq(cohort) & edges["age_band"].eq(age_band)]).head(45)
        add_network_traces(fig, aggregate_node_table(nodes, panel_edges), panel_edges, row=row, col=col, showlegend=(idx == 0), title_prefix=f"{cohort}-{age_band} ")
    fig.update_layout(
        title="Cohort-Specific PGx Network Small Multiples<br><sup>Dynamics are visible as shifts in drug-gene modules across cohorts and age bands.</sup>",
        width=1600,
        height=1150,
        plot_bgcolor="white",
        margin=dict(l=20, r=240, t=105, b=95),
    )
    for axis in fig.layout:
        if str(axis).startswith("xaxis") or str(axis).startswith("yaxis"):
            fig.layout[axis].visible = False
    add_context_annotation(fig)
    write_figure(fig, FigurePaths(out_dir / "pgx_cohort_small_multiples.html", out_dir / "pgx_cohort_small_multiples.png"), height=1200)


def make_cluster_ego_networks(nodes: pd.DataFrame, edges: pd.DataFrame, out_dir: Path) -> None:
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(CLUSTER_RULES.keys()), horizontal_spacing=0.03, vertical_spacing=0.08)
    for idx, (cluster, rule) in enumerate(CLUSTER_RULES.items()):
        row = idx // 2 + 1
        col = idx % 2 + 1
        keep = rule["drugs"] | rule["genes"]
        cluster_edges = seed_edges(edges[edges["source"].isin(keep) | edges["target"].isin(keep)]).head(60)
        add_network_traces(fig, aggregate_node_table(nodes, cluster_edges), cluster_edges, row=row, col=col, showlegend=(idx == 0), title_prefix=f"{cluster} ")
    fig.update_layout(
        title="Therapeutic Cluster Ego Networks<br><sup>Clusters separate kinetic pathways from underappreciated signaling modules.</sup>",
        width=1600,
        height=1150,
        plot_bgcolor="white",
        margin=dict(l=20, r=240, t=105, b=95),
    )
    for axis in fig.layout:
        if str(axis).startswith("xaxis") or str(axis).startswith("yaxis"):
            fig.layout[axis].visible = False
    add_context_annotation(fig)
    write_figure(fig, FigurePaths(out_dir / "pgx_cluster_ego_networks.html", out_dir / "pgx_cluster_ego_networks.png"), height=1200)


def intervention_priority(edges: pd.DataFrame) -> pd.DataFrame:
    priority = seed_edges(edges).dropna(subset=["feature_importance"]).copy()
    priority["drug"] = priority["target"]
    priority["gene"] = priority["source"]
    priority["rank"] = priority["rank"].fillna(priority.groupby(["cohort", "age_band"]).cumcount() + 1)
    priority["importance_norm"] = priority.groupby(["cohort", "age_band"])["feature_importance"].transform(lambda s: s / s.max() if s.max() else s)
    priority["inv_rank_norm"] = (1 / priority["rank"]).groupby([priority["cohort"], priority["age_band"]]).transform(lambda s: s / s.max() if s.max() else s)
    priority["tier_weight"] = priority["gene"].isin(UNDERAPPRECIATED_SIGNALING_GENES).map({True: 1.2, False: 1.0})
    priority["context_tags"] = priority.apply(lambda r: ";".join(context_tags_for_pair(r["gene"], r["drug"])), axis=1)
    priority["primary_context"] = priority.apply(lambda r: primary_context_for_pair(r["gene"], r["drug"]), axis=1)
    priority["intervention_priority"] = (
        priority["importance_norm"].fillna(0) * 0.5
        + priority["inv_rank_norm"].fillna(0) * 0.3
        + ((priority["tier_weight"] - 1.0) / 0.2).fillna(0) * 0.2
    )
    return priority


def make_priority_heatmap(edges: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    priority = intervention_priority(edges)
    if priority.empty:
        return priority
    top = priority.sort_values("intervention_priority", ascending=False).groupby(["cohort", "age_band"]).head(12).copy()
    top["pair_context"] = top["gene"] + " -> " + top["drug"] + " [" + top["primary_context"] + "]"
    pivot = top.pivot_table(index="pair_context", columns="panel", values="intervention_priority", aggfunc="max", fill_value=0)
    fig = go.Figure(go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index), colorscale="Reds", colorbar=dict(title="Priority")))
    fig.update_layout(
        title="PGx Intervention Priority Heatmap<br><sup>Score combines importance, inverse rank, and pathway-context tags.</sup>",
        width=1200,
        height=max(650, 28 * len(pivot.index)),
        margin=dict(l=320, r=40, t=100, b=130),
    )
    add_context_annotation(fig, y=-0.18)
    write_figure(fig, FigurePaths(out_dir / "pgx_intervention_priority_heatmap.html", out_dir / "pgx_intervention_priority_heatmap.png"), width=1300, height=max(800, 30 * len(pivot.index)))
    return priority


def make_pathway_context_panel(priority: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in priority.iterrows():
        for tag in str(row.get("context_tags", "context_review")).split(";"):
            rows.append({"context": tag, **row.to_dict()})
    context_df = pd.DataFrame(rows)
    if context_df.empty:
        return context_df
    totals = context_df.groupby("context").agg(edge_count=("drug", "size"), max_priority=("intervention_priority", "max")).reset_index()
    counts = context_df.groupby(["context", "panel"]).size().reset_index(name="edge_count")
    heat = counts.pivot_table(index="context", columns="panel", values="edge_count", fill_value=0)
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "bar"}, {"type": "heatmap"}], [{"type": "table", "colspan": 2}, None]], subplot_titles=["Context frequency", "Context by cohort/age panel", "Interpretive context key", ""])
    fig.add_trace(go.Bar(x=totals["context"], y=totals["edge_count"], marker_color="#5DADE2"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=heat.values, x=list(heat.columns), y=list(heat.index), colorscale="Blues", colorbar=dict(title="Edges")), row=1, col=2)
    fig.add_trace(
        go.Table(
            header=dict(values=["Context", "Meaning"], fill_color="#D6EAF8", align="left"),
            cells=dict(values=[list(CONTEXT_DEFINITIONS.keys()), list(CONTEXT_DEFINITIONS.values())], align="left", height=28),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title="PGx Pathway Context Panel<br><sup>Dynamics, kinetics, allergic response, underappreciated signaling, and kinetic pathways.</sup>",
        width=1500,
        height=1000,
        margin=dict(l=60, r=60, t=110, b=60),
        showlegend=False,
    )
    write_figure(fig, FigurePaths(out_dir / "pgx_pathway_context_panel.html", out_dir / "pgx_pathway_context_panel.png"), width=1600, height=1100)
    return context_df


def make_time_to_event_panel(out_dir: Path) -> pd.DataFrame:
    time_df = pd.DataFrame(
        [
            {"cohort": "opioid_ed", "age_band": "65-74", "drug": "FUROSEMIDE", "median_days_before_event": 25.5, "window_low": 21, "window_high": 42},
            {"cohort": "non_opioid_ed", "age_band": "75-84", "drug": "FUROSEMIDE", "median_days_before_event": 37.0, "window_low": 21, "window_high": 42},
        ]
    )
    time_df["panel"] = time_df["cohort"].map(COHORT_LABELS).fillna(time_df["cohort"]) + " " + time_df["age_band"]
    fig = go.Figure()
    for _, row in time_df.iterrows():
        fig.add_trace(go.Scatter(x=[row["window_low"], row["window_high"]], y=[row["panel"], row["panel"]], mode="lines", line=dict(color="#AED6F1", width=18), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[row["median_days_before_event"]], y=[row["panel"]], mode="markers+text", marker=dict(size=14, color="#1F618D"), text=[f"{row['drug']} ({row['median_days_before_event']} d)"], textposition="top center", showlegend=False))
    fig.update_layout(
        title="Medication Lead-Time Before Event<br><sup>Connects kinetics and kinetic pathways to dynamics: when a medication-review signal appears before outcome.</sup>",
        width=1100,
        height=500,
        xaxis=dict(title="Days before event", autorange="reversed"),
        yaxis=dict(title=""),
        plot_bgcolor="white",
        margin=dict(l=130, r=40, t=100, b=120),
    )
    add_context_annotation(fig, y=-0.25)
    write_figure(fig, FigurePaths(out_dir / "pgx_time_to_event_panel.html", out_dir / "pgx_time_to_event_panel.png"), width=1200, height=650)
    return time_df


def generate_figure_pack(project_root: Path) -> None:
    networks_root = project_root / "10_risk_dashboard" / "visualizations" / "cohort_pgx" / "networks"
    out_dir = project_root / "10_risk_dashboard" / "visualizations" / "cohort_pgx" / "figure_pack"
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes, edges = load_network_tables(networks_root)
    make_global_network(nodes, edges, out_dir)
    make_cohort_small_multiples(nodes, edges, out_dir)
    make_cluster_ego_networks(nodes, edges, out_dir)
    priority = make_priority_heatmap(edges, out_dir)
    context_df = make_pathway_context_panel(priority, out_dir)
    time_df = make_time_to_event_panel(out_dir)
    priority.to_csv(out_dir / "pgx_intervention_priority_scores.csv", index=False)
    context_df.to_csv(out_dir / "pgx_pathway_context_edges.csv", index=False)
    time_df.to_csv(out_dir / "pgx_time_to_event_windows.csv", index=False)
    manifest = {
        "description": "Publication-oriented PGx network figure pack.",
        "figures": [p.name for p in sorted(out_dir.glob("*.html")) + sorted(out_dir.glob("*.png"))],
        "priority_rows": int(len(priority)),
        "pathway_context_rows": int(len(context_df)),
        "time_window_rows": int(len(time_df)),
        "pathway_context_definitions": CONTEXT_DEFINITIONS,
    }
    with open(out_dir / "figure_pack_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Figure pack written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cohort PGx publication figure pack.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root.")
    args = parser.parse_args()
    generate_figure_pack(args.project_root.resolve())


if __name__ == "__main__":
    main()
