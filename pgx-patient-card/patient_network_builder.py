#!/usr/bin/env python3
"""
Patient-Specific PGx Network Builder

Creates personalized multi-layer network topology visualizations for individual patients
based on their genotype data. Similar to Cohort PGx network but filtered to patient's
specific variants and actionable genes.

Key Features:
- Patient genotype-driven filtering (only show genes with variants)
- Risk-level coloring (high risk = red, moderate = orange, normal = green)
- Phenotype-weighted edges (poor metabolizer = thicker edges)
- Medication list integration (highlight drugs patient is taking)
- Interactive filtering by risk level, CPIC guidelines, current meds
- Drug-drug interaction warnings for patient's medication list
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# Try to import spacy + pytextrank
try:
    import spacy
    import pytextrank
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spacy/pytextrank not available. Install with: pip install spacy pytextrank")

# Try to import boto3 for AWS Comprehend
try:
    import boto3
    COMPREHEND_AVAILABLE = True
except ImportError:
    COMPREHEND_AVAILABLE = False


# Phenotype/risk level mappings
METABOLIZER_PHENOTYPES = {
    "poor metabolizer": "high_risk",
    "intermediate metabolizer": "moderate_risk",
    "rapid metabolizer": "moderate_risk",
    "ultra-rapid metabolizer": "high_risk",
    "normal metabolizer": "normal",
    "extensive metabolizer": "normal",
}

RISK_COLORS = {
    "high_risk": "#DC143C",      # Crimson red
    "moderate_risk": "#FF8C00",  # Dark orange
    "normal": "#32CD32",         # Lime green
    "unknown": "#808080"         # Gray
}

CPIC_LEVEL_COLORS = {
    "A": "#8B0000",  # Dark red - Prescribing change recommended
    "B": "#FF4500",  # Orange red - Prescribing change may be considered
    "C": "#FFA500",  # Orange - Optional prescribing change
    "D": "#FFD700",  # Gold - Limited evidence
    "Unknown": "#D3D3D3"  # Light gray
}


class PatientPGxNetworkBuilder:
    """Build patient-specific PGx network from genotype and medication data."""
    
    def __init__(
        self,
        patient_variants: List[Dict],
        vip_reports: Dict,
        current_medications: Optional[List[str]] = None,
        cpic_data: Optional[pd.DataFrame] = None,
        use_comprehend: bool = False
    ):
        """
        Initialize patient network builder.
        
        Args:
            patient_variants: List of patient's genetic variants
                [{"gene": "CYP2D6", "variant": "*4/*4", "phenotype": "Poor Metabolizer", "impact": "High"}]
            vip_reports: Dict mapping gene symbol -> VIP report data
            current_medications: Optional list of patient's current medications
            cpic_data: Optional CPIC gene-drug pairs DataFrame
            use_comprehend: Whether to use AWS Comprehend (requires boto3)
        """
        self.patient_variants = patient_variants
        self.vip_reports = vip_reports
        self.current_medications = set(med.lower() for med in (current_medications or []))
        self.cpic_data = cpic_data
        self.use_comprehend = use_comprehend and COMPREHEND_AVAILABLE
        
        # Build variant lookup
        self.patient_genes = {}  # gene -> {variant, phenotype, risk_level}
        for var in patient_variants:
            gene = var.get("gene", "").upper()
            phenotype = var.get("phenotype", "Unknown").lower()
            
            # Determine risk level from phenotype
            risk_level = "unknown"
            for pheno_term, risk in METABOLIZER_PHENOTYPES.items():
                if pheno_term in phenotype:
                    risk_level = risk
                    break
            
            # Override with explicit impact if available
            impact = var.get("impact", "").lower()
            if "high" in impact:
                risk_level = "high_risk"
            elif "moderate" in impact or "medium" in impact:
                risk_level = "moderate_risk"
            
            self.patient_genes[gene] = {
                "variant": var.get("variant", "Unknown"),
                "phenotype": var.get("phenotype", "Unknown"),
                "risk_level": risk_level,
                "diplotype": var.get("diplotype", ""),
                "function": var.get("function", "")
            }
        
        # Initialize network components
        self.graph = nx.Graph()
        self.entities = defaultdict(set)
        self.relationships = []
        self.drug_interactions = []
        self.actionable_drugs = set()  # Drugs affected by patient's variants
        self.high_risk_phenotypes = set()  # Phenotypes patient is at risk for
        
        # Initialize spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                if "textrank" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("textrank")
            except OSError:
                print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        
        # Initialize Comprehend if requested
        self.comprehend_client = None
        if self.use_comprehend:
            try:
                self.comprehend_client = boto3.client("comprehend", region_name="us-east-1")
            except Exception as e:
                print(f"Warning: Could not initialize AWS Comprehend: {e}")
                self.use_comprehend = False
    
    def extract_drugs_from_text(self, text: str) -> Set[str]:
        """Extract drug names from VIP text."""
        drugs = set()
        
        # Common drug name patterns
        drug_patterns = [
            r'\b([A-Z][a-z]+(?:ine|ol|pril|sartan|statin|mab|ib|tin|mine|ide|zole|pam|done|morph))\b',
            r'\b(codeine|morphine|oxycodone|hydrocodone|tramadol|fentanyl)\b',
            r'\b(warfarin|clopidogrel|aspirin|heparin)\b',
            r'\b(simvastatin|atorvastatin|pravastatin|rosuvastatin)\b',
            r'\b(citalopram|escitalopram|sertraline|fluoxetine|paroxetine|venlafaxine)\b',
            r'\b(omeprazole|esomeprazole|lansoprazole|pantoprazole)\b',
            r'\b(metoprolol|carvedilol|atenolol|propranolol)\b',
        ]
        
        text_lower = text.lower()
        for pattern in drug_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 4:
                    drugs.add(match.title())
        
        return drugs
    
    def extract_phenotypes_from_text(self, text: str, gene: str) -> Set[str]:
        """Extract adverse events/phenotypes from VIP text."""
        phenotypes = set()
        
        # If patient has high-risk variant, look for risk-related phenotypes
        if gene in self.patient_genes and self.patient_genes[gene]["risk_level"] == "high_risk":
            # Pattern: "increased risk of X", "may cause X", "associated with X"
            risk_patterns = [
                r'(?:risk of|incidence of|occurrence of)\s+([a-z\s-]{5,40})',
                r'(?:may cause|can cause|causes)\s+([a-z\s-]{5,40})',
                r'(?:associated with|linked to)\s+([a-z\s-]{5,40})',
                r'([a-z\s-]{5,40})\s+(?:risk|adverse event)',
            ]
            
            text_lower = text.lower()
            for pattern in risk_patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    phenotype = match.strip().strip(',;').strip()
                    if 5 < len(phenotype) < 50:
                        phenotypes.add(phenotype.title())
        
        # Specific adverse events to look for
        specific_aes = [
            "Bleeding", "Thrombosis", "Respiratory Depression", "Sedation",
            "Liver Toxicity", "Myopathy", "QT Prolongation", "Serotonin Syndrome",
            "Nausea", "Dizziness", "Hypersensitivity", "Rash"
        ]
        
        text_lower = text.lower()
        for ae in specific_aes:
            if ae.lower() in text_lower:
                phenotypes.add(ae)
        
        return phenotypes
    
    def build_patient_network(self) -> nx.Graph:
        """Build network focusing on patient's specific variants."""
        print(f"\n{'='*80}")
        print(f"Building patient-specific PGx network")
        print(f"Patient variants: {len(self.patient_genes)} genes")
        print(f"Current medications: {len(self.current_medications)}")
        print(f"{'='*80}\n")
        
        # Process VIP reports for patient's genes only
        for gene_symbol in self.patient_genes.keys():
            if gene_symbol not in self.vip_reports:
                print(f"Warning: No VIP report for {gene_symbol}")
                continue
            
            report = self.vip_reports[gene_symbol]
            patient_info = self.patient_genes[gene_symbol]
            
            print(f"Processing {gene_symbol} ({patient_info['phenotype']})...", end=" ")
            
            # Add gene node with patient-specific metadata
            risk_level = patient_info["risk_level"]
            self.graph.add_node(
                gene_symbol,
                node_type="gene",
                label=gene_symbol,
                variant=patient_info["variant"],
                phenotype=patient_info["phenotype"],
                risk_level=risk_level,
                tier=report.get("vip_tier", "Unknown"),
                cpic_gene=report.get("cpic_gene", False),
                url=report.get("vip_url", "")
            )
            self.entities["gene"].add(gene_symbol)
            
            # Extract text from VIP report
            vip_text = report.get("vip_summary_text", "")
            if not vip_text:
                print("✗ No text")
                continue
            
            # Extract drugs
            drugs = self.extract_drugs_from_text(vip_text)
            self.entities["drug"].update(drugs)
            
            # Check if any are current medications
            current_drugs = {d for d in drugs if d.lower() in self.current_medications}
            
            # Extract phenotypes (focus on high-risk variants)
            phenotypes = self.extract_phenotypes_from_text(vip_text, gene_symbol)
            if risk_level == "high_risk":
                self.high_risk_phenotypes.update(phenotypes)
            self.entities["phenotype"].update(phenotypes)
            
            # Calculate edge weights based on risk level
            risk_weight = {
                "high_risk": 1.0,
                "moderate_risk": 0.6,
                "normal": 0.3,
                "unknown": 0.5
            }[risk_level]
            
            # Add gene -> drug edges
            for drug in drugs:
                self.graph.add_node(
                    drug,
                    node_type="drug",
                    label=drug,
                    in_current_meds=(drug.lower() in self.current_medications)
                )
                self.graph.add_edge(
                    gene_symbol, drug,
                    relation="metabolizes",
                    weight=risk_weight,
                    patient_risk=risk_level
                )
                self.relationships.append((gene_symbol, drug, "metabolizes", risk_weight, patient_info["phenotype"]))
                
                # Mark as actionable if patient has high/moderate risk
                if risk_level in ["high_risk", "moderate_risk"]:
                    self.actionable_drugs.add(drug)
            
            # Add gene -> phenotype edges (only for at-risk phenotypes)
            for phenotype in phenotypes:
                self.graph.add_node(
                    phenotype,
                    node_type="phenotype",
                    label=phenotype
                )
                self.graph.add_edge(
                    gene_symbol, phenotype,
                    relation="affects_risk",
                    weight=risk_weight,
                    patient_risk=risk_level
                )
                self.relationships.append((gene_symbol, phenotype, "affects_risk", risk_weight, f"{patient_info['phenotype']} → {phenotype} risk"))
            
            print(f"✓ {len(drugs)} drugs ({len(current_drugs)} current), {len(phenotypes)} phenotypes")
        
        # Add gene-gene relationships (co-metabolizes)
        self._add_gene_relationships()
        
        # Extract drug-drug interactions for current medications
        if self.current_medications:
            self._extract_drug_interactions()
        
        # Get CPIC recommendations if data available
        if self.cpic_data is not None:
            self._add_cpic_recommendations()
        
        print(f"\n{'='*80}")
        print(f"Patient network built:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Patient genes: {len(self.patient_genes)}")
        print(f"  Affected drugs: {len(self.entities['drug'])}")
        print(f"  Current medications: {len([n for n in self.graph.nodes() if self.graph.nodes[n].get('node_type') == 'drug' and self.graph.nodes[n].get('in_current_meds')])}")
        print(f"  Actionable drugs: {len(self.actionable_drugs)}")
        print(f"  Risk phenotypes: {len(self.high_risk_phenotypes)}")
        print(f"  Drug-drug interactions: {len(self.drug_interactions)}")
        print(f"{'='*80}\n")
        
        return self.graph
    
    def _add_gene_relationships(self):
        """Add edges between genes that affect the same drugs."""
        gene_drugs = defaultdict(set)
        for gene in self.entities["gene"]:
            if gene in self.graph:
                for neighbor in self.graph.neighbors(gene):
                    if self.graph.nodes[neighbor].get("node_type") == "drug":
                        gene_drugs[gene].add(neighbor)
        
        # Find genes with shared drugs
        genes = list(gene_drugs.keys())
        for i, gene1 in enumerate(genes):
            for gene2 in genes[i+1:]:
                shared_drugs = gene_drugs[gene1] & gene_drugs[gene2]
                if len(shared_drugs) >= 1:
                    # Weight by number of shared drugs
                    weight = min(len(shared_drugs) / 5.0, 1.0)
                    self.graph.add_edge(
                        gene1, gene2,
                        relation="co_metabolizes",
                        weight=weight,
                        shared_drugs=", ".join(list(shared_drugs)[:3])
                    )
    
    def _extract_drug_interactions(self):
        """Extract drug-drug interactions from patient's medication list."""
        current_drugs = [n for n in self.graph.nodes() if self.graph.nodes[n].get("in_current_meds")]
        
        for i, drug1 in enumerate(current_drugs):
            for drug2 in current_drugs[i+1:]:
                # Check if interaction mentioned in VIP texts
                interaction_found = False
                interaction_type = "metabolic"
                evidence = ""
                
                # Check all VIP reports for interaction mention
                for vip_report in self.vip_reports.values():
                    vip_text = vip_report.get("vip_summary_text", "")
                    if not vip_text:
                        continue
                    
                    # Look for both drugs mentioned together
                    if drug1.lower() in vip_text.lower() and drug2.lower() in vip_text.lower():
                        # Look for interaction keywords
                        interaction_patterns = [
                            (r'inhibit', 'inhibition'),
                            (r'induce', 'induction'),
                            (r'enhance', 'enhancement'),
                            (r'interact', 'metabolic'),
                        ]
                        
                        for pattern, itype in interaction_patterns:
                            if re.search(pattern, vip_text.lower()):
                                interaction_found = True
                                interaction_type = itype
                                # Extract evidence snippet
                                match = re.search(f'.{{0,100}}{pattern}.{{0,100}}', vip_text.lower())
                                if match:
                                    evidence = match.group(0).strip()
                                break
                        
                        if interaction_found:
                            break
                
                if interaction_found:
                    self.graph.add_edge(
                        drug1, drug2,
                        relation=interaction_type,
                        weight=0.8,
                        evidence=evidence[:200],
                        alert=True  # Flag for patient warning
                    )
                    self.drug_interactions.append((drug1, drug2, interaction_type, evidence))
    
    def _add_cpic_recommendations(self):
        """Add CPIC guideline recommendations for patient's genes."""
        for gene in self.patient_genes.keys():
            # Find CPIC recommendations for this gene
            gene_cpics = self.cpic_data[self.cpic_data["Gene"].str.upper() == gene]
            
            for _, row in gene_cpics.iterrows():
                drug = row.get("Drug", "Unknown")
                cpic_level = row.get("CPIC Level", "Unknown")
                recommendation = row.get("Recommendation", "")
                
                # Add CPIC annotation to edge if exists
                if drug in self.graph.nodes():
                    # Update existing edge or create new one
                    if self.graph.has_edge(gene, drug):
                        self.graph.edges[gene, drug]["cpic_level"] = cpic_level
                        self.graph.edges[gene, drug]["cpic_recommendation"] = recommendation
                    else:
                        self.graph.add_edge(
                            gene, drug,
                            relation="cpic_guideline",
                            cpic_level=cpic_level,
                            cpic_recommendation=recommendation,
                            weight=1.0
                        )
    
    def create_interactive_visualization(self, output_file: Path, patient_id: str = "Patient"):
        """Create patient-specific interactive visualization with filters."""
        print(f"Creating patient network visualization...")
        
        # Use spring layout
        pos = nx.spring_layout(self.graph, k=1.5, iterations=50, seed=42)
        
        # Edge traces by type
        edge_traces = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            relation = edge[2].get("relation", "unknown")
            weight = edge[2].get("weight", 0.5)
            patient_risk = edge[2].get("patient_risk", "normal")
            is_alert = edge[2].get("alert", False)
            
            # Edge color based on risk/alert
            if is_alert:
                edge_color = "#FF0000"  # Red for alerts
                edge_width = 3
            elif relation == "affects_risk":
                edge_color = "#FF1493"  # Deep pink for risk associations
                edge_width = max(1, weight * 4)
            elif patient_risk == "high_risk":
                edge_color = "#FF6B6B"  # Light red
                edge_width = max(1.5, weight * 3)
            elif patient_risk == "moderate_risk":
                edge_color = "#FFB366"  # Light orange
                edge_width = max(1, weight * 2.5)
            else:
                edge_color = "#888888"  # Gray
                edge_width = max(0.5, weight * 2)
            
            # Hover text
            hover_text = f"{edge[0]} → {edge[1]}<br>{relation}"
            if patient_risk:
                hover_text += f"<br>Risk: {patient_risk.replace('_', ' ')}"
            if is_alert:
                hover_text += "<br>⚠️ INTERACTION ALERT"
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=edge_width, color=edge_color),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Node traces by type and risk level
        node_traces = []
        
        # Gene nodes - color by risk level
        for risk_level in ["high_risk", "moderate_risk", "normal", "unknown"]:
            gene_nodes = [
                n for n, d in self.graph.nodes(data=True)
                if d.get("node_type") == "gene" and d.get("risk_level") == risk_level
            ]
            
            if not gene_nodes:
                continue
            
            x_vals = [pos[node][0] for node in gene_nodes]
            y_vals = [pos[node][1] for node in gene_nodes]
            labels = [node for node in gene_nodes]
            sizes = [15 + self.graph.degree(node) * 3 for node in gene_nodes]
            
            hover_texts = []
            for node in gene_nodes:
                attrs = self.graph.nodes[node]
                patient_info = self.patient_genes[node]
                hover_texts.append(
                    f"GENE: {node}<br>"
                    f"Variant: {patient_info['variant']}<br>"
                    f"Phenotype: {patient_info['phenotype']}<br>"
                    f"Risk: {risk_level.replace('_', ' ').title()}<br>"
                    f"Tier: {attrs.get('tier', 'Unknown')}<br>"
                    f"Connections: {self.graph.degree(node)}"
                )
            
            risk_label = risk_level.replace("_", " ").title()
            node_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=RISK_COLORS[risk_level],
                    line=dict(width=2, color="white"),
                    symbol="circle"
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=11, color="black"),
                hovertext=hover_texts,
                hoverinfo="text",
                name=f"Gene ({risk_label})",
                showlegend=True
            )
            node_traces.append(node_trace)
        
        # Drug nodes - distinguish current meds
        for in_current_meds in [True, False]:
            drug_nodes = [
                n for n, d in self.graph.nodes(data=True)
                if d.get("node_type") == "drug" and d.get("in_current_meds", False) == in_current_meds
            ]
            
            if not drug_nodes:
                continue
            
            x_vals = [pos[node][0] for node in drug_nodes]
            y_vals = [pos[node][1] for node in drug_nodes]
            labels = [node for node in drug_nodes]
            
            # Make actionable drugs larger
            sizes = [
                12 + self.graph.degree(node) * 2 if node in self.actionable_drugs else 8 + self.graph.degree(node)
                for node in drug_nodes
            ]
            
            hover_texts = [
                f"DRUG: {node}<br>"
                f"Current Medication: {'Yes' if in_current_meds else 'No'}<br>"
                f"Actionable: {'Yes' if node in self.actionable_drugs else 'No'}<br>"
                f"Connections: {self.graph.degree(node)}"
                for node in drug_nodes
            ]
            
            # Current meds in bright cyan, others in muted cyan
            drug_color = "#00CED1" if in_current_meds else "#87CEEB"
            drug_label = "Current Medications" if in_current_meds else "Other Drugs"
            
            node_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=drug_color,
                    line=dict(width=2 if in_current_meds else 1, color="white"),
                    symbol="diamond"
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=9, color="black"),
                hovertext=hover_texts,
                hoverinfo="text",
                name=drug_label,
                showlegend=True
            )
            node_traces.append(node_trace)
        
        # Phenotype nodes
        phenotype_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "phenotype"]
        if phenotype_nodes:
            x_vals = [pos[node][0] for node in phenotype_nodes]
            y_vals = [pos[node][1] for node in phenotype_nodes]
            labels = [node for node in phenotype_nodes]
            sizes = [10 + self.graph.degree(node) * 2 for node in phenotype_nodes]
            
            hover_texts = [
                f"PHENOTYPE: {node}<br>"
                f"Adverse Event Risk<br>"
                f"High Risk: {'Yes' if node in self.high_risk_phenotypes else 'No'}<br>"
                f"Associated Genes: {self.graph.degree(node)}"
                for node in phenotype_nodes
            ]
            
            node_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color="#FFB6C1",  # Light pink for phenotypes
                    line=dict(width=1.5, color="white"),
                    symbol="square"
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=8, color="black"),
                hovertext=hover_texts,
                hoverinfo="text",
                name="Adverse Events",
                showlegend=True
            )
            node_traces.append(node_trace)
        
        # Combine traces
        all_traces = edge_traces + node_traces
        
        # Filter buttons
        filter_buttons = [
            dict(
                label="Show All",
                method="update",
                args=[{"visible": [True] * len(all_traces)}]
            ),
            dict(
                label="High Risk Only",
                method="update",
                args=[{"visible": [
                    (trace.name and "High Risk" in trace.name) or "Gene (High Risk)" in str(trace.name)
                    for trace in all_traces
                ]}]
            ),
            dict(
                label="Current Medications",
                method="update",
                args=[{"visible": [
                    (trace.name and ("Current Medications" in trace.name or "Gene" in trace.name))
                    for trace in all_traces
                ]}]
            ),
            dict(
                label="Actionable Drugs",
                method="update",
                args=[{"visible": [
                    # Show genes with high/moderate risk + their connected drugs
                    True  # Simplified - show all for actionable view
                    for trace in all_traces
                ]}]
            ),
            dict(
                label="Adverse Event Risks",
                method="update",
                args=[{"visible": [
                    (trace.name and ("Gene" in trace.name or "Adverse Events" in trace.name))
                    for trace in all_traces
                ]}]
            ),
        ]
        
        # Create figure
        fig = go.Figure(
            data=all_traces,
            layout=go.Layout(
                title=dict(
                    text=f"Patient PGx Network - {patient_id}",
                    font=dict(size=20)
                ),
                showlegend=True,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=80),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
                height=900,
                updatemenus=[
                    dict(
                        buttons=filter_buttons,
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.01,
                        xanchor="left",
                        y=1.15,
                        yanchor="top",
                        bgcolor="white",
                        bordercolor="#888",
                        borderwidth=1
                    )
                ],
                annotations=[
                    dict(
                        text="Filter View:",
                        x=0,
                        xref="paper",
                        y=1.12,
                        yref="paper",
                        align="left",
                        showarrow=False,
                        font=dict(size=14)
                    )
                ]
            )
        )
        
        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))
        print(f"✓ Saved patient network to {output_file}")
        
        return fig
    
    def export_patient_report(self, output_dir: Path, patient_id: str = "Patient"):
        """Export patient-specific PGx report data."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Actionable drugs report
        actionable_data = []
        for drug in self.actionable_drugs:
            # Find genes affecting this drug
            affecting_genes = []
            for neighbor in self.graph.neighbors(drug):
                if self.graph.nodes[neighbor].get("node_type") == "gene":
                    gene_info = self.patient_genes[neighbor]
                    affecting_genes.append({
                        "gene": neighbor,
                        "variant": gene_info["variant"],
                        "phenotype": gene_info["phenotype"],
                        "risk": gene_info["risk_level"]
                    })
            
            actionable_data.append({
                "drug": drug,
                "in_current_meds": drug.lower() in self.current_medications,
                "affecting_genes": affecting_genes,
                "num_genes": len(affecting_genes)
            })
        
        actionable_df = pd.DataFrame(actionable_data)
        actionable_file = output_dir / f"{patient_id}_actionable_drugs.csv"
        actionable_df.to_csv(actionable_file, index=False)
        print(f"✓ Saved {len(actionable_df)} actionable drugs to {actionable_file}")
        
        # High-risk phenotypes report
        risk_phenotypes = []
        for phenotype in self.high_risk_phenotypes:
            associated_genes = []
            for neighbor in self.graph.neighbors(phenotype):
                if self.graph.nodes[neighbor].get("node_type") == "gene":
                    gene_info = self.patient_genes[neighbor]
                    associated_genes.append({
                        "gene": neighbor,
                        "risk_level": gene_info["risk_level"]
                    })
            
            risk_phenotypes.append({
                "phenotype": phenotype,
                "associated_genes": ", ".join([g["gene"] for g in associated_genes]),
                "num_genes": len(associated_genes)
            })
        
        if risk_phenotypes:
            risk_df = pd.DataFrame(risk_phenotypes)
            risk_file = output_dir / f"{patient_id}_risk_phenotypes.csv"
            risk_df.to_csv(risk_file, index=False)
            print(f"✓ Saved {len(risk_df)} risk phenotypes to {risk_file}")
        
        # Drug-drug interactions
        if self.drug_interactions:
            ddi_df = pd.DataFrame(self.drug_interactions, columns=["drug1", "drug2", "interaction_type", "evidence"])
            ddi_file = output_dir / f"{patient_id}_drug_interactions.csv"
            ddi_df.to_csv(ddi_file, index=False)
            print(f"✓ Saved {len(ddi_df)} drug-drug interactions to {ddi_file}")
        
        # Patient summary JSON
        summary = {
            "patient_id": patient_id,
            "genes_tested": list(self.patient_genes.keys()),
            "high_risk_genes": [g for g, info in self.patient_genes.items() if info["risk_level"] == "high_risk"],
            "current_medications": list(self.current_medications),
            "actionable_drugs": list(self.actionable_drugs),
            "num_drug_interactions": len(self.drug_interactions),
            "high_risk_phenotypes": list(self.high_risk_phenotypes),
            "network_stats": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "genes": len(self.entities["gene"]),
                "drugs": len(self.entities["drug"]),
                "phenotypes": len(self.entities["phenotype"])
            }
        }
        
        summary_file = output_dir / f"{patient_id}_pgx_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved patient summary to {summary_file}")


def main():
    """Generate patient-specific PGx network."""
    parser = argparse.ArgumentParser(
        description="Build patient-specific PGx network from genotype data"
    )
    parser.add_argument("--variants", type=Path, required=True, help="Path to patient variants JSON")
    parser.add_argument("--vip-reports", type=Path, required=True, help="Path to VIP reports JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--patient-id", default="Patient", help="Patient identifier")
    parser.add_argument("--medications", nargs="+", help="Current medications (optional)")
    parser.add_argument("--cpic-data", type=Path, help="CPIC gene-drug pairs Excel (optional)")
    parser.add_argument("--no-comprehend", action="store_true", help="Disable AWS Comprehend")
    
    args = parser.parse_args()
    
    # Load patient variants
    with open(args.variants) as f:
        patient_variants = json.load(f)
    
    # Load VIP reports
    with open(args.vip_reports) as f:
        vip_reports = json.load(f)
    
    # Load CPIC data if provided
    cpic_data = None
    if args.cpic_data and args.cpic_data.exists():
        cpic_data = pd.read_excel(args.cpic_data)
    
    # Build network
    builder = PatientPGxNetworkBuilder(
        patient_variants=patient_variants,
        vip_reports=vip_reports,
        current_medications=args.medications,
        cpic_data=cpic_data,
        use_comprehend=not args.no_comprehend
    )
    
    graph = builder.build_patient_network()
    
    # Create visualization
    viz_file = args.output_dir / f"{args.patient_id}_pgx_network.html"
    builder.create_interactive_visualization(viz_file, args.patient_id)
    
    # Export reports
    builder.export_patient_report(args.output_dir, args.patient_id)
    
    print(f"\n✓ Patient PGx network complete!")
    print(f"  Visualization: {viz_file}")
    print(f"  Reports: {args.output_dir}")


if __name__ == "__main__":
    main()
