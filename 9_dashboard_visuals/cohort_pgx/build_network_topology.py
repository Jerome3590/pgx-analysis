#!/usr/bin/env python3
"""
Build network topology from PharmGKB VIP reports using pytextrank and AWS Comprehend.

Combines:
- pytextrank: Extract key phrases and relationships from reports
- AWS Comprehend: Entity recognition, key phrase extraction, sentiment
- Network analysis: Build gene-drug-phenotype topology

Creates interactive network visualizations for the Cohort PGx dashboard tab.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import spacy
import pytextrank

try:
    import boto3
    COMPREHEND_AVAILABLE = True
except ImportError:
    print("Warning: boto3 not available. AWS Comprehend features disabled.")
    COMPREHEND_AVAILABLE = False


class CohortPGxNetworkBuilder:
    """Build network topology from VIP reports."""
    
    def __init__(self, reports_file: Path, use_comprehend: bool = True):
        """
        Initialize network builder.
        
        Args:
            reports_file: Path to VIP reports JSON
            use_comprehend: Whether to use AWS Comprehend (requires boto3)
        """
        self.reports_file = reports_file
        self.use_comprehend = use_comprehend and COMPREHEND_AVAILABLE
        
        # Load reports
        with open(reports_file, "r", encoding="utf-8") as f:
            self.reports = json.load(f)
        
        # Initialize spaCy + pytextrank
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model en_core_web_sm...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Add pytextrank to pipeline
        if "textrank" not in self.nlp.pipe_names:
            self.nlp.add_pipe("textrank")
        
        # Initialize AWS Comprehend if available
        self.comprehend_client = None
        if self.use_comprehend:
            try:
                self.comprehend_client = boto3.client("comprehend", region_name="us-east-1")
                print("✓ AWS Comprehend initialized")
            except Exception as e:
                print(f"Warning: Could not initialize AWS Comprehend: {e}")
                self.use_comprehend = False
        
        # Network storage
        self.graph = nx.Graph()
        self.entities = defaultdict(set)  # entity_type -> set of entities
        self.relationships = []  # list of (source, target, relation_type, weight, evidence)
        self.key_phrases = defaultdict(list)  # gene_symbol -> list of key phrases
        self.drug_interactions = []  # list of (drug1, drug2, interaction_type, evidence)
        self.gene_tiers = {}  # gene_symbol -> tier (Tier 1, Tier 2, etc.)
        self.cpic_genes = set()  # genes with CPIC guidelines
    
    def extract_text_from_report(self, report: Dict) -> str:
        """Extract all text content from a report."""
        texts = []
        
        # Gene name and basic info
        if report.get("gene_name"):
            texts.append(report["gene_name"])
        
        # VIP summary text (main source of clinical information)
        if report.get("vip_summary_text"):
            texts.append(report["vip_summary_text"])
        
        # Citation text
        if report.get("citation_text"):
            texts.append(report["citation_text"])
        
        # VIP page text (if available from HTML scraping)
        if report.get("vip_text"):
            vip_text = report["vip_text"]
            for section in ["overview", "clinical_annotations", "variant_annotations", "drug_labels"]:
                if vip_text.get(section):
                    texts.append(vip_text[section])
        
        return " ".join(texts)
    
    def extract_metadata(self, report: Dict) -> Dict:
        """Extract metadata from report (tier, CPIC status, etc.)."""
        gene_symbol = report.get("gene_symbol", "Unknown")
        
        # Get tier information
        cpic_gene = report.get("cpic_gene", False)
        amp = report.get("amp", False)
        vip_tier = report.get("vip_tier", "Unknown")
        
        # Store for later use
        self.gene_tiers[gene_symbol] = vip_tier
        if cpic_gene:
            self.cpic_genes.add(gene_symbol)
        
        return {
            "gene_symbol": gene_symbol,
            "cpic_gene": cpic_gene,
            "amp": amp,
            "vip_tier": vip_tier,
        }
    
    def extract_phenotypes(self, text: str) -> Set[str]:
        """Extract phenotypes and adverse events from text."""
        phenotypes = set()
        
        # Common adverse event patterns
        ae_patterns = [
            r'(?:risk of|incidence of|occurrence of)\s+([a-z\s-]{3,30})',
            r'(?:adverse events?|side effects?|toxicity|reaction):\s*([a-z\s,-]+)',
            r'([a-z\s-]{3,30})\s+(?:risk|toxicity|reaction)',
        ]
        
        text_lower = text.lower()
        
        for pattern in ae_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Clean up the match
                phenotype = match.strip().strip(',;').strip()
                if 5 < len(phenotype) < 50:  # Reasonable length
                    phenotypes.add(phenotype.title())
        
        # Look for specific adverse events mentioned
        specific_aes = [
            "Bleeding", "Thrombosis", "Nausea", "Vomiting", "Diarrhea",
            "Constipation", "Respiratory Depression", "Sedation", "Dizziness",
            "Headache", "Liver Toxicity", "Nephrotoxicity", "Cardiotoxicity",
            "Myopathy", "Neuropathy", "Hypersensitivity", "Rash",
            "Stevens-Johnson Syndrome", "Agranulocytosis", "Pancytopenia",
            "QT Prolongation", "Seizures", "Serotonin Syndrome",
            "Extrapyramidal Symptoms", "Tardive Dyskinesia",
        ]
        
        for ae in specific_aes:
            if ae.lower() in text_lower:
                phenotypes.add(ae)
        
        return phenotypes
    
    def extract_drug_interactions(self, text: str, drug_set: Set[str]) -> List[Tuple]:
        """
        Extract drug-drug interactions from text.
        
        Args:
            text: VIP summary text
            drug_set: Set of known drugs in the network
            
        Returns:
            List of (drug1, drug2, interaction_type, evidence_text)
        """
        interactions = []
        text_lower = text.lower()
        
        # Interaction patterns
        interaction_patterns = [
            (r'(\w+)\s+(?:and|with)\s+(\w+)\s+(?:interact|interaction|increase|decrease|inhibit)', 'metabolic'),
            (r'(\w+)\s+inhibits\s+(\w+)', 'inhibition'),
            (r'(\w+)\s+induces\s+(\w+)', 'induction'),
            (r'combination of\s+(\w+)\s+(?:and|with)\s+(\w+)', 'combination'),
            (r'(\w+)\s+enhances\s+(\w+)', 'enhancement'),
        ]
        
        for pattern, interaction_type in interaction_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                drug1, drug2 = match[0].title(), match[1].title()
                
                # Check if both are in our drug set or reasonably look like drug names
                if (drug1 in drug_set or len(drug1) > 4) and (drug2 in drug_set or len(drug2) > 4):
                    # Extract evidence context (100 chars around match)
                    match_pos = text_lower.find(match[0].lower())
                    if match_pos != -1:
                        start = max(0, match_pos - 50)
                        end = min(len(text), match_pos + 100)
                        evidence = text[start:end].strip()
                    else:
                        evidence = ""
                    
                    interactions.append((drug1, drug2, interaction_type, evidence))
        
        return interactions
    

        """Extract entities and key phrases using pytextrank."""
        if not text or len(text) < 10:
            return {"phrases": [], "entities": []}
        
        # Limit text length for processing
        text = text[:50000]  # First 50k chars
        
        doc = self.nlp(text)
        
        # Extract key phrases
        phrases = []
        for phrase in doc._.phrases[:20]:  # Top 20 phrases
            phrases.append({
                "text": phrase.text,
                "rank": phrase.rank,
                "count": phrase.count
            })
            self.key_phrases[gene_symbol].append(phrase.text)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
            # Store in entity collections
            if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
                # Could be drug names or organizations
                self.entities["drug"].add(ent.text)
            elif ent.label_ in ["DISEASE", "SYMPTOM"]:
                self.entities["phenotype"].add(ent.text)
        
        return {"phrases": phrases, "entities": entities}
    
    def extract_entities_comprehend(self, text: str, gene_symbol: str) -> Dict:
        """Extract entities using AWS Comprehend."""
        if not self.use_comprehend or not text or len(text) < 10:
            return {}
        
        # Comprehend has 5000 byte limit per request
        text = text[:5000].encode("utf-8").decode("utf-8", errors="ignore")
        
        try:
            # Detect entities
            entities_response = self.comprehend_client.detect_entities(
                Text=text,
                LanguageCode="en"
            )
            
            # Detect key phrases
            phrases_response = self.comprehend_client.detect_key_phrases(
                Text=text,
                LanguageCode="en"
            )
            
            # Detect medical entities (if available in your AWS account)
            try:
                medical_response = self.comprehend_client.detect_entities_v2(
                    Text=text
                )
                medical_entities = medical_response.get("Entities", [])
            except Exception:
                medical_entities = []
            
            # Store entities
            for entity in entities_response.get("Entities", []):
                entity_type = entity["Type"]
                entity_text = entity["Text"]
                
                if entity_type in ["COMMERCIAL_ITEM", "TITLE"]:
                    self.entities["drug"].add(entity_text)
                elif entity_type in ["EVENT", "OTHER"]:
                    self.entities["phenotype"].add(entity_text)
            
            # Store key phrases
            for phrase in phrases_response.get("KeyPhrases", []):
                phrase_text = phrase["Text"]
                self.key_phrases[gene_symbol].append(phrase_text)
            
            return {
                "entities": entities_response.get("Entities", []),
                "key_phrases": phrases_response.get("KeyPhrases", []),
                "medical_entities": medical_entities
            }
        
        except Exception as e:
            print(f"  Warning: Comprehend error for {gene_symbol}: {e}")
            return {}
    
    def extract_drug_names(self, text: str) -> Set[str]:
        """Extract drug names using pattern matching."""
        drugs = set()
        
        # Common drug name patterns
        # Capitalized words followed by ®, ™, or in parentheses
        patterns = [
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)?)\s*[®™]',  # Trademarked names
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)?)\s*\([a-z]+\)',  # Brand (generic)
            r'\b([a-z]{4,})\s*\(.*?\)',  # generic (info)
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                drug = match.group(1).strip()
                if len(drug) >= 4:
                    drugs.add(drug)
        
        return drugs
    
    def build_network(self) -> nx.Graph:
        """Build network graph from all reports."""
        print(f"\n{'='*80}")
        print(f"Building network topology from {len(self.reports)} reports")
        print(f"{'='*80}\n")
        
        # Process each report
        for i, report in enumerate(self.reports, 1):
            gene_symbol = report.get("gene_symbol", f"GENE_{i}")
            gene_name = report.get("gene_name", gene_symbol)
            
            print(f"[{i}/{len(self.reports)}] Processing {gene_symbol}...", end=" ")
            
            # Extract metadata (tier, CPIC status)
            metadata = self.extract_metadata(report)
            
            # Add gene node with tier and CPIC info
            self.graph.add_node(
                gene_symbol,
                node_type="gene",
                label=gene_name,
                url=report.get("vip_url", ""),
                tier=metadata["vip_tier"],
                cpic_gene=metadata["cpic_gene"],
                amp=metadata["amp"]
            )
            self.entities["gene"].add(gene_symbol)
            
            # Extract text
            text = self.extract_text_from_report(report)
            
            if not text:
                print("✗ No text")
                continue
            
            # Extract entities with pytextrank
            pytextrank_results = self.extract_entities_pytextrank(text, gene_symbol)
            
            # Extract entities with AWS Comprehend
            if self.use_comprehend:
                comprehend_results = self.extract_entities_comprehend(text, gene_symbol)
            
            # Extract drug names
            drugs = self.extract_drug_names(text)
            self.entities["drug"].update(drugs)
            
            # Extract phenotypes (adverse events)
            phenotypes = self.extract_phenotypes(text)
            self.entities["phenotype"].update(phenotypes)
            
            # Calculate evidence weight based on text mentions
            drug_mentions = {drug: text.lower().count(drug.lower()) for drug in drugs}
            phenotype_mentions = {phen: text.lower().count(phen.lower()) for phen in phenotypes}
            
            # Build relationships
            # Gene -> Drug relationships (weighted by mention frequency)
            for drug in drugs:
                if drug and len(drug) >= 4:
                    weight = min(drug_mentions.get(drug, 1), 10) / 10.0  # Normalize to 0-1
                    
                    self.graph.add_node(drug, node_type="drug", label=drug)
                    self.graph.add_edge(
                        gene_symbol, drug,
                        relation="metabolizes",
                        weight=weight,
                        mentions=drug_mentions.get(drug, 1)
                    )
                    self.relationships.append((gene_symbol, drug, "metabolizes", weight, f"{drug_mentions.get(drug, 1)} mentions"))
            
            # Gene -> Phenotype relationships (weighted by mention frequency)
            for phenotype in phenotypes:
                if phenotype and len(phenotype) >= 5:
                    weight = min(phenotype_mentions.get(phenotype, 1), 10) / 10.0
                    
                    self.graph.add_node(phenotype, node_type="phenotype", label=phenotype)
                    self.graph.add_edge(
                        gene_symbol, phenotype,
                        relation="affects_risk",
                        weight=weight,
                        mentions=phenotype_mentions.get(phenotype, 1)
                    )
                    self.relationships.append((gene_symbol, phenotype, "affects_risk", weight, f"{phenotype_mentions.get(phenotype, 1)} mentions"))
            
            # Extract drug-drug interactions
            drug_interactions = self.extract_drug_interactions(text, drugs)
            self.drug_interactions.extend(drug_interactions)
            
            print(f"✓ {len(pytextrank_results['phrases'])} phrases, {len(drugs)} drugs, {len(phenotypes)} phenotypes, {len(drug_interactions)} interactions")
        
        # Add drug-drug interaction edges
        print(f"\nAdding {len(self.drug_interactions)} drug-drug interactions...")
        for drug1, drug2, interaction_type, evidence in self.drug_interactions:
            if drug1 in self.entities["drug"] and drug2 in self.entities["drug"]:
                # Weight based on evidence text length (proxy for detail)
                weight = min(len(evidence) / 100.0, 1.0) if evidence else 0.5
                
                self.graph.add_edge(
                    drug1, drug2,
                    relation=interaction_type,
                    weight=weight,
                    evidence=evidence[:200]  # Truncate long evidence
                )
                self.relationships.append((drug1, drug2, interaction_type, weight, evidence[:100]))
        
        # Add cross-gene relationships based on shared drugs
        self._add_shared_entity_relationships()
        
        print(f"\n{'='*80}")
        print(f"Network built:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Genes: {len(self.entities['gene'])}")
        print(f"  Drugs: {len(self.entities['drug'])}")
        print(f"  Phenotypes: {len(self.entities['phenotype'])}")
        print(f"  Drug-Drug Interactions: {len(self.drug_interactions)}")
        print(f"  CPIC Genes: {len(self.cpic_genes)}")
        print(f"{'='*80}\n")
        
        return self.graph
    
    def _add_shared_entity_relationships(self):
        """Add edges between genes that share drugs/phenotypes."""
        # Find genes that share drugs
        gene_drugs = defaultdict(set)
        for gene in self.entities["gene"]:
            if gene in self.graph:
                for neighbor in self.graph.neighbors(gene):
                    if self.graph.nodes[neighbor].get("node_type") == "drug":
                        gene_drugs[gene].add(neighbor)
        
        # Connect genes with shared drugs
        genes = list(gene_drugs.keys())
        for i, gene1 in enumerate(genes):
            for gene2 in genes[i+1:]:
                shared = gene_drugs[gene1] & gene_drugs[gene2]
                if shared:
                    weight = len(shared)
                    self.graph.add_edge(gene1, gene2, relation="co_metabolizes", weight=weight)
    
    def create_interactive_visualization(self, output_file: Path):
        """Create interactive Plotly network visualization with filters."""
        print(f"Creating interactive visualization...")
        
        # Use spring layout for positioning
        pos = nx.spring_layout(self.graph, k=1, iterations=50, seed=42)
        
        # Define colors
        tier_colors = {
            "Tier 1": "#FF0000",  # Red - most important
            "Tier 2": "#FF6B00",  # Orange
            "Tier 3": "#FFB800",  # Yellow-orange
            "Unknown": "#999999"  # Gray
        }
        
        relation_colors = {
            "metabolizes": "#888888",
            "affects_risk": "#FF69B4",
            "co_metabolizes": "#4169E1",
            "metabolic": "#9370DB",
            "inhibition": "#DC143C",
            "induction": "#32CD32",
            "combination": "#FFD700",
            "enhancement": "#FF6347"
        }
        
        # Create edge traces by relation type (for filtering)
        edge_trace_dict = defaultdict(list)
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            relation = edge[2].get("relation", "unknown")
            weight = edge[2].get("weight", 0.5)
            
            # Edge width based on weight
            edge_width = max(0.5, min(weight * 5, 5))
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=edge_width,
                    color=relation_colors.get(relation, "#888888")
                ),
                hoverinfo="text",
                hovertext=f"{edge[0]} → {edge[1]}<br>{relation}<br>Weight: {weight:.2f}",
                showlegend=False,
                name=f"edge_{relation}",
                visible=True  # All visible by default
            )
            edge_trace_dict[relation].append(edge_trace)
        
        # Flatten edge traces
        all_edge_traces = []
        for relation, traces in edge_trace_dict.items():
            all_edge_traces.extend(traces)
        
        # Create node traces by type and tier
        node_trace_dict = {}
        
        # Gene nodes - separate by tier for filtering
        for tier in ["Tier 1", "Tier 2", "Tier 3", "Unknown"]:
            gene_nodes = [
                n for n, d in self.graph.nodes(data=True) 
                if d.get("node_type") == "gene" and d.get("tier", "Unknown") == tier
            ]
            
            if not gene_nodes:
                continue
            
            x_vals = [pos[node][0] for node in gene_nodes]
            y_vals = [pos[node][1] for node in gene_nodes]
            labels = [self.graph.nodes[node].get("label", node) for node in gene_nodes]
            
            # Node size based on degree
            sizes = [10 + self.graph.degree(node) * 2 for node in gene_nodes]
            
            # Hover text with metadata
            hover_texts = []
            for node in gene_nodes:
                attrs = self.graph.nodes[node]
                cpic = "✓ CPIC" if attrs.get("cpic_gene") else ""
                amp = "✓ AMP" if attrs.get("amp") else ""
                hover_texts.append(
                    f"GENE: {attrs.get('label', node)}<br>"
                    f"Tier: {tier}<br>"
                    f"{cpic} {amp}<br>"
                    f"Connections: {self.graph.degree(node)}"
                )
            
            node_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=tier_colors[tier],
                    line=dict(width=2, color="white"),
                    symbol="circle"
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=10),
                hovertext=hover_texts,
                hoverinfo="text",
                name=f"Gene ({tier})",
                showlegend=True,
                visible=True
            )
            node_trace_dict[f"gene_{tier}"] = node_trace
        
        # Drug nodes
        drug_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "drug"]
        if drug_nodes:
            x_vals = [pos[node][0] for node in drug_nodes]
            y_vals = [pos[node][1] for node in drug_nodes]
            labels = [self.graph.nodes[node].get("label", node) for node in drug_nodes]
            sizes = [8 + self.graph.degree(node) for node in drug_nodes]
            
            hover_texts = [
                f"DRUG: {self.graph.nodes[node].get('label', node)}<br>"
                f"Connections: {self.graph.degree(node)}"
                for node in drug_nodes
            ]
            
            drug_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color="#4ECDC4",  # Cyan
                    line=dict(width=1.5, color="white"),
                    symbol="diamond"
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                hovertext=hover_texts,
                hoverinfo="text",
                name="Drugs",
                showlegend=True,
                visible=True
            )
            node_trace_dict["drugs"] = drug_trace
        
        # Phenotype nodes
        phenotype_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "phenotype"]
        if phenotype_nodes:
            x_vals = [pos[node][0] for node in phenotype_nodes]
            y_vals = [pos[node][1] for node in phenotype_nodes]
            labels = [self.graph.nodes[node].get("label", node) for node in phenotype_nodes]
            sizes = [8 + self.graph.degree(node) for node in phenotype_nodes]
            
            hover_texts = [
                f"PHENOTYPE: {self.graph.nodes[node].get('label', node)}<br>"
                f"Adverse Event<br>"
                f"Associated Genes: {self.graph.degree(node)}"
                for node in phenotype_nodes
            ]
            
            phenotype_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color="#95E1D3",  # Mint green
                    line=dict(width=1.5, color="white"),
                    symbol="square"
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                hovertext=hover_texts,
                hoverinfo="text",
                name="Phenotypes",
                showlegend=True,
                visible=True
            )
            node_trace_dict["phenotypes"] = phenotype_trace
        
        # Combine all traces
        all_traces = all_edge_traces + list(node_trace_dict.values())
        
        # Create filter buttons
        filter_buttons = [
            # Show all
            dict(
                label="Show All",
                method="update",
                args=[{"visible": [True] * len(all_traces)}]
            ),
            # Genes only
            dict(
                label="Genes Only",
                method="update",
                args=[{"visible": [
                    False if "edge" in trace.name else 
                    (True if "gene_" in trace.name else False)
                    for trace in all_traces
                ]}]
            ),
            # Genes + Drugs
            dict(
                label="Genes + Drugs",
                method="update",
                args=[{"visible": [
                    ("edge_metabolizes" in trace.name or 
                     "edge_co_metabolizes" in trace.name or
                     "gene_" in trace.name or 
                     trace.name == "drugs")
                    for trace in all_traces
                ]}]
            ),
            # Genes + Phenotypes
            dict(
                label="Genes + Phenotypes",
                method="update",
                args=[{"visible": [
                    ("edge_affects_risk" in trace.name or 
                     "gene_" in trace.name or 
                     trace.name == "phenotypes")
                    for trace in all_traces
                ]}]
            ),
            # Drug-Drug Interactions
            dict(
                label="Drug-Drug Interactions",
                method="update",
                args=[{"visible": [
                    ("edge_metabolic" in trace.name or
                     "edge_inhibition" in trace.name or
                     "edge_induction" in trace.name or
                     "edge_combination" in trace.name or
                     "edge_enhancement" in trace.name or
                     trace.name == "drugs")
                    for trace in all_traces
                ]}]
            ),
            # Tier 1 Genes Only
            dict(
                label="Tier 1 Only",
                method="update",
                args=[{"visible": [
                    ("edge" in trace.name or trace.name == "Gene (Tier 1)")
                    for trace in all_traces
                ]}]
            ),
        ]
        
        # Create figure
        fig = go.Figure(
            data=all_traces,
            layout=go.Layout(
                title=dict(
                    text="Cohort PGx Network Topology - Interactive",
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
        print(f"✓ Saved interactive visualization to {output_file}")
        print(f"  - {len(node_trace_dict)} node types")
        print(f"  - {len(edge_trace_dict)} edge types")
        print(f"  - Interactive filters enabled")
        
        return fig
    
    def export_network_data(self, output_dir: Path):
        """Export network data in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export nodes with full metadata
        nodes_data = []
        for node, attrs in self.graph.nodes(data=True):
            node_data = {
                "id": node,
                "type": attrs.get("node_type", "unknown"),
                "label": attrs.get("label", node),
                "degree": self.graph.degree(node)
            }
            
            # Add gene-specific metadata
            if attrs.get("node_type") == "gene":
                node_data.update({
                    "tier": attrs.get("tier", "Unknown"),
                    "cpic_gene": attrs.get("cpic_gene", False),
                    "amp": attrs.get("amp", False),
                    "url": attrs.get("url", "")
                })
            
            nodes_data.append(node_data)
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_file = output_dir / "network_nodes.csv"
        nodes_df.to_csv(nodes_file, index=False)
        print(f"✓ Saved {len(nodes_df)} nodes to {nodes_file}")
        
        # Export edges with weights and evidence
        edges_data = []
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                "relation": attrs.get("relation", "related"),
                "weight": attrs.get("weight", 1.0),
                "mentions": attrs.get("mentions", ""),
                "evidence": attrs.get("evidence", "")[:200]  # Truncate long evidence
            }
            edges_data.append(edge_data)
        
        edges_df = pd.DataFrame(edges_data)
        edges_file = output_dir / "network_edges.csv"
        edges_df.to_csv(edges_file, index=False)
        print(f"✓ Saved {len(edges_df)} edges to {edges_file}")
        
        # Export drug-drug interactions separately
        if self.drug_interactions:
            ddi_data = []
            for drug1, drug2, interaction_type, evidence in self.drug_interactions:
                ddi_data.append({
                    "drug1": drug1,
                    "drug2": drug2,
                    "interaction_type": interaction_type,
                    "evidence": evidence[:200]
                })
            
            ddi_df = pd.DataFrame(ddi_data)
            ddi_file = output_dir / "drug_interactions.csv"
            ddi_df.to_csv(ddi_file, index=False)
            print(f"✓ Saved {len(ddi_df)} drug-drug interactions to {ddi_file}")
        
        # Export key phrases
        phrases_file = output_dir / "key_phrases.json"
        with open(phrases_file, "w", encoding="utf-8") as f:
            json.dump(dict(self.key_phrases), f, indent=2, ensure_ascii=False)
        print(f"✓ Saved key phrases to {phrases_file}")
        
        # Export network statistics
        stats = {
            "nodes_total": self.graph.number_of_nodes(),
            "edges_total": self.graph.number_of_edges(),
            "genes": len(self.entities["gene"]),
            "drugs": len(self.entities["drug"]),
            "phenotypes": len(self.entities["phenotype"]),
            "cpic_genes": len(self.cpic_genes),
            "drug_drug_interactions": len(self.drug_interactions),
            "density": nx.density(self.graph),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            "gene_tiers": {
                tier: len([g for g, t in self.gene_tiers.items() if t == tier])
                for tier in set(self.gene_tiers.values())
            }
        }
        
        stats_file = output_dir / "network_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved network statistics to {stats_file}")
        
        # Export tier and CPIC information
        tier_info = {
            "gene_tiers": self.gene_tiers,
            "cpic_genes": list(self.cpic_genes)
        }
        tier_file = output_dir / "gene_metadata.json"
        with open(tier_file, "w", encoding="utf-8") as f:
            json.dump(tier_info, f, indent=2)
        print(f"✓ Saved gene metadata to {tier_file}")


def main():
    """Build network topology from VIP reports."""
    parser = argparse.ArgumentParser(
        description="Build network topology from PharmGKB VIP reports"
    )
    parser.add_argument("--reports", type=Path, required=True, help="Path to VIP reports JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--no-comprehend", action="store_true", help="Disable AWS Comprehend")
    
    args = parser.parse_args()
    
    if not args.reports.exists():
        print(f"Error: Reports file not found: {args.reports}")
        return
    
    # Build network
    builder = CohortPGxNetworkBuilder(
        reports_file=args.reports,
        use_comprehend=not args.no_comprehend
    )
    
    graph = builder.build_network()
    
    # Create visualization
    viz_file = args.output_dir / "network_topology.html"
    builder.create_interactive_visualization(viz_file)
    
    # Export data
    builder.export_network_data(args.output_dir)
    
    print("\n" + "="*80)
    print("Network topology build complete!")
    print("="*80)


if __name__ == "__main__":
    main()
