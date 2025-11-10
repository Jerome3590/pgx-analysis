"""
Visualization utilities for data analysis and reporting.
"""

import os
import sys

# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

# Standard library imports
import json
import logging
import io
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Third-party imports
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# Local imports
from helpers_1997_13.common_imports import *
from helpers_1997_13.data_utils import convert_json_serializable

from helpers_1997_13.s3_utils import (
    save_to_s3_html,
    get_output_paths,
    write_parquet_and_csv_latest,
    write_drug_frequency_latest,
    write_drug_pairs_latest,
    write_target_code_latest,
)


def plot_consolidated_metrics(consolidated_df: pd.DataFrame, output_path: str, logger: Optional[logging.Logger] = None) -> None:
    """Create and save visualization of consolidated metrics.
    
    Args:
        consolidated_df: DataFrame containing consolidated metrics
        output_path: S3 path to save the visualization
        logger: Optional logger for error tracking
    """
    try:
        # Set up the plot
        plt.figure(figsize=(15, 10))
        sns.set_style("whitegrid")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Support Distribution
        sns.histplot(data=consolidated_df, x='support', bins=30, ax=axes[0,0])
        axes[0,0].set_title('Support Distribution')
        axes[0,0].set_xlabel('Support')
        axes[0,0].set_ylabel('Count')
        
        # Plot 2: Confidence Distribution
        sns.histplot(data=consolidated_df, x='confidence', bins=30, ax=axes[0,1])
        axes[0,1].set_title('Confidence Distribution')
        axes[0,1].set_xlabel('Confidence')
        axes[0,1].set_ylabel('Count')
        
        # Plot 3: Lift Distribution
        sns.histplot(data=consolidated_df, x='lift', bins=30, ax=axes[1,0])
        axes[1,0].set_title('Lift Distribution')
        axes[1,0].set_xlabel('Lift')
        axes[1,0].set_ylabel('Count')
        
        # Plot 4: Support vs Confidence
        sns.scatterplot(data=consolidated_df, x='support', y='confidence', ax=axes[1,1])
        axes[1,1].set_title('Support vs Confidence')
        axes[1,1].set_xlabel('Support')
        axes[1,1].set_ylabel('Confidence')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot to S3
        save_to_s3_html(fig, output_path, logger)
        
        if logger:
            logger.info(f"Saved consolidated metrics visualization to {output_path}")
            
    except Exception as e:
        if logger:
            logger.error(f"Error creating consolidated metrics visualization: {str(e)}")
        raise


def format_for_quicksight(df, column, logger=None):
    try:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(
                lambda x: '|'.join(sorted(str(item) for item in x)) 
                if isinstance(x, (frozenset, set, list)) 
                else str(x) if pd.notnull(x) else ''
            )
        return df
    except Exception as e:
        if logger:
            logger.warning(f"Error formatting {column} for QuickSight: {str(e)}")
        return df


# Determine path relative to this script file
js_path = os.path.join(os.path.dirname(__file__), "js", "cytoscape.min.js")

# Load Cytoscape and FileSaver from local files
with open(js_path, "r", encoding="utf-8") as f:
    cytoscape_js = f.read()
with open(js_path, "r", encoding="utf-8") as f:
    filesaver_js = f.read()


# Define HTML template for network visualization
html_template = Template("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <script>{{ cytoscape_js }}</script>
  <script>{{ filesaver_js }}</script>
  <style>
    body { 
      font-family: Arial, sans-serif; 
      margin: 20px;
    }
    #cy { 
      width: 100%; 
      height: 800px; 
      border: 1px solid #ccc;
      background-color: white;
    }
    .controls {
      margin: 10px 0;
      padding: 10px;
      border: 1px solid #ccc;
      background: #f1f1f1;
      display: flex;
      gap: 10px;
    }
    button { 
      margin: 2px; 
      padding: 8px 15px;
      background-color: #61bffc;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    button:hover {
      background-color: #4fa8e0;
    }
    #legend-container {
      display: flex;
      gap: 20px;
      margin: 10px 0;
    }
    .legend-box {
      padding: 15px;
      border: 1px solid #ccc;
      background-color: #f9f9f9;
      border-radius: 4px;
      min-width: 240px;
      min-height: 120px;
    }
    .legend-box strong {
      display: block;
      margin-bottom: 5px;
    }
    .legend-box div {
      margin-bottom: 4px;
    }
  </style>
</head>
<body>
  <h2 style="font-family:sans-serif">{{ title }}</h2>

  <div class="controls">
    <button onclick="zoomIn()">Zoom In</button>
    <button onclick="zoomOut()">Zoom Out</button>
    <button onclick="resetView()">Reset View</button>
    <button onclick="downloadPNG()">Download PNG</button>
  </div>

  <div id="legend-container">
    <div class="legend-box">
      <strong>Legend:</strong>
      <div><em>Blue nodes</em>: Drugs scaled by centrality</div>
      <div><em>Edges</em>: Width = support, Arrow size = certainty</div>
    </div>
    <div class="legend-box" id="legend-content">
      Hover over a node or edge to see details here.
    </div>
  </div>

  <div id="cy"></div>

  <script>
    let cy = cytoscape({
      container: document.getElementById('cy'),
      elements: {{ elements|safe }},
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(id)',
            'background-color': '#61bffc',
            'width': 'mapData(centrality, 0, 2, 20, 100)',
            'height': 'mapData(centrality, 0, 2, 20, 100)',
            'text-valign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': '80px',
            'font-size': '12px'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 'mapData(support, 0, 1, 1, 10)',
            'curve-style': 'unbundled-bezier',
            'control-point-step-size': 80,
            'control-point-distance': 100,
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#FF0000',
            'line-color': '#00703C',
            'arrow-scale': 'mapData(certainty, 0, 1, 0.5, 2)',
            'overlay-padding': 4
          }
        }
      ],
      layout: null
    });

    function zoomIn() {
      cy.zoom({
        level: cy.zoom() * 1.2,
        renderedPosition: { x: cy.width()/2, y: cy.height()/2 }
      });
    }

    function zoomOut() {
      cy.zoom({
        level: cy.zoom() * 0.8,
        renderedPosition: { x: cy.width()/2, y: cy.height()/2 }
      });
    }

    function resetView() {
      cy.zoom(1);
      cy.center();
    }

    function downloadPNG() {
      try {
        const pngData = cy.png({ full: true, scale: 2, bg: 'white' });
        const byteString = atob(pngData.split(',')[1]);
        const mimeString = pngData.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
          ia[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([ab], { type: mimeString });
        saveAs(blob, '{{ cohort_name }}_{{ age_band }}_{{ event_year }}_drug_network.png');
      } catch (error) {
        console.error('PNG generation failed:', error);
        alert('Failed to generate PNG. Try opening the file in a browser over HTTP.');
      }
    }

    cy.ready(function() {
      document.getElementById('legend-content').innerText = "Hover over a node or edge to see details here.";
    });

    cy.on('mouseout', 'node, edge', function(evt) {
      document.getElementById('legend-content').innerText = "Hover over a node or edge to see details here.";
    });

    cy.on('mouseover', 'node', function(evt) {
      const d = evt.target.data();
      document.getElementById('legend-content').innerHTML = `
        <strong>Node:</strong> ${d.id}<br>
        Centrality: ${d.centrality?.toFixed(3) ?? 'N/A'}
      `;
    });

    cy.on('mouseover', 'edge', function(evt) {
      const d = evt.target.data();
      document.getElementById('legend-content').innerHTML = `
        <strong>Edge:</strong> ${d.source} → ${d.target}<br>
        Support: ${d.support?.toFixed(3) ?? 'N/A'}<br>
        Confidence: ${d.confidence?.toFixed(3) ?? 'N/A'}<br>
        Certainty: ${d.certainty?.toFixed(3) ?? 'N/A'}
      `;
    });
  </script>
</body>
</html>
""")


from collections import defaultdict
import networkx as nx
import json

def create_network_visualization(rules_df, title, cohort_name, age_band, event_year, itemsets_counts=None, logger=None):
    """
    Build a graph, compute centrality, render HTML, and save to S3.

    rules_df must have columns:
      ['antecedents', 'consequents', 'support', 'confidence', 'certainty']
    itemsets_counts (optional): dict mapping item (str) to support (float)
    """
    if logger:
        logger.info(f"Rendering network for {cohort_name} {age_band} {event_year}")
    else:
        print(f"Rendering network for {cohort_name} {age_band} {event_year}")

    # Aggregate edge metrics safely
    edge_agg = defaultdict(lambda: {'support': [], 'confidence': [], 'certainty': []})

    for _, row in rules_df.iterrows():
        antecedents = row['antecedents']
        consequents = row['consequents']
        try:
            support = float(row['support'])
            confidence = float(row['confidence'])
            certainty = float(row['certainty'])
        except (ValueError, TypeError) as e:
            if logger:
                logger.warning(f"Skipping row with non-numeric metric: {row.to_dict()} — {e}")
            continue

        for a in antecedents:
            for c in consequents:
                edge_agg[(a, c)]['support'].append(support)
                edge_agg[(a, c)]['confidence'].append(confidence)
                edge_agg[(a, c)]['certainty'].append(certainty)

    # Build directed graph from aggregated metrics
    G = nx.DiGraph()

    for (a, c), metrics in edge_agg.items():
        G.add_edge(
            a,
            c,
            support=sum(metrics['support']) / len(metrics['support']),
            confidence=sum(metrics['confidence']) / len(metrics['confidence']),
            certainty=sum(metrics['certainty']) / len(metrics['certainty'])
        )

    # Compute degree centrality
    centrality = nx.degree_centrality(G)
    if logger:
        logger.info(f"Centrality computed for {len(centrality)} nodes")

    # Prepare Cytoscape elements
    elements = []

    for node, cent in centrality.items():
        node_data = {
            'id': node,
            'centrality': cent
        }
        if itemsets_counts and node in itemsets_counts:
            node_data['support'] = round(itemsets_counts[node], 5)
        elements.append({'data': node_data})

    for u, v, data in G.edges(data=True):
        elements.append({'data': {
            'source': u,
            'target': v,
            'support': round(data['support'], 5),
            'confidence': round(data['confidence'], 5),
            'certainty': round(data['certainty'], 5)
        }})

    # Render HTML
    html_content = html_template.render(
        title=title,
        elements=json.dumps(convert_json_serializable(elements)),
        cohort_name=cohort_name,
        age_band=age_band,
        event_year=event_year,
        cytoscape_js=cytoscape_js,
        filesaver_js=filesaver_js
    )

    # Save to S3
    html_path = get_output_paths(cohort_name, age_band, event_year)['drug_network_plot']
    save_to_s3_html(html_content, html_path)
    if logger:
        logger.info(f"Saved network HTML to {html_path}")

    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'elements': elements
    }


def create_s3_viz_path(chart_name: str, category: str = 'general', 
                       bucket: Optional[str] = None, add_timestamp: bool = True) -> str:
    """
    Create consistent S3 path for visualization files.
    
    Args:
        chart_name: Descriptive name for chart (e.g., 'high_freq_by_year')
        category: Subfolder under visualizations/ (e.g., 'drug_frequency', 'cohort_analysis')
        bucket: S3 bucket name (uses S3_BUCKET from constants if not provided)
        add_timestamp: Whether to add timestamp to filename
    
    Returns:
        Full S3 path (s3://bucket/visualizations/category/chart_name_timestamp.png)
    
    Example:
        >>> create_s3_viz_path('high_freq_by_year', 'drug_frequency')
        's3://pgx-repository/visualizations/drug_frequency/high_freq_by_year_20241017_143022.png'
    """
    import time
    from helpers_1997_13.constants import S3_BUCKET
    
    if bucket is None:
        bucket = S3_BUCKET
    
    if add_timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{chart_name}_{timestamp}.png"
    else:
        filename = f"{chart_name}.png"
    
    s3_key = f"visualizations/{category}/{filename}"
    s3_url = f"s3://{bucket}/{s3_key}"
    
    return s3_url


def save_chart_to_s3(fig, chart_name: str, category: str = 'general', 
                     bucket: Optional[str] = None, dpi: int = 300,
                     add_timestamp: bool = True, metadata: Optional[Dict[str, str]] = None,
                     logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Save matplotlib figure to S3.
    
    Args:
        fig: Matplotlib figure object
        chart_name: Descriptive name (e.g., 'high_freq_by_year')
        category: Subfolder under visualizations/ (e.g., 'drug_frequency')
        bucket: S3 bucket name (uses S3_BUCKET if not provided)
        dpi: Resolution (default 300 for high quality)
        add_timestamp: Whether to add timestamp to filename
        metadata: Optional metadata dict to attach to S3 object
        logger: Optional logger for output
    
    Returns:
        S3 URL of saved chart, or None if save failed
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> s3_url = save_chart_to_s3(fig, 'my_chart', 'analysis')
        >>> print(s3_url)
        's3://pgx-repository/visualizations/analysis/my_chart_20241017_143022.png'
    """
    try:
        import boto3
        from io import BytesIO
        import time
        from helpers_1997_13.constants import S3_BUCKET
        
        if bucket is None:
            bucket = S3_BUCKET
        
        # Generate S3 path
        if add_timestamp:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{chart_name}_{timestamp}.png"
        else:
            filename = f"{chart_name}.png"
        
        s3_key = f"visualizations/{category}/{filename}"
        
        # Save figure to buffer
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
        buffer.seek(0)
        
        # Prepare S3 upload parameters
        upload_params = {
            'Bucket': bucket,
            'Key': s3_key,
            'Body': buffer,
            'ContentType': 'image/png'
        }
        
        # Add metadata if provided
        if metadata:
            upload_params['Metadata'] = metadata
        else:
            # Add default metadata
            upload_params['Metadata'] = {
                'chart_name': chart_name,
                'category': category,
                'timestamp': timestamp if add_timestamp else time.strftime("%Y%m%d_%H%M%S"),
                'dpi': str(dpi)
            }
        
        # Upload to S3
        s3 = boto3.client('s3')
        s3.put_object(**upload_params)
        
        # Generate S3 URL
        s3_url = f"s3://{bucket}/{s3_key}"
        
        # Log success
        size_kb = buffer.getbuffer().nbytes / 1024
        if logger:
            logger.info(f"✅ Chart saved: {s3_url} ({size_kb:.1f} KB)")
        else:
            print(f"✅ Chart saved: {s3_url}")
            print(f"📏 Size: {size_kb:.1f} KB | Resolution: {dpi} DPI")
        
        return s3_url
        
    except Exception as e:
        if logger:
            logger.error(f"⚠️ Failed to save chart to S3: {e}")
        else:
            print(f"⚠️ S3 save failed: {e}")
        return None


def save_and_display_chart(fig, chart_name: str, category: str = 'general',
                           bucket: Optional[str] = None, dpi: int = 300,
                           display: bool = True, close_fig: bool = True,
                           add_timestamp: bool = True, metadata: Optional[Dict[str, str]] = None,
                           logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Display matplotlib figure in notebook AND save to S3.
    
    Args:
        fig: Matplotlib figure object
        chart_name: Descriptive name (e.g., 'high_freq_by_year')
        category: Subfolder under visualizations/ (e.g., 'drug_frequency')
        bucket: S3 bucket name (uses S3_BUCKET if not provided)
        dpi: Resolution (default 300 for high quality)
        display: Whether to display in notebook (default True)
        close_fig: Whether to close figure after saving (default True)
        add_timestamp: Whether to add timestamp to filename
        metadata: Optional metadata dict to attach to S3 object
        logger: Optional logger for output
    
    Returns:
        S3 URL of saved chart, or None if save failed
    
    Example:
        >>> fig, ax = plt.subplots(figsize=(12, 6))
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> ax.set_title('My Chart')
        >>> s3_url = save_and_display_chart(fig, 'my_chart', 'analysis')
        # Chart displays in notebook AND saves to S3
    """
    # Display in notebook first
    if display:
        plt.show()
    
    # Save to S3
    s3_url = save_chart_to_s3(
        fig=fig,
        chart_name=chart_name,
        category=category,
        bucket=bucket,
        dpi=dpi,
        add_timestamp=add_timestamp,
        metadata=metadata,
        logger=logger
    )
    
    # Close figure to free memory
    if close_fig:
        plt.close(fig)
    
    return s3_url


def save_displayed_chart(chart_name: str, category: str = 'general',
                        bucket: Optional[str] = None, dpi: int = 300,
                        fig=None, display: bool = True, close_fig: bool = True,
                        add_timestamp: bool = True, metadata: Optional[Dict[str, str]] = None,
                        logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Save a matplotlib figure to S3 and optionally display it.
    
    IMPORTANT: This function saves FIRST, then displays (if display=True).
    This prevents the blank image issue that occurs when plt.show() clears the buffer.
    
    Args:
        chart_name: Descriptive name (e.g., 'high_freq_by_year')
        category: Subfolder under visualizations/ (e.g., 'drug_frequency')
        bucket: S3 bucket name (uses S3_BUCKET if not provided)
        dpi: Resolution (default 300 for high quality)
        fig: Matplotlib figure object (uses current figure if not provided)
        display: Whether to display in notebook (default True)
        close_fig: Whether to close figure after saving (default True)
        add_timestamp: Whether to add timestamp to filename
        metadata: Optional metadata dict to attach to S3 object
        logger: Optional logger for output
    
    Returns:
        S3 URL of saved chart, or None if save failed
    
    Example - Correct Way (Save BEFORE plt.show()):
        >>> plt.figure(figsize=(12, 6))
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> plt.title('My Chart')
        >>> 
        >>> # Save and display (saves first, then shows)
        >>> s3_url = save_displayed_chart('my_chart', 'analysis', display=True)
        >>> print(s3_url)
        's3://pgx-repository/visualizations/analysis/my_chart_20241017_143022.png'
    
    Example - After plt.show() Already Called:
        >>> plt.plot([1, 2, 3])
        >>> plt.show()  # Already displayed - buffer is cleared!
        >>> 
        >>> # This will NOT work - image will be blank!
        >>> # Instead, use the pattern above or use save_and_display_chart()
    """
    # Get current figure if not provided
    if fig is None:
        fig = plt.gcf()
    
    # IMPORTANT: Save FIRST (before display clears the buffer)
    s3_url = save_chart_to_s3(
        fig=fig,
        chart_name=chart_name,
        category=category,
        bucket=bucket,
        dpi=dpi,
        add_timestamp=add_timestamp,
        metadata=metadata,
        logger=logger
    )
    
    # Then display (if requested)
    if display:
        plt.show()
    
    # Close figure to free memory
    if close_fig:
        plt.close(fig)
    
    return s3_url


def save_multiple_charts(charts_config: List[Dict[str, Any]], 
                        bucket: Optional[str] = None, dpi: int = 300,
                        logger: Optional[logging.Logger] = None) -> Dict[str, Optional[str]]:
    """
    Save multiple charts to S3 in batch.
    
    Args:
        charts_config: List of chart configurations, each containing:
            - fig: Matplotlib figure object
            - chart_name: Descriptive name
            - category: Subfolder under visualizations/
            - metadata: (optional) Additional metadata
        bucket: S3 bucket name (uses S3_BUCKET if not provided)
        dpi: Resolution (default 300 for high quality)
        logger: Optional logger for output
    
    Returns:
        Dictionary mapping chart_name to S3 URL (or None if failed)
    
    Example:
        >>> charts = [
        ...     {'fig': fig1, 'chart_name': 'chart1', 'category': 'analysis'},
        ...     {'fig': fig2, 'chart_name': 'chart2', 'category': 'analysis'}
        ... ]
        >>> results = save_multiple_charts(charts)
        >>> print(results)
        {'chart1': 's3://...', 'chart2': 's3://...'}
    """
    results = {}
    
    for config in charts_config:
        chart_name = config['chart_name']
        
        s3_url = save_chart_to_s3(
            fig=config['fig'],
            chart_name=chart_name,
            category=config.get('category', 'general'),
            bucket=bucket,
            dpi=dpi,
            add_timestamp=config.get('add_timestamp', True),
            metadata=config.get('metadata'),
            logger=logger
        )
        
        results[chart_name] = s3_url
        
        # Close figure if requested
        if config.get('close_fig', True):
            plt.close(config['fig'])
    
    if logger:
        success_count = sum(1 for url in results.values() if url is not None)
        logger.info(f"📊 Saved {success_count}/{len(charts_config)} charts to S3")
    
    return results


# -----------------------------------------------------------------------------
# Reusable frequency visualization helpers
# -----------------------------------------------------------------------------

def set_plot_style() -> None:
    plt.style.use("default")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (16, 10)
    plt.rcParams["font.size"] = 10


def plot_stacked_by_year(
    df: pd.DataFrame,
    target_col: str = "target_code",
    year_col: str = "event_year",
    freq_col: str = "frequency",
    ordered_targets: Optional[List[str]] = None,
    title_suffix: str = "",
) -> None:
    set_plot_style()
    pivot = (
        df.pivot_table(index=target_col, columns=year_col, values=freq_col, aggfunc="sum")
        .fillna(0)
    )
    if ordered_targets is not None:
        pivot = pivot.reindex(ordered_targets)
    ax = pivot.plot(kind="bar", stacked=True, width=0.82)
    base_title = f"Stacked Frequency by Year"
    if title_suffix:
        base_title += f" - {title_suffix}"
    plt.title(base_title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel(target_col, fontsize=12, fontweight="bold")
    plt.ylabel("Frequency (Count)", fontsize=12, fontweight="bold")
    plt.legend(title="Year", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    for container in getattr(ax, 'containers', []):
        try:
            ax.bar_label(container, label_type="center", fontsize=8, rotation=90)
        except Exception:
            pass
    plt.tight_layout()


def plot_top_bars(
    df: pd.DataFrame,
    target_col: str,
    value_col: str,
    top_n: int = 10,
    title: str = "",
) -> None:
    set_plot_style()
    top_df = df.nlargest(top_n, value_col)
    ax = top_df.plot(x=target_col, y=value_col, kind="barh", color="steelblue")
    ax.invert_yaxis()
    plt.title(title or f"Top {top_n} by {value_col}", fontsize=14, fontweight="bold")
    plt.xlabel(value_col)
    plt.ylabel(target_col)
    plt.tight_layout()


def plot_heatmap_from_pairs(
    pairs_df: pd.DataFrame,
    row_col: str = "target_icd",
    col_col: str = "target_cpt",
    value_col: str = "frequency",
    title: str = "ICD vs CPT Co-Occurrence",
) -> None:
    set_plot_style()
    pivot = (
        pairs_df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="sum")
        .fillna(0)
    )
    sns.heatmap(pivot, cmap="viridis")
    plt.title(title)
    plt.xlabel(col_col)
    plt.ylabel(row_col)
    plt.tight_layout()


def save_current_chart(chart_name: str, category: str, dpi: int = 300) -> Optional[str]:
    """Compatibility wrapper around save_displayed_chart."""
    return save_displayed_chart(chart_name=chart_name, category=category, dpi=dpi)


# -----------------------------------------------------------------------------
# Plotly Dashboard (HTML) for Frequency Exploration
# -----------------------------------------------------------------------------

def create_plotly_frequency_dashboard(
    df: pd.DataFrame,
    title: str,
    s3_output_path: str,
    target_col: str = 'target_code',
    year_col: str = 'event_year',
    freq_col: str = 'frequency',
    system_col: Optional[str] = 'target_system',
    top_n: int = 20,
    pairs_df: Optional[pd.DataFrame] = None,
    pair_row_col: Optional[str] = None,
    pair_col_col: Optional[str] = None,
    pair_year_col: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Render a self-contained Plotly HTML dashboard with:
      - Text filter on target (target_code or drug_name)
      - Optional system filter (ICD/CPT/etc.) if system_col present
      - Year range filter
      - Grouped/stacked bar chart (frequency by year for top N targets)
      - Multi-select to focus on a custom set
      - Optional co-occurrence heatmap for selected items (if pairs_df provided)
      - Export filtered rows as CSV

    Saves directly to s3_output_path via save_to_s3_html.
    """
    import json as _json

    # Prepare data for embedding
    # Ensure correct dtypes for JSON serialization
    safe_df = df[[c for c in [target_col, year_col, freq_col, system_col] if c and c in df.columns]].copy()
    # Coerce to basic types
    safe_df[year_col] = safe_df[year_col].astype(int)
    safe_df[freq_col] = safe_df[freq_col].astype(float)
    # Optional system column
    have_system = system_col is not None and system_col in safe_df.columns

    data_records = safe_df.to_dict(orient='records')
    data_json = _json.dumps(data_records)
    target_col_js = _json.dumps(target_col)
    year_col_js = _json.dumps(year_col)
    freq_col_js = _json.dumps(freq_col)
    system_col_js = _json.dumps(system_col if have_system else '')

    # Optional pairs data for co-occurrence (e.g., within-domain or cross-domain)
    if pairs_df is not None and pair_row_col and pair_col_col:
        pairs_keep = [c for c in [pair_row_col, pair_col_col, pair_year_col, 'frequency'] if c and c in pairs_df.columns]
        pairs_sanitized = pairs_df[pairs_keep].copy()
        if pair_year_col and pair_year_col in pairs_sanitized.columns:
            pairs_sanitized[pair_year_col] = pairs_sanitized[pair_year_col].astype(int)
        if 'frequency' in pairs_sanitized.columns:
            pairs_sanitized['frequency'] = pairs_sanitized['frequency'].astype(float)
        pairs_json = _json.dumps(pairs_sanitized.to_dict(orient='records'))
        row_col_js = _json.dumps(pair_row_col)
        col_col_js = _json.dumps(pair_col_col)
        year_col_pairs_js = _json.dumps(pair_year_col or '')
    else:
        pairs_json = '[]'
        row_col_js = _json.dumps('')
        col_col_js = _json.dumps('')
        year_col_pairs_js = _json.dumps('')

    # Build HTML with Plotly CDN and simple controls using Template to avoid f-string brace escaping
    from string import Template as _Template
    html_template = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>${title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 12px; }
    .controls label { font-weight: bold; margin-right: 6px; }
    #chart { width: 100%; height: 640px; }
    #cochart { width: 100%; height: 640px; margin-top: 18px; }
    .btn { padding: 8px 12px; background: #1769aa; color: white; border: 0; border-radius: 4px; cursor: pointer; }
  </style>
</head>
<body>
  <h2>${title}</h2>
  <div class="controls">
    <div>
      <label for="txtFilter">Filter (${target_col}):</label>
      <input type="text" id="txtFilter" placeholder="contains..." />
    </div>
    <div>
      <label for="txtMulti">Multi-select (comma-separated):</label>
      <input type="text" id="txtMulti" placeholder="A10,B20,C30" />
    </div>
    <div id="systemWrap" style="display:none;">
      <label for="selSystem">System:</label>
      <select id="selSystem"><option value="">All</option></select>
    </div>
    <div>
      <label for="yrMin">Year Min:</label>
      <input type="number" id="yrMin" />
      <label for="yrMax" style="margin-left:8px;">Year Max:</label>
      <input type="number" id="yrMax" />
    </div>
    <div>
      <label for="topN">Top N:</label>
      <input type="number" id="topN" value="${top_n}" min="1" max="200" />
    </div>
    <button class="btn" id="btnApply">Apply Filters</button>
    <button class="btn" id="btnExport">Export CSV</button>
  </div>

  <div id="chart"></div>
  <div id="cochart" style="display:none;"></div>

  <script>
    const raw = ${data_json};
    const targetCol = ${target_col_js};
    const yearCol = ${year_col_js};
    const freqCol = ${freq_col_js};
    const systemCol = ${system_col_js};
    const pairs = ${pairs_json};
    const rowCol = ${row_col_js};
    const colCol = ${col_col_js};
    const yearPairsCol = ${year_col_pairs_js};

    // Init year bounds
    const years = Array.from(new Set(raw.map(r => r[yearCol]))).sort((a,b)=>a-b);
    const minYear = years[0] || 2016;
    const maxYear = years[years.length-1] || 2020;
    document.getElementById('yrMin').value = minYear;
    document.getElementById('yrMax').value = maxYear;

    // Populate system select if available
    if (systemCol) {
      const systems = Array.from(new Set(raw.map(r => r[systemCol]).filter(v => v !== undefined && v !== null)));
      if (systems.length > 0) {
        const sel = document.getElementById('selSystem');
        systems.sort().forEach(s => {
          const opt = document.createElement('option');
          opt.value = s; opt.textContent = s; sel.appendChild(opt);
        });
        document.getElementById('systemWrap').style.display = 'block';
      }
    }

    function applyFilters() {
      const needle = (document.getElementById('txtFilter').value || '').toLowerCase();
      const multiRaw = (document.getElementById('txtMulti').value || '').trim();
      const multiSet = new Set(multiRaw ? multiRaw.split(',').map(s=>s.trim()).filter(Boolean) : []);
      const sysVal = systemCol ? document.getElementById('selSystem').value : '';
      const yMin = parseInt(document.getElementById('yrMin').value || minYear, 10);
      const yMax = parseInt(document.getElementById('yrMax').value || maxYear, 10);
      const topN = Math.max(1, parseInt(document.getElementById('topN').value || ${top_n}, 10));

      let rows = raw.filter(r => r[yearCol] >= yMin && r[yearCol] <= yMax);
      if (needle) rows = rows.filter(r => String(r[targetCol] || '').toLowerCase().includes(needle));
      if (systemCol && sysVal) rows = rows.filter(r => r[systemCol] === sysVal);

      // Aggregate totals by target to pick topN
      const totals = new Map();
      rows.forEach(r => {
        const key = r[targetCol];
        totals.set(key, (totals.get(key) || 0) + Number(r[freqCol] || 0));
      });
      let ranked = Array.from(totals.entries()).sort((a,b)=>b[1]-a[1]).slice(0, topN).map(p=>p[0]);
      if (multiSet.size > 0) {
        ranked = Array.from(multiSet);
      }
      const filtered = rows.filter(r => ranked.includes(r[targetCol]));

      // Build traces per selected target
      const yearSet = Array.from(new Set(filtered.map(r => r[yearCol]))).sort((a,b)=>a-b);
      const traces = [];
      ranked.forEach(tgt => {
        const vals = yearSet.map(y => {
          const sum = filtered.filter(r => r[targetCol]===tgt && r[yearCol]===y).reduce((acc, r) => acc + Number(r[freqCol]||0), 0);
          return sum;
        });
        traces.push({
          x: yearSet,
          y: vals,
          type: 'bar',
          name: String(tgt)
        });
      });

      const layout = {
        barmode: 'group',
        title: 'Frequency by Year',
        xaxis: {title: 'Year'},
        yaxis: {title: 'Frequency'},
        legend: {orientation: 'h', y: -0.2}
      };
      Plotly.newPlot('chart', traces, layout, {responsive: true});

      // Store current filtered rows for export
      window.__filteredRows = filtered;

      // Co-occurrence heatmap (if pairs provided and multi-select active)
      const co = document.getElementById('cochart');
      if (pairs && Array.isArray(pairs) && rowCol && colCol && multiSet.size > 0) {
        co.style.display = '';
        const wanted = new Set(ranked);
        const p = pairs.filter(r => wanted.has(String(r[rowCol])) && wanted.has(String(r[colCol])));
        const mat = {};
        p.forEach(r => {
          const a = String(r[rowCol]);
          const b = String(r[colCol]);
          const v = Number(r['frequency'] || 0);
          if (!mat[a]) mat[a] = {};
          mat[a][b] = (mat[a][b] || 0) + v;
        });
        const rowsLab = Array.from(new Set(Object.keys(mat))).sort();
        const colsLab = Array.from(new Set(rowsLab.flatMap(a => Object.keys(mat[a]||{})))).sort();
        const z = rowsLab.map(a => colsLab.map(b => (mat[a] && mat[a][b]) ? mat[a][b] : 0));
        const heat = [{z, x: colsLab, y: rowsLab, type: 'heatmap', colorscale: 'Viridis'}];
        const hl = {title: 'Co-Occurrence (selected)', xaxis:{title: colCol}, yaxis:{title: rowCol}};
        Plotly.newPlot('cochart', heat, hl, {responsive: true});
      } else {
        co.style.display = 'none';
      }
    }

    function exportCSV() {
      const rows = window.__filteredRows || raw;
      if (!rows.length) return;
      const cols = Object.keys(rows[0]);
      const csv = [cols.join(',')].concat(rows.map(r => cols.map(c => JSON.stringify(r[c]===undefined?'':r[c])).join(','))).join('\n');
      const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = 'filtered_frequency.csv'; a.click();
      URL.revokeObjectURL(url);
    }

    document.getElementById('btnApply').addEventListener('click', applyFilters);
    document.getElementById('btnExport').addEventListener('click', exportCSV);

    // Initial render
    applyFilters();
  </script>
</body>
</html>
"""
    html = _Template(html_template).substitute(
        title=title,
        target_col=target_col,
        top_n=top_n,
        data_json=data_json,
        target_col_js=target_col_js,
        year_col_js=year_col_js,
        freq_col_js=freq_col_js,
        system_col_js=system_col_js,
        pairs_json=pairs_json,
        row_col_js=row_col_js,
        col_col_js=col_col_js,
        year_col_pairs_js=year_col_pairs_js,
    )

    # Save to S3
    try:
        save_to_s3_html(html, s3_output_path)
        if logger:
            logger.info(f"Plotly frequency dashboard saved: {s3_output_path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save dashboard: {e}")
        raise


# Reusable frequency visualization helpers

# writers now re-exported from s3_utils
