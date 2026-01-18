#!/usr/bin/env python3
"""Create top sequences from traces if missing."""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def create_top_sequences_from_traces(cohort_name: str, age_band: str, train_label: str = "train"):
    """Create top sequences from traces files if top sequence files are missing."""
    
    age_band_fname = age_band.replace("-", "_")
    bupar_output_dir = PROJECT_ROOT / "5_bupaR_analysis" / "outputs" / cohort_name / age_band_fname / "features"
    
    # Check for traces files
    overall_traces_path = bupar_output_dir / f"{cohort_name}_{age_band_fname}_{train_label}_target_traces_bupar.csv"
    pre_traces_path = bupar_output_dir / f"{cohort_name}_{age_band_fname}_{train_label}_target_pre_f1120_traces_bupar.csv"
    post_traces_path = bupar_output_dir / f"{cohort_name}_{age_band_fname}_{train_label}_target_post_f1120_traces_bupar.csv"
    
    # Check for top sequence files
    overall_top_path = bupar_output_dir / f"{cohort_name}_{age_band_fname}_{train_label}_target_traces_top_bupar.csv"
    pre_top_path = bupar_output_dir / f"{cohort_name}_{age_band_fname}_{train_label}_target_pre_f1120_traces_top_bupar.csv"
    post_top_path = bupar_output_dir / f"{cohort_name}_{age_band_fname}_{train_label}_target_post_f1120_traces_top_bupar.csv"
    
    def create_top_from_traces(traces_path: Path, top_path: Path, name: str):
        if top_path.exists():
            print(f"[INFO] Top sequences file already exists: {top_path.name}")
            return
        
        if not traces_path.exists():
            print(f"[WARN] Traces file not found: {traces_path.name}")
            return
        
        traces_df = pd.read_csv(traces_path)
        
        if traces_df.empty:
            print(f"[WARN] Traces file is empty: {traces_path.name}")
            return
        
        # Calculate frequency
        trace_counts = traces_df.groupby('trace').size().reset_index(name='absolute_frequency')
        total_cases = len(traces_df)
        trace_counts['relative_frequency'] = trace_counts['absolute_frequency'] / total_cases
        trace_counts['sequence_category'] = 'top'
        
        # Top sequences: top 20% by frequency or top 20 sequences, whichever is larger
        top_n_threshold = max(20, int(total_cases * 0.1))
        top_sequences = trace_counts.nlargest(top_n_threshold, 'absolute_frequency').copy()
        
        if not top_sequences.empty:
            top_sequences.to_csv(top_path, index=False)
            print(f"[INFO] Created {len(top_sequences)} top {name} sequences: {top_path.name}")
        else:
            print(f"[WARN] No top sequences created for {name}")
    
    # Create top sequences for each type
    if overall_traces_path.exists():
        create_top_from_traces(overall_traces_path, overall_top_path, "overall")
    
    if pre_traces_path.exists():
        create_top_from_traces(pre_traces_path, pre_top_path, "pre-F1120")
    
    if post_traces_path.exists():
        create_top_from_traces(post_traces_path, post_top_path, "post-F1120")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--age-band", required=True)
    parser.add_argument("--train-label", default="train")
    args = parser.parse_args()
    
    create_top_sequences_from_traces(args.cohort, args.age_band, args.train_label)

