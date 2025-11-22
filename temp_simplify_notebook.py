import json

# Load notebook
with open('3_fpgrowth_analysis/global_fpgrowth_feature_importance.ipynb', 'r') as f:
    nb = json.load(f)

# Keep only cells 0-10 and add summary
kept_cells = nb['cells'][:11]

# Add summary markdown
summary_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Summary\n', '\n', 'View detailed results for each item type.']
}

# Add results code cell
results_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import pandas as pd\n',
        '\n',
        '# Display results summary\n',
        'results_df = pd.DataFrame([r for r in results if "error" not in r])\n',
        '\n',
        'print("=" * 80)\n',
        'print("GLOBAL FPGROWTH ANALYSIS - FINAL SUMMARY")\n',
        'print("=" * 80)\n',
        '\n',
        'print("\\n Results by Item Type:")\n',
        'print(results_df)\n',
        '\n',
        'print("\\n Output Locations:")\n',
        'for result in results:\n',
        '    if "error" not in result:\n',
        '        print(f"  {result[\'item_type\']}: {result[\'s3_folder\']}")\n',
        '\n',
        'print("\\n Next Steps:")\n',
        'print("  1. Load encoding maps for CatBoost")\n',
        'print("  2. Use itemsets/rules for pattern analysis")\n',
        'print("  3. Run cohort-specific FPGrowth")\n',
        '\n',
        'print(f"\\n Analysis complete: {datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\')}")\n',
        'print("=" * 80)\n'
    ]
}

kept_cells.append(summary_md)
kept_cells.append(results_code)

nb['cells'] = kept_cells

# Save
with open('3_fpgrowth_analysis/global_fpgrowth_feature_importance.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Updated: {len(nb["cells"])} cells')
