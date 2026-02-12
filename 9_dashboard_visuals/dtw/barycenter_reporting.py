"""
Map DTW barycenters (or consensus sequences) back to discrete codes for clinical reports.

Supports two workflows:
1. **Numeric barycenter** (e.g. from tslearn DBA): map float values to nearest integer
   code and look up descriptions from metadata.
2. **Categorical trajectories** (our pipeline: "ICD:E11.9", "DRUG:Metformin", "CPT:99213"):
   mode-based consensus at each step, then optional description lookup.

Output: DataFrame suitable for "Clinical Journey" summary tables (Step, Code, Description, Category).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def map_barycenter_to_codes(
    barycenter_seq: Union[np.ndarray, List[float]],
    metadata: Dict[int, Dict[str, Any]],
    default_desc: str = "Unknown",
    default_cat: str = "N/A",
) -> pd.DataFrame:
    """
    Map a numeric barycenter sequence (floats) to discrete codes via nearest-integer
    and pull descriptions from a code metadata dict.

    Parameters
    ----------
    barycenter_seq : array-like of float
        One-dimensional sequence of barycenter values (e.g. from tslearn DBA).
    metadata : dict
        Maps integer code -> {"desc": str, "cat": str} (and optionally other keys).
    default_desc, default_cat : str
        Used when the rounded code is not in metadata.

    Returns
    -------
    pd.DataFrame
        Columns: Step (1-based), Mapped_Code, Description, Category.
    """
    report = []
    for i, val in enumerate(barycenter_seq):
        closest_code = int(round(float(val)))
        info = metadata.get(closest_code, {"desc": default_desc, "cat": default_cat})
        report.append({
            "Step": i + 1,
            "Mapped_Code": closest_code,
            "Description": info.get("desc", default_desc),
            "Category": info.get("cat", default_cat),
        })
    return pd.DataFrame(report)


def _activity_category(activity: str) -> str:
    """Return ICD, DRUG, or CPT from activity string like 'ICD:E11.9' or 'DRUG:Metformin'."""
    if not activity or not isinstance(activity, str):
        return "Unknown"
    a = activity.strip().upper()
    if a.startswith("ICD:"):
        return "ICD-10"
    if a.startswith("DRUG:"):
        return "Drug"
    if a.startswith("CPT:"):
        return "CPT"
    return "Unknown"


def _activity_code_only(activity: str) -> str:
    """Return code part only, e.g. 'ICD:E11.9' -> 'E11.9', 'DRUG:Metformin' -> 'Metformin'."""
    if not activity or not isinstance(activity, str):
        return ""
    for prefix in ("ICD:", "DRUG:", "CPT:"):
        if activity.upper().startswith(prefix.upper()):
            return activity[len(prefix):].strip()
    return activity.strip()


def mode_based_consensus_table(
    sequences: Sequence[Sequence[str]],
    step_labels: Optional[List[str]] = None,
    code_descriptions: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Build a consensus "journey" table from a list of categorical code sequences
    (e.g. list of trajectories like ["ICD:I10", "DRUG:Lisinopril", "CPT:99213", ...]).
    At each step index, the representative code is the **mode** (most frequent) among
    all sequences that have a value at that step (handles variable length by
    taking mode over present indices only, or use step_labels to fix length).

    Parameters
    ----------
    sequences : list of list of str
        Each inner list is one patient's activity sequence (e.g. "ICD:xxx", "DRUG:yyy").
    step_labels : list of str, optional
        If provided, length must match max sequence length; used as Step label.
    code_descriptions : dict, optional
        Maps full activity string or code-only string -> description for report.

    Returns
    -------
    pd.DataFrame
        Columns: Step, Representative_Code, Category, Description (if lookup provided).
    """
    if not sequences:
        return pd.DataFrame(columns=["Step", "Representative_Code", "Category", "Description"])

    max_len = max(len(s) for s in sequences)
    rows = []
    for step in range(max_len):
        codes_at_step = []
        for seq in sequences:
            if step < len(seq) and seq[step]:
                codes_at_step.append(seq[step])
        if not codes_at_step:
            rows.append({
                "Step": step_labels[step] if step_labels and step < len(step_labels) else step + 1,
                "Representative_Code": "",
                "Category": "",
                "Description": "",
            })
            continue
        mode_code = Counter(codes_at_step).most_common(1)[0][0]
        code_only = _activity_code_only(mode_code)
        cat = _activity_category(mode_code)
        desc = ""
        if code_descriptions:
            desc = code_descriptions.get(mode_code) or code_descriptions.get(code_only) or ""
        rows.append({
            "Step": step_labels[step] if step_labels and step < len(step_labels) else step + 1,
            "Representative_Code": code_only or mode_code,
            "Category": cat,
            "Description": desc,
        })
    return pd.DataFrame(rows)


def load_code_metadata_from_csv(
    path: Path,
    code_col: str = "code",
    desc_col: str = "description",
    category_col: Optional[str] = "category",
) -> Dict[str, str]:
    """
    Load a CSV with code -> description (and optional category) for use as
    code_descriptions in mode_based_consensus_table. Keys are normalized to string.

    CSV columns: code_col (e.g. 'E11.9', '99213', 'Metformin'), desc_col, optionally category_col.
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if code_col not in df.columns or desc_col not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        code = str(row[code_col]).strip()
        desc = str(row[desc_col]).strip()
        out[code] = desc
    return out


# Example: numeric barycenter + integer code metadata (as in user snippet)
if __name__ == "__main__":
    code_metadata = {
        1: {"desc": "Essential Hypertension", "cat": "ICD-10"},
        2: {"desc": "Metformin 500mg", "cat": "Drug"},
        3: {"desc": "Office Visit, Est.", "cat": "CPT"},
        4: {"desc": "HbA1c Test", "cat": "CPT"},
        5: {"desc": "Type 2 Diabetes", "cat": "ICD-10"},
    }
    average_journey = np.array([1.1, 1.9, 2.2, 3.0, 4.1, 4.0, 2.1, 1.0, 5.0])
    journey_report = map_barycenter_to_codes(average_journey, code_metadata)
    print("Numeric barycenter -> codes:")
    print(journey_report)

    # Categorical mode-based example (our pipeline format)
    sequences = [
        ["ICD:I10", "DRUG:Lisinopril", "CPT:99213", "ICD:E11.9", "DRUG:Metformin"],
        ["ICD:I10", "ICD:I10", "DRUG:Lisinopril", "CPT:99213", "DRUG:Metformin"],
        ["ICD:I10", "DRUG:Lisinopril", "CPT:99213", "CPT:82947", "DRUG:Metformin"],
    ]
    consensus = mode_based_consensus_table(sequences)
    print("\nMode-based consensus (categorical):")
    print(consensus)
