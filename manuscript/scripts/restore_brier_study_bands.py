"""
Restore original brier/ICI values for the 7 primary study bands.
New extended computation missed medium/high/extreme bin joblib models,
causing n_test to shrink to low-bin only. Restore originals and keep
newly-computed cross-cohort bands.
"""
import json

ORIGINAL = {
    "opioid_ed": {
        "13-24":  {"brier": 0.0083, "ici": 0.1635, "n_test": 7728,  "model": "catboost_per_bin"},
        "25-44":  {"brier": 0.0129, "ici": 0.1084, "n_test": 63817, "model": "catboost_per_bin"},
        "45-54":  {"brier": 0.0070, "ici": 0.1536, "n_test": 25661, "model": "catboost_per_bin"},
        "55-64":  {"brier": 0.0509, "ici": 0.1384, "n_test": 26361, "model": "catboost_per_bin"},
    },
    "non_opioid_ed": {
        "65-74":  {"brier": 0.0079, "ici": 0.0707, "n_test": 8196,  "model": "catboost_per_bin"},
        "75-84":  {"brier": 0.0071, "ici": 0.2896, "n_test": 2230,  "model": "catboost_per_bin"},
        "85-114": {"brier": 0.0070, "ici": 0.2120, "n_test": 1643,  "model": "catboost_per_bin"},
    },
}

with open("brier_ici_results.json") as f:
    current = json.load(f)

for cohort, bands in ORIGINAL.items():
    for band, vals in bands.items():
        current[cohort][band] = vals
        print(f"Restored {cohort}/{band}: brier={vals['brier']}, n_test={vals['n_test']}")

print("\nFinal brier_ici_results.json:")
for cohort in ("opioid_ed", "non_opioid_ed"):
    for band, vals in sorted(current[cohort].items()):
        b = vals.get("brier")
        n = vals.get("n_test")
        flag = " [placeholder]" if b is None else ""
        print(f"  {cohort}/{band}: brier={b}, n_test={n}{flag}")

with open("brier_ici_results.json", "w") as f:
    json.dump(current, f, indent=2)
print("\nSaved brier_ici_results.json")
