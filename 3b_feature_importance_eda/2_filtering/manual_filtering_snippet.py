# ============================================
# MANUAL CODE FILTERING
# ============================================
# Add codes here that you want to filter based on your review
# Format: one code per line as a string
# 
# NOTE: This section is idempotent - it will load existing config if it exists
# and merge new codes with existing ones. Running multiple times is safe.

# Initialize filtering_recommendations if not already defined
# Works in both scripts and notebooks
try:
    # Check if it exists by trying to access it
    _ = filtering_recommendations
except NameError:
    # Doesn't exist, create it
    filtering_recommendations = {
        'administrative_codes': set(),
        'bupar_post_target': set(),
        'manual_additional': set()
    }
    print("ℹ️  Initialized filtering_recommendations with empty sets")

# Load existing config if it exists (idempotency)
filtering_config_path = OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_manual_filtering_config.json"
existing_manual_codes = []
existing_codes_to_keep = []

if filtering_config_path.exists():
    try:
        with open(filtering_config_path, 'r') as f:
            existing_config = json.load(f)
        existing_codes_to_keep = existing_config.get('codes_to_keep', [])
        existing_manual_codes = []
        # Extract manual codes from existing config
        if 'manual_additional_count' in existing_config:
            manual_count = existing_config['manual_additional_count']
            all_codes = existing_config.get('codes_to_filter', [])
            admin_count = existing_config.get('administrative_codes_count', 0)
            bupar_count = existing_config.get('bupar_post_target_count', 0)
            if len(all_codes) > (admin_count + bupar_count):
                start_idx = admin_count + bupar_count
                existing_manual_codes = all_codes[start_idx:]
        print(f"✅ Loaded existing config: {len(existing_manual_codes)} manual codes, {len(existing_codes_to_keep)} codes to keep")
    except Exception as e:
        print(f"⚠️  Could not load existing config: {e}")
        print(f"   Starting with empty lists")

# Default: empty lists (can be overridden by user)
MANUAL_CODES_TO_FILTER = [
    # Example: "Z00.00",  # Administrative code
    # Example: "V70.0",   # Routine exam
    # Add your codes here:
]

# Remove codes from filtering if they should be kept
CODES_TO_KEEP = [
    # Example: "F11.20",  # Keep this code even if flagged
    # Add codes to keep here:
]

# Merge with existing codes (idempotent - union operation)
# Start with existing manual codes, then add new ones
all_manual_codes = set(existing_manual_codes) | set(MANUAL_CODES_TO_FILTER)
all_codes_to_keep = set(existing_codes_to_keep) | set(CODES_TO_KEEP)

# Ensure filtering_recommendations has required keys (safety check)
if 'manual_additional' not in filtering_recommendations:
    filtering_recommendations['manual_additional'] = set()
if 'administrative_codes' not in filtering_recommendations:
    filtering_recommendations['administrative_codes'] = set()
if 'bupar_post_target' not in filtering_recommendations:
    filtering_recommendations['bupar_post_target'] = set()

# Update filtering recommendations (merge, don't replace)
filtering_recommendations['manual_additional'].update(all_manual_codes)

# Remove codes that should be kept from all categories
for code in all_codes_to_keep:
    filtering_recommendations['administrative_codes'].discard(code)
    filtering_recommendations['bupar_post_target'].discard(code)
    filtering_recommendations['manual_additional'].discard(code)

# Final list of codes to filter (union of all categories)
final_codes_to_filter = (
    filtering_recommendations['administrative_codes'] |
    filtering_recommendations['bupar_post_target'] |
    filtering_recommendations['manual_additional']
)

print(f"✅ Updated filtering list")
print(f"   Total codes to filter: {len(final_codes_to_filter)}")
print(f"     - Administrative codes: {len(filtering_recommendations['administrative_codes'])}")
print(f"     - BupaR post-target codes: {len(filtering_recommendations['bupar_post_target'])}")
print(f"     - Manual additional codes: {len(filtering_recommendations['manual_additional'])}")
print(f"     - Codes to keep: {len(all_codes_to_keep)}")

# Safe to proceed even if all lists are empty - workflow will continue
if len(final_codes_to_filter) == 0:
    print(f"\n   ℹ️  No codes to filter - workflow will proceed without filtering")
    print(f"   (This is safe and expected if no manual codes are added)")

if len(final_codes_to_filter) > 0:
    print(f"\n   Codes to filter (showing first 50):")
    for code in sorted(list(final_codes_to_filter))[:50]:
        print(f"     - {code}")
    if len(final_codes_to_filter) > 50:
        print(f"     ... and {len(final_codes_to_filter) - 50} more")

# Save filtering list to JSON for use in next step (idempotent - overwrites with complete state)
filtering_config = {
    'codes_to_filter': sorted(list(final_codes_to_filter)),
    'codes_to_keep': sorted(list(all_codes_to_keep)),
    'administrative_codes_count': len(filtering_recommendations['administrative_codes']),
    'bupar_post_target_count': len(filtering_recommendations['bupar_post_target']),
    'manual_additional_count': len(filtering_recommendations['manual_additional'])
}

with open(filtering_config_path, 'w') as f:
    json.dump(filtering_config, f, indent=2)

print(f"\n   💾 Saved filtering config to: {filtering_config_path}")
