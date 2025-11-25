#!/bin/bash
# Fix duplicate partition columns in Glue tables after crawler runs
# Usage: ./fix_glue_table_duplicates.sh <database> <table> <region>

DATABASE=${1:-medical}
TABLE=${2:-medical}
REGION=${3:-us-east-1}

echo "Fixing duplicate partition columns in ${DATABASE}.${TABLE}..."

# Get table definition
aws glue get-table --database-name "$DATABASE" --name "$TABLE" --region "$REGION" --output json > /tmp/table_def.json

# Get partition keys
PARTITION_KEYS=$(aws glue get-table --database-name "$DATABASE" --name "$TABLE" --region "$REGION" --query 'Table.PartitionKeys[].Name' --output json | python3 -c "import sys, json; print(','.join(json.load(sys.stdin)))")

echo "Partition keys: $PARTITION_KEYS"

# Remove partition keys from columns
python3 << PYEOF
import json
import sys

with open('/tmp/table_def.json', 'r') as f:
    table = json.load(f)['Table']

partition_keys = [pk['Name'] for pk in table['PartitionKeys']]
original_cols = table['StorageDescriptor']['Columns']
filtered_cols = [col for col in original_cols if col['Name'] not in partition_keys]

print(f"Removed {len(original_cols) - len(filtered_cols)} duplicate partition column(s)")

table_input = {
    'Name': table['Name'],
    'StorageDescriptor': {
        'Columns': filtered_cols,
        'Location': table['StorageDescriptor']['Location'],
        'InputFormat': table['StorageDescriptor']['InputFormat'],
        'OutputFormat': table['StorageDescriptor']['OutputFormat'],
        'SerdeInfo': table['StorageDescriptor']['SerdeInfo']
    },
    'PartitionKeys': table['PartitionKeys'],
    'TableType': table.get('TableType', 'EXTERNAL_TABLE'),
    'Parameters': table.get('Parameters', {})
}

# Add optional fields
for key in ['Compressed', 'NumberOfBuckets', 'BucketColumns', 'SortColumns', 'StoredAsSubDirectories']:
    if key in table['StorageDescriptor']:
        table_input['StorageDescriptor'][key] = table['StorageDescriptor'][key]

with open('/tmp/table_input.json', 'w') as f:
    json.dump(table_input, f, indent=2)

print("✅ Created updated table definition")
PYEOF

# Update table
aws glue update-table --database-name "$DATABASE" --table-input file:///tmp/table_input.json --region "$REGION" --output json

# Verify
aws glue get-table --database-name "$DATABASE" --name "$TABLE" --region "$REGION" --query 'Table.{PartitionKeys:PartitionKeys[].Name,ColumnCount:length(StorageDescriptor.Columns)}' --output json

echo "✅ Table fixed!"

# Cleanup
rm -f /tmp/table_def.json /tmp/table_input.json

