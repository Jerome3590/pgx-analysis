#!/usr/bin/env python3
"""Check medical table schema for duplicates."""
import json
import subprocess
import sys

def check_schema():
    # Get table definition
    result = subprocess.run(
        ['aws', 'glue', 'get-table', '--database-name', 'medical', 
         '--name', 'medical', '--region', 'us-east-1', '--output', 'json'],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return 1
    
    table = json.loads(result.stdout)['Table']
    
    print("=== GLUE TABLE SCHEMA ===")
    print(f"\nTable: {table['Name']}")
    print(f"Database: medical")
    
    print(f"\nPartition Keys ({len(table['PartitionKeys'])}):")
    for pk in table['PartitionKeys']:
        print(f"  - {pk['Name']} ({pk.get('Type', 'string')})")
    
    col_names = [col['Name'] for col in table['StorageDescriptor']['Columns']]
    print(f"\nRegular Columns: {len(col_names)}")
    print(f"  Has 'age_band': {'age_band' in col_names}")
    print(f"  Has 'event_year': {'event_year' in col_names}")
    
    pk_names = {pk['Name'] for pk in table['PartitionKeys']}
    col_names_set = set(col_names)
    overlap = pk_names & col_names_set
    
    if overlap:
        print(f"\n⚠️  DUPLICATES FOUND: {overlap}")
        print("   These columns appear as BOTH partition keys AND regular columns!")
    else:
        print(f"\n✅ NO DUPLICATES - Partition keys are separate from regular columns")
    
    print(f"\n📝 IMPORTANT:")
    print(f"   - Partition keys (age_band, event_year) come from directory structure")
    print(f"   - Regular columns come from parquet file schema")
    print(f"   - In Athena/Glue Console UI, partition columns WILL appear")
    print(f"   - In query results, partition columns are automatically available")
    print(f"   - This is EXPECTED behavior for partitioned tables")
    print(f"   - The 'duplicate' you see is just partition metadata, not actual duplicates")
    
    return 0

if __name__ == '__main__':
    sys.exit(check_schema())

