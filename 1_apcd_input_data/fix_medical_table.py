#!/usr/bin/env python3
"""Fix medical table by removing duplicate partition columns from regular columns."""
import json
import boto3
import sys

def fix_medical_table():
    glue = boto3.client('glue', region_name='us-east-1')
    
    # Get current table definition
    print("Fetching current table definition...")
    table = glue.get_table(DatabaseName='medical', Name='medical')['Table']
    
    # Remove age_band and event_year from columns (they're partition keys)
    original_count = len(table['StorageDescriptor']['Columns'])
    columns = [col for col in table['StorageDescriptor']['Columns'] 
               if col['Name'] not in ['age_band', 'event_year']]
    removed_count = original_count - len(columns)
    
    print(f"Original columns: {original_count}")
    print(f"Removed duplicates: {removed_count}")
    print(f"Final columns: {len(columns)}")
    
    # Build table input for update
    table_input = {
        'Name': table['Name'],
        'StorageDescriptor': {
            'Columns': columns,
            'Location': table['StorageDescriptor']['Location'],
            'InputFormat': table['StorageDescriptor']['InputFormat'],
            'OutputFormat': table['StorageDescriptor']['OutputFormat'],
            'SerdeInfo': table['StorageDescriptor']['SerdeInfo'],
            'Compressed': table['StorageDescriptor'].get('Compressed', False),
            'NumberOfBuckets': table['StorageDescriptor'].get('NumberOfBuckets', -1)
        },
        'PartitionKeys': table['PartitionKeys'],
        'TableType': table.get('TableType', 'EXTERNAL_TABLE'),
        'Parameters': table.get('Parameters', {})
    }
    
    # Add any other StorageDescriptor fields that might exist
    for key in ['BucketColumns', 'SortColumns', 'StoredAsSubDirectories', 'SkewedInfo']:
        if key in table['StorageDescriptor']:
            table_input['StorageDescriptor'][key] = table['StorageDescriptor'][key]
    
    # Update table
    print("\nUpdating table...")
    glue.update_table(DatabaseName='medical', TableInput=table_input)
    
    # Verify
    updated_table = glue.get_table(DatabaseName='medical', Name='medical')['Table']
    updated_columns = [col['Name'] for col in updated_table['StorageDescriptor']['Columns']]
    
    print("\n✅ Table updated successfully!")
    print(f"Partition keys: {[pk['Name'] for pk in updated_table['PartitionKeys']]}")
    print(f"Has age_band in columns: {'age_band' in updated_columns}")
    print(f"Has event_year in columns: {'event_year' in updated_columns}")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(fix_medical_table())
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

