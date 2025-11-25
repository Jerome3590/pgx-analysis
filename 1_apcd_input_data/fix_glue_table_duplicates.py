#!/usr/bin/env python3
"""
Fix duplicate partition columns in Glue tables after crawler runs.

This script removes partition key columns (like age_band, event_year) from the regular
columns list since they're already available as partition keys from the directory structure.

Usage:
    python fix_glue_table_duplicates.py <database> <table> [region]
    
Example:
    python fix_glue_table_duplicates.py medical medical us-east-1
"""
import json
import sys
import boto3
import argparse

def fix_table_duplicates(database, table_name, region='us-east-1'):
    """Remove partition key columns from regular columns list."""
    glue = boto3.client('glue', region_name=region)
    
    try:
        # Get current table definition
        print(f"Fetching table definition for {database}.{table_name}...")
        table = glue.get_table(DatabaseName=database, Name=table_name)['Table']
        
        # Get partition keys
        partition_keys = [pk['Name'] for pk in table['PartitionKeys']]
        print(f"Partition keys: {partition_keys}")
        
        # Remove partition keys from columns
        original_cols = table['StorageDescriptor']['Columns']
        filtered_cols = [col for col in original_cols 
                        if col['Name'] not in partition_keys]
        
        removed_count = len(original_cols) - len(filtered_cols)
        
        if removed_count == 0:
            print("✅ No duplicate partition columns found. Table is already correct.")
            return 0
        
        print(f"Removing {removed_count} duplicate partition column(s)...")
        print(f"  Original columns: {len(original_cols)}")
        print(f"  Final columns: {len(filtered_cols)}")
        
        # Build table input
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
        
        # Add optional StorageDescriptor fields
        optional_fields = ['Compressed', 'NumberOfBuckets', 'BucketColumns', 
                          'SortColumns', 'StoredAsSubDirectories', 'SkewedInfo']
        for key in optional_fields:
            if key in table['StorageDescriptor']:
                table_input['StorageDescriptor'][key] = table['StorageDescriptor'][key]
        
        # Update table
        print("Updating table...")
        glue.update_table(DatabaseName=database, TableInput=table_input)
        
        # Verify
        updated_table = glue.get_table(DatabaseName=database, Name=table_name)['Table']
        updated_columns = [col['Name'] for col in updated_table['StorageDescriptor']['Columns']]
        
        print("\n✅ Table updated successfully!")
        print(f"Partition keys: {[pk['Name'] for pk in updated_table['PartitionKeys']]}")
        print(f"Total columns: {len(updated_columns)}")
        
        for pk in partition_keys:
            has_duplicate = pk in updated_columns
            status = "❌ STILL PRESENT" if has_duplicate else "✅ REMOVED"
            print(f"  {pk} in columns: {status}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix duplicate partition columns in Glue tables')
    parser.add_argument('database', help='Glue database name')
    parser.add_argument('table', help='Glue table name')
    parser.add_argument('--region', default='us-east-1', help='AWS region (default: us-east-1)')
    
    args = parser.parse_args()
    sys.exit(fix_table_duplicates(args.database, args.table, args.region))

