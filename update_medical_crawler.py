#!/usr/bin/env python3
"""
Update AWS Glue crawler to point to medical_raw S3 path
"""

import boto3
from botocore.exceptions import ClientError

# Configuration
CRAWLER_NAME = "medical"  # Update this if your crawler has a different name
REGION = "us-east-1"
NEW_S3_PATH = "s3://pgxdatalake/silver/medical_raw/"

def update_crawler_targets():
    """Update the Glue crawler's S3 targets"""
    glue = boto3.client('glue', region_name=REGION, profile_name='mushin')
    
    try:
        # Get current crawler configuration
        print(f"Getting current crawler configuration: {CRAWLER_NAME}")
        crawler = glue.get_crawler(Name=CRAWLER_NAME)
        
        current_targets = crawler['Crawler']['Targets']
        print(f"\nCurrent targets:")
        if 'S3Targets' in current_targets:
            for target in current_targets['S3Targets']:
                print(f"  - {target.get('Path', 'N/A')}")
        else:
            print("  No S3 targets found")
        
        # Update crawler with new S3 path
        print(f"\nUpdating crawler to use: {NEW_S3_PATH}")
        
        # Prepare update request
        update_params = {
            'Name': CRAWLER_NAME,
            'Targets': {
                'S3Targets': [
                    {
                        'Path': NEW_S3_PATH
                    }
                ]
            }
        }
        
        # Copy other important settings from current crawler
        crawler_config = crawler['Crawler']
        if 'Role' in crawler_config:
            update_params['Role'] = crawler_config['Role']
        if 'DatabaseName' in crawler_config:
            update_params['DatabaseName'] = crawler_config['DatabaseName']
        if 'Classifiers' in crawler_config:
            update_params['Classifiers'] = crawler_config['Classifiers']
        if 'SchemaChangePolicy' in crawler_config:
            update_params['SchemaChangePolicy'] = crawler_config['SchemaChangePolicy']
        if 'RecrawlPolicy' in crawler_config:
            update_params['RecrawlPolicy'] = crawler_config['RecrawlPolicy']
        if 'LineageConfiguration' in crawler_config:
            update_params['LineageConfiguration'] = crawler_config['LineageConfiguration']
        
        # Update the crawler
        glue.update_crawler(**update_params)
        print(f"✅ Successfully updated crawler '{CRAWLER_NAME}'")
        
        # Verify the update
        print("\nVerifying update...")
        updated_crawler = glue.get_crawler(Name=CRAWLER_NAME)
        updated_targets = updated_crawler['Crawler']['Targets']
        print(f"\nUpdated targets:")
        if 'S3Targets' in updated_targets:
            for target in updated_targets['S3Targets']:
                print(f"  - {target.get('Path', 'N/A')}")
        
        print(f"\n✅ Crawler updated successfully!")
        print(f"\nNext steps:")
        print(f"  1. Start the crawler: aws glue start-crawler --name {CRAWLER_NAME} --profile mushin")
        print(f"  2. Or use: python -c \"import boto3; boto3.client('glue', profile_name='mushin').start_crawler(Name='{CRAWLER_NAME}')\"")
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'EntityNotFoundException':
            print(f"❌ Error: Crawler '{CRAWLER_NAME}' not found")
            print(f"   Available crawlers:")
            try:
                crawlers = glue.list_crawlers()
                for crawler_name in crawlers.get('CrawlerNames', []):
                    print(f"     - {crawler_name}")
            except Exception as list_error:
                print(f"   Could not list crawlers: {list_error}")
        else:
            print(f"❌ Error updating crawler: {e}")
            raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    update_crawler_targets()

