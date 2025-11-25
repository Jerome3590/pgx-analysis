# Fixing Lake Formation Permissions for AWS Glue

## Error Message
```
Insufficient Lake Formation permission(s) on s3://pgxdatalake/silver/imputed/pharmacy_partitioned/ 
(Database name: pharmacy_raw, Table name: pharmacy_partitioned)
```

## Solution: Grant Lake Formation Permissions

### Option 1: Grant Permissions via AWS Console

1. **Open AWS Lake Formation Console**
   - Navigate to: https://console.aws.amazon.com/lakeformation/
   - Select your region (us-east-1)

2. **Grant Database Permissions**
   - Go to **Data permissions** → **Named data catalog resources**
   - Select **Database**: `pharmacy_raw`
   - Click **Grant**
   - Select your IAM role/user (the one used by Glue)
   - Grant permissions:
     - `DESCRIBE`
     - `ALTER` (if needed for crawler)
   - Click **Grant**

3. **Grant Table Permissions**
   - Go to **Data permissions** → **Named data catalog resources**
   - Select **Database**: `pharmacy_raw`
   - Select **Table**: `pharmacy_partitioned`
   - Click **Grant**
   - Select your IAM role/user
   - Grant permissions:
     - `SELECT`
     - `DESCRIBE`
   - Click **Grant**

4. **Grant S3 Location Permissions** (Alternative/Additional)
   - Go to **Data permissions** → **Data location permissions**
   - Click **Grant**
   - Select your IAM role/user
   - Enter S3 path: `s3://pgxdatalake/silver/imputed/pharmacy_partitioned/`
   - Grant permissions:
     - `DATA_LOCATION_ACCESS`
   - Click **Grant**

### Option 2: Grant Permissions via AWS CLI

```bash
# Grant database permissions
aws lakeformation grant-permissions \
  --principal DataLakePrincipalIdentifier=arn:aws:iam::ACCOUNT_ID:role/GLUE_ROLE_NAME \
  --resource '{"Database":{"Name":"pharmacy_raw"}}' \
  --permissions DESCRIBE ALTER \
  --region us-east-1

# Grant table permissions
aws lakeformation grant-permissions \
  --principal DataLakePrincipalIdentifier=arn:aws:iam::ACCOUNT_ID:role/GLUE_ROLE_NAME \
  --resource '{"Table":{"DatabaseName":"pharmacy_raw","Name":"pharmacy_partitioned"}}' \
  --permissions SELECT DESCRIBE \
  --region us-east-1

# Grant S3 location permissions
aws lakeformation grant-permissions \
  --principal DataLakePrincipalIdentifier=arn:aws:iam::ACCOUNT_ID:role/GLUE_ROLE_NAME \
  --resource '{"DataLocation":{"ResourceArn":"arn:aws:s3:::pgxdatalake/silver/imputed/pharmacy_partitioned/"}}' \
  --permissions DATA_LOCATION_ACCESS \
  --region us-east-1
```

### Option 3: Use Super User (Quick Fix - Not Recommended for Production)

If you're a Lake Formation admin, you can temporarily grant yourself super user permissions:

1. Go to **Administrative roles and tasks** → **Administrators**
2. Add your IAM role/user as an administrator
3. This grants full access to all databases/tables

**Note:** This bypasses fine-grained permissions and is not recommended for production.

## Verify Permissions

After granting permissions, verify access:

```bash
# Test Glue crawler access
aws glue get-table \
  --database-name pharmacy_raw \
  --name pharmacy_partitioned \
  --region us-east-1
```

## Common Issues

1. **IAM Role Not Found**: Ensure the IAM role/user exists and is correctly specified
2. **S3 Path Mismatch**: Ensure the S3 path exactly matches: `s3://pgxdatalake/silver/imputed/pharmacy_partitioned/`
3. **Region Mismatch**: Ensure you're granting permissions in the same region as your Glue resources (us-east-1)
4. **Database/Table Doesn't Exist**: If the database or table doesn't exist in Glue catalog, create it first or let the crawler create it

## Required IAM Permissions

The IAM role used by Glue also needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::pgxdatalake",
        "arn:aws:s3:::pgxdatalake/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "glue:GetDatabase",
        "glue:GetTable",
        "glue:CreateDatabase",
        "glue:CreateTable",
        "glue:UpdateTable"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "lakeformation:GetDataAccess"
      ],
      "Resource": "*"
    }
  ]
}
```

## Next Steps

After fixing permissions:
1. Re-run your Glue crawler
2. Verify the table appears in Glue Data Catalog
3. Test querying via Athena

