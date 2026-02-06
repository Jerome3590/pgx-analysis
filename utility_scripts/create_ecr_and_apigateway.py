#!/usr/bin/env python3
"""
Create ECR repository and API Gateway for PGx Risk Calculator (idempotent).
Uses boto3; works on Windows and Linux. Run where AWS credentials are configured.

Usage:
  python utility_scripts/create_ecr_and_apigateway.py [--profile PROFILE]
  On EC2: no profile (uses instance role).
"""
import argparse
import os
import sys
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional: use credentials file at repo parent (e.g. C:\Projects\credentials)
_creds = REPO_ROOT.parent / "credentials"
if _creds.exists() and not os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(_creds)

REGION = os.environ.get("AWS_REGION", "us-east-1")
ECR_REPO = "pgx-risk-dashboard"
API_NAME = "pgx-calculator-api"


def main():
    parser = argparse.ArgumentParser(description="Create ECR repo and API Gateway if not present")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"), help="AWS profile")
    args = parser.parse_args()

    session_kw = {}
    if args.profile:
        session_kw["profile_name"] = args.profile
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("boto3 required: pip install boto3", file=sys.stderr)
        sys.exit(1)

    session = boto3.Session(**session_kw)
    print(f"Region: {REGION}")
    print()

    # ECR
    print("=== ECR repository ===")
    ecr = session.client("ecr", region_name=REGION)
    try:
        ecr.describe_repositories(repositoryNames=[ECR_REPO])
        print(f"Already exists: {ECR_REPO}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            ecr.create_repository(repositoryName=ECR_REPO)
            print(f"Created: {ECR_REPO}")
        else:
            raise
    print()

    # API Gateway
    print("=== API Gateway (template) ===")
    apigw = session.client("apigateway", region_name=REGION)
    apis = apigw.get_rest_apis(limit=500)
    existing = [a for a in apis.get("items", []) if a.get("name") == API_NAME]
    if existing:
        print(f"Already exists: {API_NAME} (id: {existing[0]['id']})")
    else:
        apigw.create_rest_api(
            name=API_NAME,
            description="PGx Risk Calculator API",
            endpointConfiguration={"types": ["EDGE"]},
        )
        print(f"Created: {API_NAME}")
    print()
    print("Next (API Gateway console): add resource (e.g. /predict), add POST method, integrate with Lambda (container image from ECR).")


if __name__ == "__main__":
    main()
