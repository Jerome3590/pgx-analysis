#!/usr/bin/env python3
"""
Apply CORS headers to API Gateway gateway responses so the browser receives
Access-Control-Allow-Origin on every response (including 4XX, 5XX, and Lambda errors).

When the dashboard at https://jerome-dixon.io fetches the API, the browser requires
CORS headers. Lambda already returns them for 200 responses, but:
- If Lambda returns 4xx/5xx, API Gateway may still pass headers (proxy integration).
- If Lambda throws or times out, API Gateway returns 502/503 without invoking Lambda,
  so no CORS headers are sent unless we set them on gateway responses.

Usage:
  python apply_api_gateway_cors.py --api-id cmv0qislq3 [--region us-east-1] [--profile PROFILE]
  python apply_api_gateway_cors.py --check --api-id cmv0qislq3   # print current and exit

After running, redeploy is not required; gateway responses take effect immediately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# CORS headers to add to gateway responses (value must be quoted for API Gateway template)
CORS_RESPONSE_PARAMS = {
    "gatewayresponse.header.Access-Control-Allow-Origin": "'*'",
    "gatewayresponse.header.Access-Control-Allow-Headers": "'Content-Type,Accept'",
    "gatewayresponse.header.Access-Control-Allow-Methods": "'GET,POST,OPTIONS'",
    "gatewayresponse.header.Access-Control-Max-Age": "'86400'",
}

# Response types that should include CORS (so errors from API Gateway still allow origin)
# DEFAULT is not a valid type; INTEGRATION_FAILURE / INTEGRATION_TIMEOUT cover Lambda 502/504
GATEWAY_RESPONSE_TYPES = (
    "DEFAULT_4XX",
    "DEFAULT_5XX",
    "INTEGRATION_FAILURE",   # Lambda throws / 502
    "INTEGRATION_TIMEOUT",   # Lambda timeout / 504
)


def apply_gateway_cors(api_id: str, region: str = "us-east-1", profile: str | None = None) -> None:
    import boto3
    from botocore.exceptions import ClientError

    session = boto3.Session(region_name=region, profile_name=profile)
    client = session.client("apigateway")

    for response_type in GATEWAY_RESPONSE_TYPES:
        try:
            client.put_gateway_response(
                restApiId=api_id,
                responseType=response_type,
                responseParameters=CORS_RESPONSE_PARAMS,
            )
            print(f"  Set CORS on gateway response: {response_type}")
        except ClientError as e:
            print(f"  Failed {response_type}: {e}", file=sys.stderr)
            raise


def get_gateway_responses(api_id: str, region: str = "us-east-1", profile: str | None = None) -> dict:
    import boto3

    session = boto3.Session(region_name=region, profile_name=profile)
    client = session.client("apigateway")
    return client.get_gateway_responses(restApiId=api_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add CORS headers to API Gateway gateway responses (4XX, 5XX, DEFAULT)."
    )
    parser.add_argument("--api-id", required=True, help="REST API id (e.g. cmv0qislq3)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", default=None, help="AWS profile name")
    parser.add_argument("--check", action="store_true", help="Print current gateway responses and exit")
    args = parser.parse_args()

    if args.check:
        try:
            resp = get_gateway_responses(args.api_id, args.region, args.profile)
            print("Gateway responses:")
            for gr in resp.get("items", []):
                params = gr.get("responseParameters") or {}
                cors = "Access-Control-Allow-Origin" in str(params)
                print(f"  {gr['responseType']}: statusCode={gr.get('statusCode')} CORS={cors}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    print(f"Applying CORS to API {args.api_id} (region={args.region})")
    try:
        apply_gateway_cors(args.api_id, args.region, args.profile)
        print("Done. Requests from https://jerome-dixon.io (or any origin) will receive CORS headers.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
