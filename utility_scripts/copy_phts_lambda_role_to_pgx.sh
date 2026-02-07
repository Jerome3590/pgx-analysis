#!/bin/bash
# Copy IAM role permissions from phts-lambda-role to pgx-lambda-role.
# Run from project root. Requires AWS CLI and permissions to get role details and create/update roles.
#
# Source role example: arn:aws:iam::535362115856:role/phts-lambda-role
# Target role will be: arn:aws:iam::535362115856:role/pgx-lambda-role
#
# Usage: bash utility_scripts/copy_phts_lambda_role_to_pgx.sh [--dry-run]

set -e
SOURCE_ROLE="phts-lambda-role"
TARGET_ROLE="pgx-lambda-role"
DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

echo "Copying role: $SOURCE_ROLE -> $TARGET_ROLE"
echo ""

# Get trust policy from source role
TRUST_POLICY=$(aws iam get-role --role-name "$SOURCE_ROLE" --query 'Role.AssumeRolePolicyDocument' --output json 2>/dev/null) || {
  echo "ERROR: Could not get role $SOURCE_ROLE (missing or no permission)"
  exit 1
}
echo "Got trust policy from $SOURCE_ROLE"

if [ "$DRY_RUN" = true ]; then
  echo "[dry-run] Would create role $TARGET_ROLE with same trust policy"
  aws iam list-attached-role-policies --role-name "$SOURCE_ROLE" --query 'AttachedPolicies[*].PolicyArn' --output text
  echo "[dry-run] Would attach above policies to $TARGET_ROLE"
  exit 0
fi

# Create target role with same trust policy (ignore if already exists)
TRUST_FILE=$(mktemp)
echo "$TRUST_POLICY" > "$TRUST_FILE"
if aws iam create-role --role-name "$TARGET_ROLE" --assume-role-policy-document "file://$TRUST_FILE" --description "PGx Lambda execution role (copied from $SOURCE_ROLE)" 2>/dev/null; then
  echo "Created role $TARGET_ROLE"
else
  if aws iam get-role --role-name "$TARGET_ROLE" &>/dev/null; then
    echo "Role $TARGET_ROLE already exists; updating trust policy"
    aws iam update-assume-role-policy --role-name "$TARGET_ROLE" --policy-document "file://$TRUST_FILE"
  else
    echo "ERROR: Failed to create role $TARGET_ROLE"
    rm -f "$TRUST_FILE"
    exit 1
  fi
fi
rm -f "$TRUST_FILE"

# Attach same managed policies as source role
for ARN in $(aws iam list-attached-role-policies --role-name "$SOURCE_ROLE" --query 'AttachedPolicies[*].PolicyArn' --output text); do
  [ -z "$ARN" ] && continue
  echo "Attaching $ARN to $TARGET_ROLE"
  aws iam attach-role-policy --role-name "$TARGET_ROLE" --policy-arn "$ARN"
done

echo ""
echo "Done. Role $TARGET_ROLE has same trust policy and attached managed policies as $SOURCE_ROLE."
echo "Use role name '$TARGET_ROLE' in the workflow (Update Lambda cell)."
