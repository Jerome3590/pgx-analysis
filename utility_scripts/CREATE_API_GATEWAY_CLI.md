# Create API Gateway "pgx-risk-calculator" via AWS CLI

Use these commands to create a new REST API named **pgx-risk-calculator** and connect it to the Lambda function **pgx-risk-calculator**. Replace `REGION`, `ACCOUNT_ID`, and the generated IDs as you go.

**Prerequisite:** Lambda function `pgx-risk-calculator` must already exist.

---

## Option A: Run the script (recommended)

From repo root:

- **Windows (PowerShell):**
  ```powershell
  .\utility_scripts\create_api_gateway_pgx_risk_calculator.ps1
  ```
  With profile: `.\utility_scripts\create_api_gateway_pgx_risk_calculator.ps1 -Profile your_profile`

- **Linux / macOS (bash):**
  ```bash
  bash utility_scripts/create_api_gateway_pgx_risk_calculator.sh
  ```
  With profile: `AWS_PROFILE=your_profile bash utility_scripts/create_api_gateway_pgx_risk_calculator.sh`

---

## Option B: Commands step-by-step

Set variables (use your region and get account from `aws sts get-caller-identity`):

```bash
REGION=us-east-1
API_NAME=pgx-risk-calculator
LAMBDA_NAME=pgx-risk-calculator
ACCOUNT_ID=535362115856
```

### 1. Create REST API

```bash
API_ID=$(aws apigateway create-rest-api \
  --name "$API_NAME" \
  --description "PGx Risk Calculator API" \
  --endpoint-configuration types=EDGE \
  --region "$REGION" \
  --query id --output text)
echo "API_ID=$API_ID"
```

### 2. Get root resource ID

```bash
ROOT_ID=$(aws apigateway get-resources --rest-api-id "$API_ID" --region "$REGION" --query "items[?path=='/'].id" --output text)
echo "ROOT_ID=$ROOT_ID"
```

### 3. Create {proxy+} resource

```bash
PROXY_ID=$(aws apigateway create-resource \
  --rest-api-id "$API_ID" \
  --parent-id "$ROOT_ID" \
  --path-part "{proxy+}" \
  --region "$REGION" \
  --query id --output text)
echo "PROXY_ID=$PROXY_ID"
```

### 4. Put ANY method on {proxy+}

```bash
aws apigateway put-method \
  --rest-api-id "$API_ID" \
  --resource-id "$PROXY_ID" \
  --http-method ANY \
  --authorization-type NONE \
  --request-parameters "method.request.path.proxy=true" \
  --region "$REGION"
```

### 5. Lambda proxy integration on {proxy+}

```bash
LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${LAMBDA_NAME}"
URI="arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations"

aws apigateway put-integration \
  --rest-api-id "$API_ID" \
  --resource-id "$PROXY_ID" \
  --http-method ANY \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri "$URI" \
  --region "$REGION"
```

### 6. Allow API Gateway to invoke Lambda

```bash
SOURCE_ARN="arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*"
aws lambda add-permission \
  --function-name "$LAMBDA_NAME" \
  --statement-id "apigateway-invoke-${API_ID}" \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "$SOURCE_ARN" \
  --region "$REGION"
```

### 7. Deploy to prod stage

```bash
aws apigateway create-deployment \
  --rest-api-id "$API_ID" \
  --stage-name prod \
  --region "$REGION"
```

### 8. Invoke URL

```
https://<API_ID>.execute-api.<REGION>.amazonaws.com/prod
```

Example: `https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod`

Update your frontend `API_BASE` (e.g. in `9_risk_dashboard/frontend/index.html`) to this URL.

---

## Add PHTS-style resources (per-tab paths)

The create script above only adds `/` and `/{proxy+}`. To match the PHTS calculator API layout with explicit resources per tab:

- **/metadata** – GET, OPTIONS  
- **/risk** – POST, OPTIONS  
- **/risk/comparison** – POST, OPTIONS  
- **/causal/importance** – POST, OPTIONS  
- **/causal/interactions** – POST, OPTIONS  

Run **after** the API exists (same Lambda; these paths take precedence over `{proxy+}`):

- **Windows (PowerShell):** `.\utility_scripts\add_api_gateway_resources_phts_style.ps1`
- **Linux / macOS (bash):** `bash utility_scripts/add_api_gateway_resources_phts_style.sh`

Then redeploy the API (the script deploys to `prod` automatically).
