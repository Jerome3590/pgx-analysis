---
name: google-integration
description: Manage Google integrations including Google Drive OAuth, NotebookLM content workflows, and Google Cloud DataCloud MCP tools. Use when working with Google Drive secrets, NotebookLM automation, or GCP DataCloud resources.
---

# Google Integrations

Guidance for managing Google services across the workspace, including Google Drive OAuth, NotebookLM automation, and Google Cloud Platform (GCP) DataCloud tools.

## 1. Google Drive & OAuth Configuration
- **Desktop OAuth Secret**: `C:\Projects\notebookLM-edit-content\secrets\gdrive_jerome_dixon90_client_secret.json`
- **Associated Environment**: `GMAIL_MCP_CLIENT_SECRET` points to this shared desktop secret.
- **Security**: Never commit this file or expose client IDs/secrets in code, logs, or commits.

## 2. NotebookLM Integration
- **Repositories**:
  - `C:\Projects\notebookLM-automation`: EC2 runner, spot compute, corpus indexing, and notebook pipelines.
  - `C:\Projects\notebookLM-edit-content`: Content ingestion, drive cleanup, and secret management.
- **Corpus & Notes**: Adhere to the corpus-first workflow before querying external endpoints.

## 3. Google Cloud / GCP DataCloud MCP Tools
Configured globally via `~/.gemini/config/mcp_config.json`:
- **`notebooks`**: Node-based DataCloud MCP server for Jupyter notebook exploration and execution:
  `node c:\Users\jerom\.cursor\extensions\googlecloudtools.datacloud-0.11.0-universal\mcp_servers\cli\mcp_proxy_bundle.js notebooks-cursor`
- **`visualization`**: DataCloud MCP server for telemetry and charting:
  `node c:\Users\jerom\.cursor\extensions\googlecloudtools.datacloud-0.11.0-universal\mcp_servers\cli\mcp_proxy_bundle.js visualization-cursor`
- **GCP Data Skills**: Use installed GCP skills (`bigquery`, `google-cloud-storage-basics`, `cloud-sql-*`, `spanner-data`, `dataflow`, `composer`, etc.) for cloud infrastructure.
