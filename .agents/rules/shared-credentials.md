# Shared Credentials

- AWS shared credentials file: `C:\Projects\credentials` (env: `AWS_SHARED_CREDENTIALS_FILE`)
- SSH / PEM keys: `C:\Projects\*.pem` (e.g. `mushin_pgx.pem`)
- Never commit secrets, print credential file contents, or invent per-repo `.aws/` key stores
- Prefer `AWS_PROFILE` / `EC2_KEY_PATH` / `GITHUB_PAT` env overrides when needed
- AWS-hosted secrets (IAM ARNs, GitHub PATs, API tokens, OAuth client secrets) live in SSM Parameter Store, path convention `/<project>/<component>/<resource>`

Canonical docs: `C:\Projects\project_utility_scripts\CURSOR_DEV_RULES.md`
