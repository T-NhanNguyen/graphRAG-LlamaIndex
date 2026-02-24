# 🪣 GraphRAG S3 Cloud Backup Cheat Sheet

Quick reference for managing your knowledge graph backups and generic S3 uploads.

---

## 🛠️ Setup & Configuration

Ensure your `.env` file is configured:

```dotenv
# S3 Configuration
S3_BUCKET_NAME=investment-research-graphrag-tnhan
S3_DEFAULT_PREFIX=
S3_DB_VAULT_DIR=C:/Users/nhan/.graphrag/index-vault
```

Reload aliases in a new terminal session:

```powershell
. .\.graphrag-alias.ps1
```

---

## 🚀 Common Commands

### 1. Database Backups (The "Push")

Sync your local knowledge graph database to the S3 vault.

```powershell
# Backup the active database (defined in .env)
graphrag-push

# Backup a specific database by name
graphrag-push my-other-project
```

### 2. Manual File/Folder Uploads

Upload any project file or data folder to S3.

```powershell
# Upload a single file
.\s3.ps1 .\README.md

# Upload a folder to a specific S3 path
.\s3.ps1 .\my-docs "your-bucket-name" "optional/prefix"
```

---

## 📥 Restoration (The "Pull")

If you are on a new machine or need to recover data, use the AWS CLI to pull from your vault:

### Restore a Database:

```powershell
aws s3 sync s3://investment-research-graphrag-tnhan/backups/index-vault/investment-analysis C:/Users/nhan/.graphrag/index-vault/investment-analysis
```

### Restore Generic Files:

```powershell
aws s3 cp s3://investment-research-graphrag-tnhan/README.md ./README.md
```

---

## 🧹 Maintenance

### Local Cleanup

Keep your repository clean by removing local database fragments (after pushing to S3):

```powershell
# Remove old DuckDB fragments in the root
Remove-Item -Force .\investment-analysis, .\investment-analysis.wal

# Remove the old .DuckDB folder
Remove-Item -Recurse -Force .\.DuckDB
```

### AWS CLI Checks

````powershell
# List all files in your backup vault
aws s3 ls s3://investment-research-graphrag-tnhan/backups/index-vault/ --recursive

---

## 🐳 Docker Cloud Deployment (ECR)

Before your web app can run in the cloud, you need to push the "software" (Image) to AWS ECR.

### 1. Setup Repo (One time only)
Creates the repository in AWS and saves the URI to your `.env`.
```powershell
.\aws-ecr-setup.ps1 -Region "us-west-1"
````

### 2. Build & Push Image

Authenticates Docker and pushes the latest `graphrag-query` image.

```powershell
# Ensure you've built the image locally first:
docker compose build query

# Push to ECR:
.\docker-push-ecr.ps1
```

_(Note: The `graphrag-query` image is configured to automatically download your database from S3 on startup if `S3_BUCKET_NAME` is provided.)_

```

```
