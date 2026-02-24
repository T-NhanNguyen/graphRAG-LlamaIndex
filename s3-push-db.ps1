param(
    [Parameter(Position=0, Mandatory=$false, HelpMessage="Database name to push")]
    [string]$DbName
)

# Load variables from .env
if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } | ForEach-Object {
        $name, $value = $_.Split('=', 2)
        $ExecutionContext.SessionState.PSVariable.Set($name.Trim(), $value.Trim())
    }
}

# Resolve target database
$targetDb = if (![string]::IsNullOrEmpty($DbName)) { $DbName } else { $GRAPHRAG_DATABASE }

if ([string]::IsNullOrEmpty($targetDb)) {
    Write-Host "Error: No database name provided and GRAPHRAG_DATABASE not set in .env" -ForegroundColor Red
    exit 1
}

if ([string]::IsNullOrEmpty($S3_DB_VAULT_DIR)) {
    Write-Host "Error: S3_DB_VAULT_DIR not set in .env" -ForegroundColor Red
    exit 1
}

$dbPath = Join-Path $S3_DB_VAULT_DIR $targetDb

if (-Not (Test-Path $dbPath)) {
    Write-Host "Error: Database folder not found at $dbPath" -ForegroundColor Red
    exit 1
}

# S3 Destination
$s3Dest = "s3://$S3_BUCKET_NAME/backups/index-vault/$targetDb"

Write-Host "Pushing database '$targetDb' to S3..." -ForegroundColor Cyan
Write-Host "Source: $dbPath" -ForegroundColor Gray
Write-Host "Destination: $s3Dest" -ForegroundColor Gray

# Sync the folder
aws s3 sync "$dbPath" "$s3Dest" --delete

if ($LASTEXITCODE -eq 0) {
    Write-Host "Database push successful!" -ForegroundColor Green
} else {
    Write-Host "Push failed. Check AWS credentials and bucket permissions." -ForegroundColor Red
}
