param(
    [Parameter(Position=0, Mandatory=$true, HelpMessage="Path to the file or directory to upload")]
    [string]$LocalPath,

    [Parameter(Position=1, Mandatory=$false, HelpMessage="AWS S3 Bucket Name")]
    [string]$BucketName,

    [Parameter(Position=2, Mandatory=$false, HelpMessage="Optional S3 Prefix")]
    [string]$S3Prefix
)

# Load variables from .env if it exists
if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } | ForEach-Object {
        $name, $value = $_.Split('=', 2)
        $ExecutionContext.SessionState.PSVariable.Set($name.Trim(), $value.Trim())
    }
}

# Resolve final bucket name and prefix
if ([string]::IsNullOrEmpty($BucketName)) {
    $BucketName = if (![string]::IsNullOrEmpty($S3_BUCKET_NAME)) { $S3_BUCKET_NAME } else { "your-bucket-name" }
}

if ([string]::IsNullOrEmpty($S3Prefix)) {
    $S3Prefix = if (![string]::IsNullOrEmpty($S3_DEFAULT_PREFIX)) { $S3_DEFAULT_PREFIX } else { "" }
}

if ($BucketName -eq "your-bucket-name" -or [string]::IsNullOrEmpty($BucketName)) {
    Write-Host "Error: Please specify your actual S3 bucket name in your .env file or as the second argument." -ForegroundColor Red
    Write-Host "Usage: .\s3.ps1 <local-path> [bucket-name] [s3-prefix]"
    exit 1
}

if (-Not (Test-Path $LocalPath)) {
    Write-Host "Error: Local path '$LocalPath' does not exist." -ForegroundColor Red
    exit 1
}

# Remove leading slash from prefix if necessary
$S3Prefix = $S3Prefix.TrimStart('/')

$S3Uri = "s3://$BucketName"
if (![string]::IsNullOrEmpty($S3Prefix)) {
    $S3Uri = "$S3Uri/$S3Prefix"
}

Write-Host "Uploading from: $LocalPath" -ForegroundColor Cyan
Write-Host "To S3 Destination: $S3Uri" -ForegroundColor Cyan
Write-Host "Starting upload..." -ForegroundColor Cyan

if ((Get-Item $LocalPath) -is [System.IO.DirectoryInfo]) {
    aws s3 sync "$LocalPath" "$S3Uri"
} else {
    aws s3 cp "$LocalPath" "$S3Uri"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "Upload completed successfully." -ForegroundColor Green
} else {
    Write-Host "Error: Upload failed. Please check your AWS CLI configuration, bucket name, and permissions." -ForegroundColor Red
    exit 1
}
