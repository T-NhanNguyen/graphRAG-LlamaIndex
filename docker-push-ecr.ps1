# Load variables from .env
if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $name = $Matches[1].Trim()
            $value = $Matches[2].Trim()
            Set-Variable -Name $name -Value $value -Scope Script
        }
    }
}

if ([string]::IsNullOrWhiteSpace($ECR_REPO_URI)) {
    Write-Host "Error: ECR_REPO_URI not found in .env. Please run .\aws-ecr-setup.ps1 first." -ForegroundColor Red
    exit 1
}

$Region = if (![string]::IsNullOrWhiteSpace($AWS_REGION)) { $AWS_REGION } else { "us-east-1" }
$Registry = $ECR_REPO_URI.Split('/')[0]

Write-Host "Authenticating with ECR ($Registry)..." -ForegroundColor Cyan
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to authenticate with Docker. Ensure AWS CLI is configured." -ForegroundColor Red
    exit 1
}

Write-Host "Tagging image: graphrag-query:latest -> ${ECR_REPO_URI}:latest" -ForegroundColor Gray
docker tag graphrag-query:latest "${ECR_REPO_URI}:latest"

Write-Host "Pushing image to ECR..." -ForegroundColor Cyan
docker push "${ECR_REPO_URI}:latest"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Image pushed successfully to $ECR_REPO_URI" -ForegroundColor Green
} else {
    Write-Host "❌ Push failed." -ForegroundColor Red
}
