param(
    [Parameter(Position=0, Mandatory=$false)]
    [string]$RepoName = "graphrag-query",
    
    [Parameter(Position=1, Mandatory=$false)]
    [string]$Region = "us-west-1"
)

Write-Host "Checking for ECR Repository: $RepoName in $Region..." -ForegroundColor Cyan

# Check if repo exists
$repo = aws ecr describe-repositories --repository-names $RepoName --region $Region 2>$null | ConvertFrom-Json

if ($repo -eq $null) {
    Write-Host "Repository not found. Creating..." -ForegroundColor Yellow
    $newRepo = aws ecr create-repository --repository-name $RepoName --region $Region | ConvertFrom-Json
    $uri = $newRepo.repository.repositoryUri
    Write-Host "✅ Created: $uri" -ForegroundColor Green
} else {
    $uri = $repo.repositories[0].repositoryUri
    Write-Host "✅ Repository already exists at: $uri" -ForegroundColor Green
}

# Save URIs to .env for other scripts to use
if (Test-Path ".env") {
    # Remove existing ECR_REPO_URI if present
    $envContent = Get-Content ".env" | Where-Object { $_ -notmatch "^ECR_REPO_URI=" -and $_ -notmatch "^AWS_REGION=" }
    $envContent += "AWS_REGION=$Region"
    $envContent += "ECR_REPO_URI=$uri"
    $envContent | Set-Content ".env"
    Write-Host "Updated .env with ECR_REPO_URI and AWS_REGION" -ForegroundColor Gray
}
