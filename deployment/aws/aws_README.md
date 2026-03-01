# AWS Deployment & Cloud Management

This directory contains infrastructure scripts for deploying the GraphRAG Query Engine to AWS and managing remote data synchronization.

## Overview

While Fly.io is used for lightweight deployment, this AWS suite provides a scalable alternative using **AWS App Runner** and **ECR** for the query engine, with **S3** serving as the centralized Knowledge Graph vault.

## Setup

1. **AWS CLI**: Install and run `aws configure`.
2. **Environment Variables**: Update `.env` with your AWS details:
   - `S3_BUCKET_NAME`
   - `AWS_REGION`
   - `ECR_REPO_URI`
3. **Repository Setup**: Run `.\aws-ecr-setup.ps1` once to initialize your cloud container registry.

## Usage

- **Deployment**: Use `.\docker-push-ecr.ps1` to upload your query engine image to ECR.
- **Syncing**: Use `graphrag-push` (from `.graphrag-alias.ps1`) to sync your local Knowledge Graph to S3.
- **Reference**: For specific S3 paths and restoration commands, see `S3_CHEATSHEET.md`.

## File Structure

- `Dockerfile.query`: AWS-optimized Docker configuration.
- `aws-ecr-setup.ps1`: One-time registry initializer.
- `docker-push-ecr.ps1`: Script to push local images to the cloud.
- `S3_CHEATSHEET.md`: Detailed S3 sync and recovery commands.
