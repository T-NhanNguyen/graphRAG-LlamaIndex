#!/bin/bash

# A simple script to upload files or directories to an AWS S3 bucket using .env variables.
# Ensure you have configured your AWS CLI (aws configure) before running this.

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

BUCKET_NAME="${S3_BUCKET_NAME:-your-bucket-name}"
DEFAULT_PREFIX="${S3_DEFAULT_PREFIX:-}"

usage() {
    echo "Usage: ./s3.sh <local-path> [bucket-name] [s3-prefix]"
    echo "Example: ./s3.sh ./data my-awesome-bucket project-data"
    echo ""
    echo "Defaults from .env:"
    echo "  Bucket: $BUCKET_NAME"
    echo "  Prefix: $DEFAULT_PREFIX"
}

if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
    exit 0
fi

LOCAL_PATH="$1"
# Override bucket/prefix if provided as arguments, else use .env/defaults
BUCKET_NAME="${2:-$BUCKET_NAME}"
S3_PREFIX="${3:-$DEFAULT_PREFIX}"

# Strip leading slash from prefix if exists to avoid double slashes in S3 URI
S3_PREFIX="${S3_PREFIX#/}"

if [ "$BUCKET_NAME" == "your-bucket-name" ] || [ -z "$BUCKET_NAME" ]; then
    echo "Error: Please specify your actual S3 bucket name in your .env file or as the second argument."
    usage
    exit 1
fi

if [ ! -e "$LOCAL_PATH" ]; then
    echo "Error: Local path '$LOCAL_PATH' does not exist."
    exit 1
fi

S3_URI="s3://${BUCKET_NAME}"
if [ -n "$S3_PREFIX" ]; then
    S3_URI="${S3_URI}/${S3_PREFIX}"
fi

echo "Uploading from: $LOCAL_PATH"
echo "To S3 Destination: $S3_URI"
echo "Starting upload..."

if [ -d "$LOCAL_PATH" ]; then
    # It's a directory, use sync to upload contents
    aws s3 sync "$LOCAL_PATH" "$S3_URI"
else
    # It's a file, use cp
    aws s3 cp "$LOCAL_PATH" "$S3_URI"
fi

if [ $? -eq 0 ]; then
    echo "Upload completed successfully."
else
    echo "Error: Upload failed. Please check your AWS CLI configuration, bucket name, and permissions."
    exit 1
fi
