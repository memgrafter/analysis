#!/usr/bin/env bash
set -euo pipefail

ASSET_DIR="${1:-dist}"
BUCKET_NAME="${2:-ml-llm-digests-9981ee3e6e}"
AWS_REGION="${AWS_REGION:-us-east-1}"

if [[ ! -d "$ASSET_DIR" ]]; then
  echo "Error: asset directory not found: $ASSET_DIR" >&2
  exit 1
fi

echo "Syncing $ASSET_DIR/ to s3://$BUCKET_NAME/ ..."
aws s3 sync "$ASSET_DIR/" "s3://$BUCKET_NAME/" --delete

echo "Done."
echo "S3 object URL: https://$BUCKET_NAME.s3.$AWS_REGION.amazonaws.com/index.html"
echo "Website URL (if static website hosting enabled): http://$BUCKET_NAME.s3-website-$AWS_REGION.amazonaws.com"
