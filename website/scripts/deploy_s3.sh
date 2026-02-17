#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUCKET_NAME="${1:-}"
AWS_REGION="${2:-${AWS_REGION:-us-east-1}}"

if [[ -z "$BUCKET_NAME" ]]; then
  echo "Usage: ./scripts/deploy_s3.sh <bucket-name> [aws-region]" >&2
  exit 1
fi

for required in aws mktemp; do
  if ! command -v "$required" >/dev/null 2>&1; then
    echo "Error: required command not found: $required" >&2
    exit 1
  fi
done

STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/website-deploy.XXXXXX")"
cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

echo "Staging site assets into: $STAGE_DIR"

cp "$ROOT_DIR/index.html" "$STAGE_DIR/index.html"
cp -R "$ROOT_DIR/view" "$STAGE_DIR/view"
cp -R "$ROOT_DIR/assets" "$STAGE_DIR/assets"
cp -R "$ROOT_DIR/search" "$STAGE_DIR/search"

if [[ ! -f "$STAGE_DIR/search/manifest.json" ]]; then
  echo "Error: missing search/manifest.json in staged assets." >&2
  echo "Run ./scripts/build_db.sh before deploy." >&2
  exit 1
fi

echo "Syncing staged assets to s3://$BUCKET_NAME/ ..."
aws s3 sync "$STAGE_DIR/" "s3://$BUCKET_NAME/" --delete

echo "Done."
echo "S3 object URL: https://$BUCKET_NAME.s3.$AWS_REGION.amazonaws.com/"
echo "Website URL (if static website hosting enabled): http://$BUCKET_NAME.s3-website-$AWS_REGION.amazonaws.com"
