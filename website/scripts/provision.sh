#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="$ROOT_DIR/terraform"
WITH_DEPLOY=0

usage() {
  cat <<'EOF'
Usage: ./scripts/provision.sh [--with-deploy]

Provisioning:
  - terraform init
  - terraform apply -auto-approve

Optional deployment (--with-deploy):
  - ./scripts/build_db.sh
  - ./scripts/deploy_s3.sh <bucket> <region>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-deploy)
      WITH_DEPLOY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v terraform >/dev/null 2>&1; then
  echo "Error: terraform not found" >&2
  exit 1
fi

if [[ ! -d "$TF_DIR" ]]; then
  echo "Error: terraform dir not found: $TF_DIR" >&2
  exit 1
fi

echo "==> Provisioning infrastructure (Terraform)"
terraform -chdir="$TF_DIR" init
terraform -chdir="$TF_DIR" apply -auto-approve

BUCKET_NAME="$(terraform -chdir="$TF_DIR" output -raw bucket_name)"
AWS_REGION="$(terraform -chdir="$TF_DIR" output -raw aws_region)"
WEBSITE_URL="$(terraform -chdir="$TF_DIR" output -raw website_url)"

echo "Provisioned bucket: $BUCKET_NAME"
echo "Region: $AWS_REGION"
echo "Website URL: $WEBSITE_URL"

if [[ "$WITH_DEPLOY" == "1" ]]; then
  echo "==> Building search DB"
  "$ROOT_DIR/scripts/build_db.sh"

  echo "==> Deploying website assets to S3"
  "$ROOT_DIR/scripts/deploy_s3.sh" "$BUCKET_NAME" "$AWS_REGION"
fi
