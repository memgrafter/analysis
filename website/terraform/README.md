# Terraform (Static Site + HTTPS CDN)

Minimal Terraform for this repository's S3 static site with CloudFront HTTPS in front of it.

## Usage

1. Copy vars file:

```bash
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
```

2. Edit `terraform/terraform.tfvars` and set `bucket_name`.

3. Initialize and validate:

```bash
terraform -chdir=terraform init
terraform -chdir=terraform validate
```

4. Plan/apply:

```bash
terraform -chdir=terraform plan
terraform -chdir=terraform apply
```

## Notes

- No credentials or secrets are stored in these files.
- `terraform.tfvars` is ignored by git.
- This keeps the existing public S3 website origin for simple routing behavior.
- CloudFront is created with `viewer_protocol_policy = redirect-to-https` and the default CloudFront certificate.
- Default price class is `PriceClass_100` (lowest-cost/free-tier-eligible scope).
