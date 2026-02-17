# Terraform (Static Site)

Minimal Terraform for this repository's S3 static site.

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
- This creates a public-read website bucket suitable for static hosting.
