locals {
  base_tags = merge(
    {
      IaC  = "Terraform"
      Name = var.bucket_name
    },
    var.tags
  )
}

resource "aws_s3_bucket" "site" {
  bucket = var.bucket_name
  tags   = local.base_tags
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket = aws_s3_bucket.site.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_website_configuration" "site" {
  bucket = aws_s3_bucket.site.id

  index_document {
    suffix = var.index_document
  }

  dynamic "error_document" {
    for_each = var.error_document != "" ? [1] : []
    content {
      key = var.error_document
    }
  }
}

data "aws_iam_policy_document" "public_read" {
  statement {
    sid = "PublicReadGetObject"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = ["s3:GetObject"]

    resources = [
      "${aws_s3_bucket.site.arn}/*"
    ]
  }
}

resource "aws_s3_bucket_policy" "site" {
  bucket = aws_s3_bucket.site.id
  policy = data.aws_iam_policy_document.public_read.json

  depends_on = [aws_s3_bucket_public_access_block.site]
}
