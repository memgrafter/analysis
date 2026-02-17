locals {
  base_tags = merge(
    {
      IaC  = "Terraform"
      Name = var.bucket_name
    },
    var.tags
  )

  use_custom_domain = trimspace(var.custom_domain) != ""
}

check "custom_domain_requirements" {
  assert {
    condition     = !local.use_custom_domain || trimspace(var.route53_zone_name) != ""
    error_message = "route53_zone_name must be set when custom_domain is provided."
  }

  assert {
    condition     = !local.use_custom_domain || var.aws_region == "us-east-1"
    error_message = "aws_region must be us-east-1 when custom_domain is provided (CloudFront ACM requirement)."
  }
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

data "aws_route53_zone" "site" {
  count        = local.use_custom_domain ? 1 : 0
  name         = var.route53_zone_name
  private_zone = false
}

resource "aws_acm_certificate" "site" {
  count             = local.use_custom_domain ? 1 : 0
  domain_name       = var.custom_domain
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = merge(local.base_tags, { Name = "${var.bucket_name}-cert" })
}

resource "aws_route53_record" "cert_validation" {
  for_each = local.use_custom_domain ? {
    for dvo in aws_acm_certificate.site[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      type   = dvo.resource_record_type
      record = dvo.resource_record_value
    }
  } : {}

  zone_id         = data.aws_route53_zone.site[0].zone_id
  allow_overwrite = true
  name            = each.value.name
  type            = each.value.type
  ttl             = 60
  records         = [each.value.record]
}

resource "aws_acm_certificate_validation" "site" {
  count           = local.use_custom_domain ? 1 : 0
  certificate_arn = aws_acm_certificate.site[0].arn

  validation_record_fqdns = [
    for record in aws_route53_record.cert_validation : record.fqdn
  ]
}

resource "aws_cloudfront_distribution" "site" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "CDN for ${var.bucket_name}"
  default_root_object = var.index_document
  price_class         = var.cloudfront_price_class
  wait_for_deployment = false

  aliases = local.use_custom_domain ? [var.custom_domain] : []

  origin {
    domain_name = aws_s3_bucket_website_configuration.site.website_endpoint
    origin_id   = "s3-website-${var.bucket_name}"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = "s3-website-${var.bucket_name}"
    viewer_protocol_policy = "redirect-to-https"
    compress               = true

    allowed_methods = ["GET", "HEAD", "OPTIONS"]
    cached_methods  = ["GET", "HEAD"]

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn            = local.use_custom_domain ? aws_acm_certificate_validation.site[0].certificate_arn : null
    ssl_support_method             = local.use_custom_domain ? "sni-only" : null
    minimum_protocol_version       = local.use_custom_domain ? "TLSv1.2_2021" : "TLSv1"
    cloudfront_default_certificate = local.use_custom_domain ? false : true
  }

  tags = merge(local.base_tags, { Name = "${var.bucket_name}-cdn" })
}

resource "aws_route53_record" "site_alias_a" {
  count   = local.use_custom_domain ? 1 : 0
  zone_id = data.aws_route53_zone.site[0].zone_id
  name    = var.custom_domain
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.site.domain_name
    zone_id                = aws_cloudfront_distribution.site.hosted_zone_id
    evaluate_target_health = false
  }
}

resource "aws_route53_record" "site_alias_aaaa" {
  count   = local.use_custom_domain ? 1 : 0
  zone_id = data.aws_route53_zone.site[0].zone_id
  name    = var.custom_domain
  type    = "AAAA"

  alias {
    name                   = aws_cloudfront_distribution.site.domain_name
    zone_id                = aws_cloudfront_distribution.site.hosted_zone_id
    evaluate_target_health = false
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
