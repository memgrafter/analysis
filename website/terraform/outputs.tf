data "aws_region" "current" {}

output "bucket_name" {
  value = aws_s3_bucket.site.bucket
}

output "bucket_arn" {
  value = aws_s3_bucket.site.arn
}

output "website_endpoint" {
  value = aws_s3_bucket_website_configuration.site.website_endpoint
}

output "website_url" {
  value = "http://${aws_s3_bucket.site.bucket}.s3-website-${data.aws_region.current.name}.amazonaws.com"
}

output "aws_region" {
  value = data.aws_region.current.name
}

output "cloudfront_domain_name" {
  value = aws_cloudfront_distribution.site.domain_name
}

output "cloudfront_url" {
  value = "https://${aws_cloudfront_distribution.site.domain_name}"
}
