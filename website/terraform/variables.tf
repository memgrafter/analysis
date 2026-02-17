variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "S3 bucket name for the static site"
  type        = string
}

variable "index_document" {
  description = "Index document for S3 website hosting"
  type        = string
  default     = "index.html"
}

variable "error_document" {
  description = "Error document for S3 website hosting"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}

variable "cloudfront_price_class" {
  description = "CloudFront price class (PriceClass_100 is the lowest-cost/free-tier-eligible scope)"
  type        = string
  default     = "PriceClass_100"

  validation {
    condition     = contains(["PriceClass_All", "PriceClass_200", "PriceClass_100"], var.cloudfront_price_class)
    error_message = "cloudfront_price_class must be one of: PriceClass_100, PriceClass_200, PriceClass_All"
  }
}
