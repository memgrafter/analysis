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
